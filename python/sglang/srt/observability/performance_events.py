"""Bounded operation events and condition episodes (one scheduler-thread producer).

Occurrence time is independent of delivery. The daemon writes small atomic pipe
records; queue/rate loss is reported cumulatively even when the producer is idle.
No tensors, readiness rechecks, request contents, IDs, or additional queue scans.
"""
from __future__ import annotations

import atexit
import json
import math
import os
import queue
import socket
import sys
import threading
import time
import uuid
from collections import Counter

PREFIX = "performance_event "
QUEUES = {"execution_wait", "bootstrap", "preallocation", "transfer", "retracted", "inflight"}
# A fixed vocabulary bounds cardinality, serialization size and consumer semantics.
FIELDS = {
    "queue.initial": {"queue", "depth", "high", "low"},
    "queue.pressure_enter": {"queue", "depth", "high", "low", "episode"},
    "queue.pressure_exit": {"queue", "depth", "high", "low", "episode", "duration_seconds", "peak_depth"},
    "arrival.window": {"requests", "input_tokens", "max_input_tokens", "window_seconds", "rate", "baseline_rate"},
    "arrival.surge_enter": {"rate", "baseline_rate", "episode", "ratio", "minimum_rate"},
    "arrival.surge_exit": {"rate", "baseline_rate", "episode", "duration_seconds", "peak_rate"},
    "cache.device_evict": {"component", "requested", "freed", "duration_seconds", "reason"},
    "cache.host_evict": {"component", "requested", "freed", "duration_seconds", "reason"},
    "cache.backup_complete": {"bytes", "ack_nodes"},
    "cache.loadback_complete": {"bytes", "ack_nodes"},
    "decode.retraction": {"requests", "aborted", "tokens_gained", "reason"},
    "decision.changed": {"domain", "previous", "current", "repeats", "duration_seconds"},
}


class EventRecorder:
    def __init__(self, enabled=False, *, context=None, clock=time.monotonic,
                 emit=None, max_pending=1024, max_per_second=200, high=128,
                 low=64, surge_min=32, surge_ratio=2.0):
        if not (1 <= max_pending <= 8192 and 1 <= max_per_second <= 2000
                and 0 <= low < high and surge_min > 0 and surge_ratio > 1):
            raise ValueError("Invalid performance event bounds")
        self.enabled = enabled
        self.clock = clock
        self.context = dict(context or {})
        self.context.setdefault("host", socket.gethostname())
        self.context.setdefault("pid", os.getpid())
        self.context.setdefault("incarnation", uuid.uuid4().hex)
        self.emit = emit
        self.max_per_second, self.high, self.low = max_per_second, high, low
        self.surge_min, self.surge_ratio = surge_min, surge_ratio
        self.seq = self.bucket_count = 0
        self.bucket = None
        self.loss = Counter()
        self.queues, self.decisions = {}, {}
        self.arrival_start = None
        self.arrivals = self.input_tokens = self.max_input = 0
        self.baseline = None
        self.surge = None
        self.episode = 0
        self.pending = queue.Queue(maxsize=max_pending)
        self.stop = threading.Event()
        self.thread = None
        if enabled and emit is None:
            self.thread = threading.Thread(target=self._write, name="performance-events", daemon=True)
            self.thread.start()

    def event(self, name, **values):
        if not self.enabled:
            return
        self.seq += 1
        # Malformed diagnostics must never break serving. No arbitrary repr().
        if name not in FIELDS or set(values) != FIELDS[name] or any(
            not (v is None or type(v) is str and len(v) <= 96 or
                 type(v) in (int, float) and math.isfinite(v))
            for v in values.values()
        ):
            self.loss["invalid_event"] += 1
            return
        now = self.clock()
        bucket = int(now)
        if bucket != self.bucket:
            self.bucket, self.bucket_count = bucket, 0
        self.bucket_count += 1
        if self.bucket_count > self.max_per_second:
            self.loss["rate_limited"] += 1
            return
        record = dict(schema_version=1, kind="event", event=name, seq=self.seq,
                      occurred_monotonic=now, occurred_unix=time.time(), values=values)
        if self.emit is not None:
            try:
                self.emit(dict(record, context=self.context, loss_total=dict(self.loss)))
            except Exception:
                self.loss["sink_errors"] += 1
        else:
            try:
                self.pending.put_nowait(record)
            except queue.Full:
                self.loss["queue_full"] += 1

    def queue_depth(self, name, depth):
        if not self.enabled:
            return
        if name not in QUEUES or type(depth) is not int or depth < 0:
            self.loss["invalid_queue"] += 1
            return
        now = self.clock()
        if name not in self.queues:
            self.queues[name] = None
            self.event("queue.initial", queue=name, depth=depth, high=self.high, low=self.low)
        active = self.queues[name]
        if active is None and depth >= self.high:
            self.episode += 1
            self.queues[name] = dict(start=now, peak=depth, episode=self.episode)
            self.event("queue.pressure_enter", queue=name, depth=depth,
                       high=self.high, low=self.low, episode=self.episode)
        elif active is not None:
            active["peak"] = max(active["peak"], depth)
            if depth <= self.low:
                self.event("queue.pressure_exit", queue=name, depth=depth,
                           high=self.high, low=self.low, episode=active["episode"],
                           duration_seconds=now-active["start"], peak_depth=active["peak"])
                self.queues[name] = None

    def arrival(self, input_tokens):
        if not self.enabled:
            return
        self.tick()
        self.arrivals += 1
        self.input_tokens += input_tokens
        self.max_input = max(self.max_input, input_tokens)

    def tick(self):
        if not self.enabled:
            return
        now = self.clock()
        if self.arrival_start is None:
            self.arrival_start = now
            return
        elapsed = now - self.arrival_start
        if elapsed < 1:
            return
        rate = self.arrivals / elapsed
        baseline = self.baseline
        self.event("arrival.window", requests=self.arrivals, input_tokens=self.input_tokens,
                   max_input_tokens=self.max_input, window_seconds=elapsed,
                   rate=rate, baseline_rate=baseline)
        if self.surge is None and baseline is not None and rate >= max(self.surge_min, baseline*self.surge_ratio):
            self.episode += 1
            self.surge = dict(start=now, peak=rate, baseline=baseline, episode=self.episode)
            self.event("arrival.surge_enter", rate=rate, baseline_rate=baseline,
                       episode=self.episode, ratio=self.surge_ratio, minimum_rate=self.surge_min)
        elif self.surge is not None:
            self.surge["peak"] = max(self.surge["peak"], rate)
            if rate <= max(self.surge_min / 2, self.surge["baseline"] * 1.25):
                self.event("arrival.surge_exit", rate=rate, baseline_rate=self.surge["baseline"],
                           episode=self.surge["episode"], duration_seconds=now-self.surge["start"],
                           peak_rate=self.surge["peak"])
                self.surge = None
        # Freeze the baseline during a surge; preserve previous baseline in each window.
        if self.surge is None:
            self.baseline = rate if baseline is None else 0.9*baseline + 0.1*rate
        self.arrival_start = now
        self.arrivals = self.input_tokens = self.max_input = 0

    def decision(self, name):
        if not self.enabled:
            return
        domain = ".".join(name.split(".")[:2]) if name.startswith(("graph.", "forward.")) else name.split(".")[0]
        if domain not in {"prefill", "decode", "graph.prefill", "forward.target", "forward.draft"}:
            return
        now = self.clock()
        old = self.decisions.get(domain)
        if old is not None and old[0] == name:
            old[2] += 1
            return
        self.event("decision.changed", domain=domain, previous=old[0] if old else None,
                   current=name, repeats=old[2] if old else 0,
                   duration_seconds=now-old[1] if old else 0)
        self.decisions[domain] = [name, now, 1]

    def _status(self, kind="heartbeat"):
        return dict(schema_version=1, kind=kind, last_event_seq=self.seq,
                    sampled_monotonic=self.clock(), emitted_unix=time.time(),
                    loss_total=dict(self.loss), pending=self.pending.qsize(),
                    active_queues={name: dict(state) for name, state in list(self.queues.items()) if state is not None},
                    settings=dict(max_per_second=self.max_per_second, high=self.high, low=self.low,
                                  surge_min=self.surge_min, surge_ratio=self.surge_ratio))

    def _write(self):
        number = 0
        next_status = 0.0
        def send(record):
            nonlocal number
            number += 1
            try:
                record = dict(record, context=self.context, delivery_seq=number)
                data = (PREFIX + json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
                fd = sys.stderr.fileno()
                # POSIX pipe atomicity only. Regular-file/Slurm capture needs native validation.
                if len(data) > min(4096, os.fpathconf(fd, "PC_PIPE_BUF")):
                    self.loss["oversize"] += 1
                elif os.write(fd, data) != len(data):
                    self.loss["sink_errors"] += 1
            except Exception:
                self.loss["sink_errors"] += 1
        while not self.stop.is_set() or not self.pending.empty():
            now = time.monotonic()
            if now >= next_status:
                send(self._status())
                next_status = now + 1
            try:
                item = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            send(item)
            self.pending.task_done()
        send(self._status("closed"))

    def close(self):
        if self.thread is not None:
            self.stop.set()
            self.thread.join(timeout=1)
            # Abrupt termination or blocked stderr can still lose the tail.
            if self.thread.is_alive():
                self.loss["close_timeout"] += 1


recorder = EventRecorder()
atexit.register(lambda: recorder.close())


def configure(context):
    recorder.close()
    recorder.__init__(
        os.getenv("SGLANG_PERFORMANCE_EVENTS", "0").lower() in {"1", "true"},
        context=context,
        max_per_second=int(os.getenv("SGLANG_PERFORMANCE_EVENTS_RATE", "200")),
        high=int(os.getenv("SGLANG_PERFORMANCE_EVENTS_QUEUE_HIGH", "128")),
        low=int(os.getenv("SGLANG_PERFORMANCE_EVENTS_QUEUE_LOW", "64")),
    )
