"""Bounded PD ownership diagnostics, driven by existing scheduler transitions.

No tensors, readiness polls, request contents or cross-process clock arithmetic.
One scheduler-thread writer owns the ledger. A bounded sink thread only serializes
immutable snapshots. This is an opt-in state diagnostic, not a span exporter.
"""

from __future__ import annotations

import atexit
import hashlib
import heapq
import json
import logging
import math
import os
import queue
import socket
import threading
import time
import uuid
import weakref
from collections import Counter, OrderedDict, deque

logger = logging.getLogger(__name__)


class PDLifecycleRecorder:
    def __init__(
        self,
        interval=0.0,
        *,
        context=None,
        max_requests=4096,
        max_events=1024,
        sample_every=1,
        oldest=8,
        history=12,
        clock=time.monotonic,
        emit=None,
    ):
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("PD lifecycle interval must be finite and nonnegative")
        if any(
            type(v) is not int or v <= 0
            for v in (max_requests, max_events, sample_every, oldest, history)
        ):
            raise ValueError("PD lifecycle bounds must be positive integers")
        self.enabled = interval > 0
        self.interval, self.clock = interval, clock
        self.max_requests, self.max_events = max_requests, max_events
        self.sample_every, self.oldest, self.history_limit = (
            sample_every,
            oldest,
            history,
        )
        self.context = dict(context or {})
        self.context.update(
            host=socket.gethostname(), pid=os.getpid(), incarnation=uuid.uuid4().hex
        )
        self.start = clock() if self.enabled else 0.0
        self.next_summary = self.start
        self.entries = OrderedDict()
        self.owners = {}
        self.pools = {}
        self.events = []
        self.counts = Counter()
        self.loss = Counter()
        self.seq = 0
        self.emit = emit
        self._sink = None
        self._thread = None
        if self.enabled and emit is None:
            self._sink = queue.Queue(maxsize=4)
            self._thread = threading.Thread(
                target=self._write, name="pd-lifecycle", daemon=True
            )
            self._thread.start()

    def _write(self):
        while True:
            item = self._sink.get()
            try:
                if item is None:
                    return
                logger.info("pd_lifecycle %s", json.dumps(item, separators=(",", ":")))
            except Exception:
                self.loss["sink_errors"] += 1
            finally:
                self._sink.task_done()

    @staticmethod
    def identity(req):
        rid = getattr(req, "rid", None)
        if not isinstance(rid, str):
            return None
        # Client-supplied IDs cannot make a bounded ledger unbounded in bytes.
        rid = (
            rid
            if len(rid) <= 128
            else "sha256:" + hashlib.sha256(rid.encode()).hexdigest()
        )
        room = getattr(req, "bootstrap_room", None)
        if room is not None and type(room) is not int:
            return None
        return (rid, None if room is None else str(room))

    def bind(self, req):
        if self.enabled:
            req.time_stats._pd_lifecycle_req = weakref.ref(req)
            self.transition(req, "arrival")

    def _entry(self, req):
        key = self.identity(req)
        if key is None:
            self.loss["unsupported_identity"] += 1
            return None, None
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_requests:
                self.loss["untracked_transitions"] += 1
                return key, None
            now = self.clock()
            entry = dict(
                rid=key[0],
                room=key[1],
                stage="unknown",
                stage_since=now,
                first_seen=now,
                last_progress=now,
                facts={},
                resources={},
                history=deque(maxlen=self.history_limit),
                history_lost=0,
                transitions=0,
                terminal=False,
                arrival_observed=False,
                sampled=int.from_bytes(
                    hashlib.blake2b(str(key).encode(), digest_size=8).digest(), "big"
                )
                % self.sample_every
                == 0,
            )
            self.entries[key] = entry
        return key, entry

    def _record(self, key, entry, event, values):
        now = self.clock()
        self.seq += 1
        self.counts[event] += 1
        entry["transitions"] += 1
        if (
            event != "readiness"
            or values.get("metadata_ready") is True
            or "local_transport" in values
        ):
            entry["last_progress"] = now
        if len(entry["history"]) == self.history_limit:
            entry["history_lost"] += 1
        row = dict(seq=self.seq, monotonic=now, event=event, **values)
        entry["history"].append(row)
        if entry["sampled"] or event in {
            "aborted",
            "cancel_requested",
            "completion",
            "cleanup_deferred",
        }:
            if len(self.events) < self.max_events:
                record = dict(rid=key[0], room=key[1], **row)
                if event in {"aborted", "cancel_requested", "cleanup_deferred"}:
                    record["diagnostic"] = self._view(entry, now)
                self.events.append(record)
            else:
                self.loss["event_records_dropped"] += 1

    @staticmethod
    def _valid_facts(facts):
        return len(facts) <= 32 and all(
            isinstance(k, str)
            and len(k) <= 64
            and type(v) in (str, int, float, bool, type(None))
            and (not isinstance(v, str) or len(v) <= 128)
            and (not isinstance(v, float) or math.isfinite(v))
            for k, v in facts.items()
        )

    def transition(self, req, stage, **facts):
        if not self.enabled:
            return
        key, entry = self._entry(req)
        if entry is None:
            return
        # Call sites supply a closed vocabulary and CPU scalar facts only.
        if (
            not self._valid_facts(facts)
            or len(entry["facts"].keys() | facts.keys()) > 32
        ):
            self.loss["unsupported_value"] += 1
            return
        if entry["stage"] == stage and all(
            entry["facts"].get(k) == v for k, v in facts.items()
        ):
            return
        entry["stage"] = stage
        entry["stage_since"] = self.clock()
        entry["facts"].update(facts)
        entry["arrival_observed"] |= stage == "arrival"
        entry["terminal"] |= stage in {"completion", "aborted"}
        self._record(key, entry, stage, facts)

    def observe(self, req, **facts):
        """Independent readiness facts; do not erase the lifecycle stage."""
        if not self.enabled:
            return
        key, entry = self._entry(req)
        if entry is None:
            return
        if (
            not self._valid_facts(facts)
            or len(entry["facts"].keys() | facts.keys()) > 32
        ):
            self.loss["unsupported_value"] += 1
            return
        if all(entry["facts"].get(k) == v for k, v in facts.items()):
            return
        # Budget values can change every scan. Retain the latest CPU scalars,
        # but only a change of the blocking gate is a transition record.
        if (
            facts.get("blocked_gate") is not None
            and entry["facts"].get("blocked_gate") == facts["blocked_gate"]
        ):
            entry["facts"].update(facts)
            return
        entry["facts"].update(facts)
        self._record(key, entry, "readiness", facts)

    def register_pool(self, pool, kind):
        pool_id = id(pool)
        previous = self.pools.get(pool_id)
        if previous is not None and previous[0]() is pool:
            return True
        if previous is not None:
            # Object identity reuse cannot silently attach old owners to a new pool.
            stale = [key for key in self.owners if key[0] == pool_id]
            self.loss["retired_pool_ownership_gaps"] += len(stale)
            for key in stale:
                self.owners.pop(key)
        if len(self.pools) >= 16 and previous is None:
            self.loss["unsupported_pools"] += 1
            return False
        try:
            self.pools[pool_id] = (weakref.ref(pool), kind)
        except TypeError:
            self.loss["unsupported_pools"] += 1
            return False
        return True

    def acquire(self, pool, slot, req, kind):
        if not self.enabled:
            return
        key, entry = self._entry(req)
        if entry is None:
            self.loss["untracked_acquisitions"] += 1
            return
        pool_id = id(pool)
        if not self.register_pool(pool, kind):
            return
        identity = (pool_id, slot)
        if identity in self.owners:
            if self.owners[identity][0] == key:
                return  # Chunk continuation reuses the existing row.
            self.loss["ownership_conflicts"] += 1
        acquisition_seq = self.seq + 1
        if len(self.owners) >= 2 * self.max_requests and identity not in self.owners:
            self.loss["owner_records_dropped"] += 1
            return
        self.owners[identity] = (key, acquisition_seq)
        entry["resources"][kind] = dict(slot=slot, acquisition_seq=acquisition_seq)
        self._record(
            key,
            entry,
            "reservation_acquired",
            dict(resource=kind, slot=slot, acquisition_seq=acquisition_seq),
        )

    def release(self, pool, slot):
        if not self.enabled:
            return
        owner = self.owners.pop((id(pool), slot), None)
        if owner is None:
            self.loss["unattributed_releases"] += 1
            return
        key, acquisition_seq = owner
        if key is None:  # Session retained the physical row after Req completion.
            self.counts["session_row_released"] += 1
            return
        entry = self.entries.get(key)
        if entry is None:
            self.loss["release_history_missing"] += 1
            return
        kind = self.pools[id(pool)][1]
        entry["resources"].pop(kind, None)
        self._record(
            key,
            entry,
            "resource_released",
            dict(resource=kind, slot=slot, acquisition_seq=acquisition_seq),
        )

    def transfer_to_session(self, pool, slot):
        if not self.enabled:
            return
        identity = (id(pool), slot)
        owner = self.owners.get(identity)
        if owner is None or owner[0] is None:
            self.loss["untracked_session_transfer"] += 1
            return
        key, acquisition_seq = owner
        entry = self.entries[key]
        entry["resources"].pop("request_slot", None)
        self.owners[identity] = (None, acquisition_seq)
        self._record(
            key,
            entry,
            "ownership_transferred",
            dict(
                resource="request_slot",
                slot=slot,
                owner="session",
                acquisition_seq=acquisition_seq,
            ),
        )

    def reset_pool(self, pool):
        if not self.enabled:
            return
        # The bounded ledger is scanned only on an explicit pool reset.
        for pool_id, slot in list(self.owners):
            if pool_id == id(pool):
                self.release(pool, slot)
        self.loss["pool_resets"] += 1

    def _view(self, entry, now):
        return dict(
            rid=entry["rid"],
            room=entry["room"],
            stage=entry["stage"],
            stage_age_seconds=max(0, now - entry["stage_since"]),
            progress_age_seconds=max(0, now - entry["last_progress"]),
            resources=dict(entry["resources"]),
            facts=dict(entry["facts"]),
            terminal=entry["terminal"],
            history=list(entry["history"]),
            history_lost=entry["history_lost"],
            incomplete_history=not entry["arrival_observed"]
            or entry["history_lost"] > 0,
        )

    def summary(
        self, queues, *, force=False, unsupported=(), scheduler_run_batch_count=None
    ):
        if not self.enabled:
            return
        now = self.clock()
        if not force and now < self.next_summary:
            return
        self.next_summary = now + self.interval
        stages = Counter()
        oldest = {}
        # Bounded by max_requests, never a scan of the serving queues.
        for entry in self.entries.values():
            stage = entry["stage"]
            stages[stage] += 1
            oldest[stage] = max(oldest.get(stage, 0), now - entry["stage_since"])
        holders = heapq.nsmallest(
            self.oldest,
            (e for e in self.entries.values() if e["resources"]),
            key=lambda e: e["last_progress"],
        )
        waiters = heapq.nsmallest(
            self.oldest,
            (
                e
                for e in self.entries.values()
                if not e["resources"] and not e["terminal"]
            ),
            key=lambda e: e["stage_since"],
        )
        pools = []
        for pool_id, (pool_ref, kind) in self.pools.items():
            pool = pool_ref()
            if pool is None:
                pools.append(
                    dict(resource=kind, status="pool_retired_release_history_unknown")
                )
                continue
            free = len(pool.free_slots)
            capacity = pool.size if kind == "metadata" else pool._alloc_size - 1
            known = sum(
                p == pool_id and owner[0] is not None
                for (p, _), owner in self.owners.items()
            )
            retained = sum(
                p == pool_id and owner[0] is None
                for (p, _), owner in self.owners.items()
            )
            pools.append(
                dict(
                    resource=kind,
                    capacity=capacity,
                    available=free,
                    tracked_owners=known,
                    session_retained=retained,
                    unattributed_occupied=capacity - free - known - retained,
                    reserved_padding_rows=0 if kind == "metadata" else 1,
                )
            )
        record = dict(
            schema_version=1,
            context=self.context,
            seq=self.seq,
            snapshot_monotonic=now,
            emitted_unix=time.time(),
            clock_scope="process_local_monotonic",
            interval_seconds=self.interval,
            queues=dict(queues),
            scheduler_run_batch_count=scheduler_run_batch_count,
            scheduler_run_batch_scope="includes_prebuilt_placeholders; zero_delta_rules_out_run_batch",
            pools=pools,
            tracked_stages=dict(stages),
            oldest_stage_seconds=oldest,
            oldest_holders=[self._view(e, now) for e in holders],
            oldest_waiters=[self._view(e, now) for e in waiters],
            terminal=[
                self._view(e, now) for e in list(self.entries.values()) if e["terminal"]
            ][: self.oldest],
            terminal_snapshot_omitted=max(
                0, sum(e["terminal"] for e in self.entries.values()) - self.oldest
            ),
            transitions=dict(self.counts),
            events=self.events,
            loss_total=dict(self.loss),
            sample_every=self.sample_every,
            bounds=dict(
                requests=self.max_requests,
                events=self.max_events,
                oldest=self.oldest,
                history=self.history_limit,
            ),
            unsupported=list(unsupported),
            freshness="scheduler_observed_at_snapshot; silence_is_unknown",
        )
        self.events = []
        self.counts.clear()
        if self.emit is not None:
            self.emit(record)
        else:
            try:
                self._sink.put_nowait(record)
            except queue.Full:
                self.loss["batches_dropped"] += 1
                self.loss["event_records_dropped"] += len(record["events"])
        for key, entry in list(self.entries.items()):
            if entry["terminal"] and not entry["resources"]:
                del self.entries[key]

    def close(self):
        if self.enabled:
            self.summary(
                {}, force=True, unsupported=("shutdown_queue_counts_unavailable",)
            )
        self.enabled = False
        if self._thread is not None:
            try:
                self._sink.put(None, timeout=1)
                self._thread.join(timeout=1)
                if self._thread.is_alive():
                    self.loss["shutdown_incomplete"] += 1
            except queue.Full:
                self.loss["shutdown_incomplete"] += 1


recorder = PDLifecycleRecorder()
atexit.register(recorder.close)


def from_stats(stats, stage):
    if recorder.enabled:
        ref = getattr(stats, "_pd_lifecycle_req", None)
        req = ref() if ref is not None else None
        if req is not None:
            recorder.transition(req, stage)


def configure(**context):
    from sglang.srt.environ import envs

    interval = envs.SGLANG_PD_LIFECYCLE_INTERVAL.get()
    if context.get("role") not in {"prefill", "decode"}:
        interval = 0.0
    if recorder.enabled:
        recorder.close()
    recorder.__init__(
        interval,
        context=context,
        max_requests=envs.SGLANG_PD_LIFECYCLE_MAX_REQUESTS.get(),
        max_events=envs.SGLANG_PD_LIFECYCLE_MAX_EVENTS.get(),
        sample_every=envs.SGLANG_PD_LIFECYCLE_SAMPLE_EVERY.get(),
    )


def scheduler_heartbeat(scheduler, now):
    """O(1) live queue counts, independent of forward and fully-idle reporting.

    Priority breakdowns and host subqueues are deliberately outside this narrow
    gauge freshness contract; they retain their existing producers.
    """
    role = getattr(
        scheduler.disaggregation_mode, "value", scheduler.disaggregation_mode
    )
    if role not in {"prefill", "decode"}:
        return
    reporter = scheduler.metrics_reporter
    metrics = reporter.current_scheduler_metrics_enabled
    if not recorder.enabled and not metrics:
        return
    if now < getattr(scheduler, "_pd_next_queue_snapshot", 0.0):
        return
    scheduler._pd_next_queue_snapshot = now + (
        recorder.interval if recorder.enabled else 10.0
    )
    queues = {"execution_wait": len(scheduler.waiting_queue)}
    if role == "decode":
        pre = scheduler.disagg_decode_prealloc_queue
        transfer = scheduler.disagg_decode_transfer_queue
        queues.update(
            preallocation=len(pre.queue),
            transfer=len(transfer.queue),
            retracted=len(pre.retracted_queue),
            deferred_release=transfer.num_pending_deferred_releases(),
            held_rebootstrap=len(pre.held_rebootstrap_reqs),
        )
        request_pool = pre.req_to_token_pool
        metadata_pool = pre.req_to_metadata_buffer_idx_allocator
    else:
        pre = scheduler.disagg_prefill_bootstrap_queue
        queues.update(
            bootstrap=len(pre.queue),
            inflight=len(scheduler.disagg_prefill_inflight_queue),
            pending_chunk_sends=len(scheduler.disagg_prefill_pending_chunk_rids),
        )
        request_pool = scheduler.req_to_token_pool
        metadata_pool = scheduler.req_to_metadata_buffer_idx_allocator
    if metrics:
        scheduler.metrics_collector.log_pd_queue_snapshot(queues)
    if recorder.enabled:
        recorder.register_pool(request_pool, "request_slot")
        recorder.register_pool(metadata_pool, "metadata")
        recorder.summary(
            queues,
            scheduler_run_batch_count=scheduler.forward_ct,
            unsupported=(
                "physical_KV_bytes_and_shared_prefix_ownership",
                "wire_byte_progress_unobserved",
                "PP_CP_PD_multiplex_and_concurrent_writer_paths_not_qualified",
                "shutdown_or_permanent_sink_failure_can_prevent_final_loss_report",
            ),
        )
