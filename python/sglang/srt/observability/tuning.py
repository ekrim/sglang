"""Opt-in CPU decision observations; no request identifiers or tensor reads.

One recorder per scheduler process. Only its scheduling/forward submission thread
may write it. Windows are event-driven (not a heartbeat); counts are predicate
observations, NOT distinct requests or blocked seconds. No hooks inside captured
model kernels: Python there counts capture, not replay.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import os
import socket
import time
from collections import Counter

logger = logging.getLogger(__name__)


class TuningRecorder:
    def __init__(self, interval=0.0, context=None, *, clock=time.monotonic, emit=None):
        if not math.isfinite(interval) or interval < 0:
            raise ValueError(
                "tuning observation interval must be finite and nonnegative"
            )
        self.enabled = interval > 0
        self.interval = interval
        self.context = dict(context or {})
        self.clock = clock
        self.emit = emit or self._log
        self.start = clock() if self.enabled else 0.0
        self.counts = Counter()
        self.samples = {}

    @staticmethod
    def _log(record):
        logger.info("tuning_observation %s", json.dumps(record, sort_keys=True))

    def observe(self, event, **values):
        if not self.enabled:
            return
        # Event/field names are literals at call sites. Values are CPU scalars;
        # refusing nonfinite values also makes emitted JSON portable.
        if event not in EVENTS or not set(values) <= EVENTS[event]:
            raise ValueError(f"Unknown tuning observation: {event}: {set(values)}")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) for v in values.values()
        ):
            raise ValueError("Tuning observations require finite CPU numbers")
        self.counts[event] += 1
        for name, value in values.items():
            key = (event, name)
            if key not in self.samples:
                self.samples[key] = dict(
                    count=0, sum=0, min=value, max=value, last=value
                )
            sample = self.samples[key]
            sample["count"] += 1
            sample["sum"] += value
            sample["min"] = min(sample["min"], value)
            sample["max"] = max(sample["max"], value)
            sample["last"] = value
        now = self.clock()
        if now - self.start >= self.interval:
            self.flush(now)

    def flush(self, now=None):
        if not self.enabled or not self.counts:
            return
        now = self.clock() if now is None else now
        self.emit(
            dict(
                schema_version=1,
                context=self.context,
                emitted_at_unix=time.time(),
                window_start_monotonic=self.start,
                window_end_monotonic=now,
                window_seconds=now - self.start,
                events=[
                    dict(
                        event=event,
                        observations=count,
                        values={
                            field: summary
                            for (key, field), summary in self.samples.items()
                            if key == event
                        },
                    )
                    for event, count in sorted(self.counts.items())
                ],
            )
        )
        self.counts.clear()
        self.samples.clear()
        self.start = now


# Bounded vocabulary: never use request IDs, prompts, layer IDs or shape strings.
EVENTS = {
    "graph.prefill.attention_backend": set(),
    "graph.prefill.context_coverage": {"context", "limit"},
    "graph.prefill.cp_bucket": set(),
    "graph.prefill.cp_inactive": set(),
    "graph.prefill.cp_shape": set(),
    "graph.prefill.dp_inactive": set(),
    "graph.prefill.dp_vote": set(),
    "graph.prefill.hidden_mode": set(),
    "graph.prefill.input_embeddings": set(),
    "graph.prefill.local_ineligible": set(),
    "graph.prefill.logprobs": set(),
    "graph.prefill.lora": set(),
    "graph.prefill.metadata_capacity": {"batch_size", "limit"},
    "graph.prefill.mixed_batch": set(),
    "graph.prefill.multimodal": set(),
    "graph.prefill.padding": {"tokens", "padded_tokens"},
    "graph.prefill.platform_prefix": set(),
    "graph.prefill.prefix_coverage": set(),
    "graph.prefill.replacement_embeddings": set(),
    "graph.prefill.request_capacity": {"batch_size", "limit"},
    "graph.prefill.target_verify": set(),
    "graph.prefill.token_coverage": {"tokens", "limit"},
    "prefill.full_capacity": {"required", "remaining"},
    "prefill.swa_capacity": {"required", "remaining"},
    "prefill.shared_capacity": {"full_required", "swa_required"},
    "prefill.swa_chunk_cap": {"original", "reduced"},
    "prefill.request_limit": set(),
    "prefill.request_slots": set(),
    "prefill.memory_budget": {"required_tokens", "remaining_tokens"},
    "prefill.input_budget": set(),
    "prefill.chunk_budget": set(),
    "prefill.replay_budget": {"replay_tokens", "chunk_tokens"},
    "prefill.chunk_alignment": set(),
    "prefill.chunk_slot": set(),
    "prefill.prefetch_pending": set(),
    "prefill.staging_pending": set(),
    "prefill.delayer": set(),
    "prefill.admitted": {
        "extend_tokens",
        "replay_tokens",
        "output_reserve_tokens",
        "input_remaining",
        "chunk_remaining",
    },
    "decode.handshake_pending": set(),
    "decode.metadata_slots": set(),
    "decode.request_slots": set(),
    "decode.mamba_capacity": set(),
    "decode.hisparse_capacity": set(),
    "decode.host_capacity": set(),
    "decode.kv_budget": {"full_required", "full_budget", "swa_required", "swa_budget"},
    "decode.preallocated": {"new_tokens", "prefix_tokens"},
    "forward.target.decode_runner": {"batch_size"},
    "forward.target.prefill_graph": {"batch_size"},
    "forward.target.eager": {"batch_size"},
    "forward.target.split_prefill": {"batch_size"},
    "forward.draft.decode_runner": {"batch_size"},
    "forward.draft.prefill_graph": {"batch_size"},
    "forward.draft.eager": {"batch_size"},
    "forward.draft.split_prefill": {"batch_size"},
}

recorder = TuningRecorder()
atexit.register(recorder.flush)


def configure(interval, **context):
    # Preserve object identity: call sites import this single object.
    recorder.__init__(
        interval, dict(host=socket.gethostname(), pid=os.getpid(), **context)
    )
