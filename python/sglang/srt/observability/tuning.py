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
    def __init__(
        self,
        interval=0.0,
        context=None,
        *,
        clock=time.monotonic,
        emit=None,
        allocator_sampler=None,
        event_recorder=None,
    ):
        if not math.isfinite(interval) or interval < 0:
            raise ValueError(
                "tuning observation interval must be finite and nonnegative"
            )
        self.enabled = interval > 0
        self.interval = interval
        self.context = dict(context or {})
        self.clock = clock
        self.emit = emit or self._log
        self.allocator_sampler = allocator_sampler
        self.event_recorder = event_recorder
        self.start = clock() if self.enabled else 0.0
        self.counts = Counter()
        self.samples = {}

    @staticmethod
    def _log(record):
        logger.info("tuning_observation %s", json.dumps(record, sort_keys=True))

    def observe(self, event, **values):
        if self.event_recorder is not None:
            self.event_recorder.decision(event)
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
        record = dict(
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
        # One endpoint sample per emitted window, never per observed event.
        # It is not a window mean/peak and is absent when not requested.
        if self.allocator_sampler is not None:
            try:
                record["allocator_snapshot"] = self.allocator_sampler()
            except Exception:
                # Optional diagnostics must not turn an API/backend mismatch
                # into a serving failure. Never serialize exception contents.
                record["allocator_snapshot"] = {"status": "error"}
        self.emit(record)
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
    "prefill.decode_interval": {"remaining_interval", "waiting_requests", "chunk_pending"},
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


def cuda_allocator_snapshot(cuda=None):
    """Read CPU-side allocator counters without init, sync, flush or peak reset.

    Native PyTorch allocator scope only: private/custom pools and external
    allocations may be excluded. Reserved minus allocated is not a promise of
    reclaimable memory. Retry/OOM counters are cumulative, not window counts.
    Optional peaks share this one stats read and are since the last allocator
    peak reset by any caller, not this observation window or necessarily launch.
    """
    if cuda is None:
        import torch

        cuda = torch.cuda
    if not cuda.is_initialized():
        return {"status": "not_initialized"}
    backend = cuda.memory.get_allocator_backend()
    if backend != "native":
        # cudaMallocAsync does not give meaningful values for all these stats.
        return {"status": "unsupported_backend"}
    stats = cuda.memory_stats()
    keys = {
        "allocated_bytes": "allocated_bytes.all.current",
        "reserved_bytes": "reserved_bytes.all.current",
        "active_bytes": "active_bytes.all.current",
        "inactive_split_bytes": "inactive_split_bytes.all.current",
        "allocation_retries_total": "num_alloc_retries",
        "ooms_total": "num_ooms",
    }
    values = {
        name: stats[key]
        for name, key in keys.items()
        if type(stats.get(key)) is int and stats[key] >= 0
    }
    result = {
        "status": "ok" if len(values) == len(keys) else "partial",
        "backend": "native",
        "values": values,
    }
    peak_keys = {
        name: key.removesuffix(".current") + ".peak"
        for name, key in keys.items() if key.endswith(".current")
    }
    peaks = {
        name: stats[key] for name, key in peak_keys.items()
        if type(stats.get(key)) is int and stats[key] >= 0
    }
    if peaks:
        result["peaks"] = {
            "scope": "native_allocator_since_last_peak_reset",
            "values": peaks,
        }
    return result


def log_mega_moe_buffer(buffer, *, requested_token_cap, group_size, num_sms):
    """Describe a library-returned buffer using CPU metadata only.

    Called only on a SGLang buffer-cache miss when explicitly enabled. Tensor
    nbytes is the exposed buffer size, not total allocator/device memory. No
    addresses, process-group IDs, tensor reads or extra sizing queries.
    """
    record = {
        "schema_version": 1,
        "context": recorder.context,
        "emitted_at_unix": time.time(),
        "source": "sglang_cache_miss",
    }
    try:
        values = {
            "requested_token_cap": requested_token_cap,
            "resolved_token_cap": buffer.num_max_tokens_per_rank,
            "group_size": group_size,
            "num_experts": buffer.num_experts,
            "topk": buffer.num_topk,
            "hidden": buffer.hidden,
            "intermediate_hidden": buffer.intermediate_hidden,
            "shared_experts": buffer.num_shared_experts,
            "num_sms": num_sms,
            "buffer_bytes": buffer.buffer.nbytes,
        }
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise ValueError("Unavailable buffer metadata")
        mma = buffer.mma_type
        activation = buffer.activation
        record.update(
            status="ok",
            values=values,
            mma_type=(
                mma if mma in {"fp8xfp4", "mxf4xmxf4", "nvfp4xnvfp4", "fp8xfp8"}
                else "other"
            ),
            activation=(
                activation if activation in {"swiglu", "swigluoai", "situ"}
                else "other"
            ),
        )
    except Exception:
        record["status"] = "unavailable"
    logger.info("tuning_moe_buffer %s", json.dumps(record, sort_keys=True))


def configure(interval, *, allocator_sampler=None, **context):
    from sglang.srt.observability.performance_events import recorder as events

    # Preserve object identity: call sites import this single object.
    recorder.__init__(
        interval,
        dict(host=socket.gethostname(), pid=os.getpid(), **context),
        allocator_sampler=allocator_sampler,
        event_recorder=events,
    )
