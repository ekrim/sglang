"""Sample existing SPF/HRRN passes without new scans, readiness calls or GPU work.

Queue ages are current episodes. Candidate outcomes are visit-weighted. No
sample means unknown (an earlier gate may prevent sorting), not an empty queue.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter

logger = logging.getLogger(__name__)
POLICIES = {"shortest-prefill-first", "hrrn"}
REASONS = {
    "pass_budget_stop", "lora_skip", "pooling_skip", "request_slots_stop",
    "prefetch_pending", "prefetch_reissued", "staging_pending", "admitted",
    "not_admitted_continue", "not_admitted_no_token", "not_admitted_other",
}


def age_of(req, now):
    ts = getattr(getattr(req, "time_stats", None), "wait_queue_entry_time", None)
    if type(ts) in (int, float) and math.isfinite(ts) and 0 < ts <= now:
        return now - ts
    return None


def work_of(req, policy):
    # CPU-only policy inputs; do not touch cache or recheck readiness.
    work = len(req.origin_input_ids) - req.num_matched_prefix_tokens
    return max(1, work + len(req.output_ids)) if policy == "shortest-prefill-first" else max(0, work)


def band(value, boundaries):
    return next((str(b) for b in boundaries if value < b), "inf")


class PolicyPass:
    def __init__(self, policy, now, queue_count, context, emit):
        self.policy, self.now, self.emit = policy, now, emit
        self.record = dict(schema_version=1, policy=policy, context=dict(context),
                           sampled_at_unix=time.time(), sampled_at_perf_counter=now,
                           queue_count=queue_count)
        self.queue = Counter()
        self.age_bands, self.work_bands = Counter(), Counter()
        self.oldest = None
        self.visits = {}

    def queue_key(self, req, work, deferred):
        age = age_of(req, self.now)
        self.queue["keys_seen"] += 1
        self.queue["prefix_deferred"] += int(deferred)
        self.queue["output_bearing"] += int(bool(req.output_ids))
        self.queue["already_forwarded"] += int(getattr(req.time_stats, "forward_entry_time", 0) > 0)
        self.work_bands[band(work, (1, 1024, 16384, 65536))] += 1
        if age is None:
            self.queue["age_unknown"] += 1
        else:
            self.queue["age_known"] += 1
            self.age_bands[band(age, (1, 10, 60, 300, 1200))] += 1
            if self.oldest is None or age > self.oldest["age_seconds"]:
                self.oldest = dict(age_seconds=age, work_tokens=work, prefix_deferred=bool(deferred))

    def head(self, req):
        values = dict(work_tokens=work_of(req, self.policy))
        age = age_of(req, self.now)
        if age is not None:
            values["age_seconds"] = age
        self.record["selected_head"] = values

    def visit(self, req, reason):
        assert reason in REASONS
        v = self.visits.setdefault(reason, dict(count=0, age_known=0, age_unknown=0,
                                                work_tokens_sum=0, work_tokens_max=0))
        v["count"] += 1
        work = work_of(req, self.policy)
        v["work_tokens_sum"] += work
        v["work_tokens_max"] = max(v["work_tokens_max"], work)
        age = age_of(req, self.now)
        if age is None:
            v["age_unknown"] += 1
        else:
            v["age_known"] += 1
            v["age_seconds_sum"] = v.get("age_seconds_sum", 0) + age
            v["age_seconds_max"] = max(v.get("age_seconds_max", 0), age)
            if "oldest_episode" not in v or age > v["oldest_episode"]["age_seconds"]:
                v["oldest_episode"] = dict(age_seconds=age, work_tokens=work)
        if reason == "admitted":
            v["extend_tokens_sum"] = v.get("extend_tokens_sum", 0) + req.extend_len

    def reservation(self, *, budget, applicable, reserved=None, cap=None):
        values = dict(budget_tokens=budget, policy_cap_applicable=bool(applicable),
                      policy_cap_returned=cap is not None)
        if reserved is not None:
            values["reserved_tokens"] = reserved
        if cap is not None:
            values["cap_tokens"] = cap
        self.record["continuation"] = values

    def continuation_admitted(self, req, admitted):
        self.record.setdefault("continuation", {}).update(
            admitted=bool(admitted), admitted_tokens=req.extend_len if admitted else 0)

    def finish(self):
        self.record.update(queue=dict(self.queue), age_bands=dict(self.age_bands),
                           work_bands=dict(self.work_bands), visits=self.visits)
        if self.oldest is not None:
            self.record["oldest_episode"] = self.oldest
        try:
            self.emit(self.record)
        except Exception:
            # Optional log sink failure must not change serving decisions.
            logger.warning("tuning_policy_pass emission failed")


class PolicySampler:
    def __init__(self, interval=0.0, *, context=None, clock=time.perf_counter, emit=None):
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("Policy sampling interval must be finite and nonnegative")
        self.interval, self.clock = interval, clock
        self.enabled = interval > 0
        self.context = context if context is not None else {}
        self.emit = emit or self._log
        self.next_at = 0.0

    @staticmethod
    def _log(record):
        logger.info("tuning_policy_pass %s", json.dumps(record, sort_keys=True, allow_nan=False))

    def begin(self, policy, queue_count):
        if not self.enabled or policy not in POLICIES:
            return None
        now = self.clock()
        if now < self.next_at:
            return None
        self.next_at = now + self.interval
        return PolicyPass(policy, now, queue_count, self.context, self.emit)
