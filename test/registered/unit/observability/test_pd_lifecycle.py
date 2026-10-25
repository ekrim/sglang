"""Dependency-free ledger checks; native transition tests are a separate gate."""

import ast
import importlib.util
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

if __name__ != "__main__":
    from sglang.test.ci.ci_register import register_cpu_ci

    register_cpu_ci(est_time=1, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / "python/sglang/srt/observability/pd_lifecycle.py"
spec = importlib.util.spec_from_file_location("pd_lifecycle_under_test", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Pool:
    def __init__(self):
        self.size = 1
        self.free_slots = []
        self._alloc_size = 2


class Request:
    def __init__(self, rid, room):
        self.rid, self.bootstrap_room = rid, room
        self.time_stats = SimpleNamespace()


class TestPDLedger(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.records = []
        self.rec = module.PDLifecycleRecorder(
            1, clock=lambda: self.now, emit=self.records.append
        )

    def test_actual_metadata_allocator_zero_exhaustion_release_reuse(self):
        # Compile the production allocator unchanged; it has no GPU dependency.
        # This local check is not a substitute for the full native module suite.
        source = ast.parse(
            (ROOT / "python/sglang/srt/disaggregation/utils.py").read_text()
        )
        cls = next(
            n
            for n in source.body
            if isinstance(n, ast.ClassDef) and n.name == "ReqToMetadataIdxAllocator"
        )
        namespace = dict(
            deque=deque, Optional=__import__("typing").Optional, pd_lifecycle=self.rec
        )
        exec(
            compile(
                ast.Module(body=[cls], type_ignores=[]),
                "production_metadata_allocator",
                "exec",
            ),
            namespace,
        )
        pool = namespace[cls.name](1)
        req = Request("owner", 9007199254740993)
        self.rec.bind(req)
        self.assertEqual(pool.alloc(req), 0)
        self.assertIsNone(pool.alloc(Request("waiter", 2)))
        self.now += 5
        self.rec.summary({"preallocation": 1})
        row = self.records[-1]
        self.assertEqual(row["oldest_holders"][0]["room"], "9007199254740993")
        self.assertEqual(row["pools"][0]["unattributed_occupied"], 0)
        gen = row["oldest_holders"][0]["resources"]["metadata"]["acquisition_seq"]
        pool.free(0)
        self.rec.transition(req, "completion")
        other = Request("replacement", 2)
        self.rec.bind(other)
        self.assertEqual(pool.alloc(other), 0)
        self.rec.summary({}, force=True)
        holder = self.records[-1]["oldest_holders"][0]
        self.assertEqual(holder["rid"], "replacement")
        self.assertGreater(holder["resources"]["metadata"]["acquisition_seq"], gen)

    def test_waiting_only_heartbeat_retains_age_and_does_not_count_polls_as_progress(
        self,
    ):
        req = Request("stalled", 2**63 - 1)
        self.rec.bind(req)
        self.rec.transition(req, "handshake_ready")
        for i in range(1000):
            self.now += 0.01
            self.rec.observe(req, blocked_gate="request_slots", full_budget=i)
            self.rec.summary({"preallocation": 1})
        self.rec.summary({"preallocation": 1}, force=True)
        row = self.records[-1]["oldest_waiters"][0]
        self.assertGreater(row["stage_age_seconds"], 9)
        self.assertGreater(row["progress_age_seconds"], 9)
        self.assertEqual(
            sum(r["transitions"].get("readiness", 0) for r in self.records), 1
        )

    def test_cancellation_preserves_holder_until_actual_release(self):
        pool = Pool()
        req = Request("cancelled", 3)
        self.rec.bind(req)
        self.rec.acquire(pool, 1, req, "request_slot")
        self.rec.transition(req, "aborted")
        self.rec.transition(req, "cleanup_deferred", required_acks=1)
        self.rec.summary({"deferred_release": 1})
        self.assertTrue(self.records[-1]["oldest_holders"][0]["terminal"])
        self.assertEqual(len(self.rec.entries), 1)
        pool.free_slots.append(1)
        self.rec.release(pool, 1)
        self.rec.summary({}, force=True)
        self.assertEqual(len(self.rec.entries), 0)
        self.assertEqual(self.records[-1]["pools"][0]["available"], 1)

    def test_session_retention_is_not_a_finished_request_leak(self):
        pool = Pool()
        req = Request("session", 4)
        self.rec.bind(req)
        self.rec.acquire(pool, 1, req, "request_slot")
        self.rec.transfer_to_session(pool, 1)
        self.rec.transition(req, "completion")
        self.rec.summary({})
        state = self.records[-1]["pools"][0]
        self.assertEqual(state["session_retained"], 1)
        self.assertEqual(state["unattributed_occupied"], 0)
        self.assertFalse(self.records[-1]["oldest_holders"])
        pool.free_slots.append(1)
        self.rec.release(pool, 1)
        self.assertFalse(self.rec.owners)

    def test_unbounded_or_non_scalar_facts_are_rejected_and_close_is_idempotent(self):
        req = Request("bounded", 5)
        self.rec.bind(req)
        self.rec.observe(req, bad=float("nan"))
        self.rec.observe(req, bad="x" * 129)
        self.rec.observe(req, bad=[1])
        self.assertEqual(self.rec.loss["unsupported_value"], 3)
        self.rec.close()
        count = len(self.records)
        self.rec.close()
        self.assertEqual(len(self.records), count)
        self.assertFalse(self.rec.enabled)

    def test_disabled_and_overflow_are_explicit(self):
        disabled = module.PDLifecycleRecorder(emit=self.records.append)
        disabled.bind(Request("private", 9))
        disabled.summary({})
        self.assertFalse(disabled.entries)
        self.assertFalse(self.records)
        rec = module.PDLifecycleRecorder(
            1, max_requests=1, max_events=1, history=1, emit=self.records.append
        )
        req = Request("first", 1)
        rec.bind(req)
        rec.transition(req, "handshake_wait")
        rec.bind(Request("overflow", 2))
        rec.summary({"preallocation": 2})
        row = self.records[-1]
        self.assertEqual(len(rec.entries), 1)
        self.assertEqual(len(row["events"]), 1)
        self.assertGreater(row["loss_total"]["event_records_dropped"], 0)
        self.assertGreater(row["loss_total"]["untracked_transitions"], 0)
        self.assertTrue(row["oldest_waiters"][0]["incomplete_history"])


if __name__ == "__main__":
    unittest.main()
