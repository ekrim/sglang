"""Observe real pool/cleanup/timestamp transitions with bounded CPU fixtures.

Transport and model execution are separately verified by the native PD probe.
"""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import DecodeReqToTokenPool
from sglang.srt.disaggregation.utils import ReqToMetadataIdxAllocator, prepare_abort
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.observability.pd_lifecycle import recorder
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

spec = importlib.util.spec_from_file_location(
    "pd_deferred_fixtures",
    Path(__file__).with_name("test_deferred_decode_kv_release.py"),
)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


class TestActualPDTransitions(unittest.TestCase):
    def setUp(self):
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.records = []
        recorder.__init__(1, emit=self.records.append, context={"role": "decode"})
        self.pool = DecodeReqToTokenPool(
            size=1,
            max_context_len=4,
            device="cpu",
            enable_memory_saver=False,
            pre_alloc_size=0,
        )
        self.metadata = ReqToMetadataIdxAllocator(1)
        self.addCleanup(reset_context)
        self.addCleanup(lambda: recorder.__init__())

    def request(self, name, room=9007199254740993):
        req = Req(
            name, "", [1, 2, 3], SamplingParams(max_new_tokens=2), bootstrap_room=room
        )
        req.time_stats.trace_ctx = MagicMock(tracing_enable=False)
        return req

    def test_actual_pool_exhaustion_and_reuse(self):
        first, second = self.request("first"), self.request("second", 2)
        self.assertEqual(self.pool.alloc([first]), [1])
        self.assertEqual(self.metadata.alloc(first), 0)
        self.assertIsNone(self.pool.alloc([second]))
        first.time_stats.set_decode_transfer_queue_entry_time()
        recorder.summary({"preallocation": 1, "transfer": 1}, force=True)
        holder = self.records[-1]["oldest_holders"][0]
        self.assertEqual(holder["room"], "9007199254740993")
        self.assertEqual(holder["resources"]["request_slot"]["slot"], 1)
        first_generation = holder["resources"]["request_slot"]["acquisition_seq"]
        self.metadata.free(0)
        self.pool.free(first)
        first.time_stats.set_completion_time()
        self.assertEqual(self.pool.alloc([second]), [1])
        recorder.summary({}, force=True)
        replacement = self.records[-1]["oldest_holders"][0]
        self.assertEqual(replacement["rid"], "second")
        self.assertGreater(
            replacement["resources"]["request_slot"]["acquisition_seq"],
            first_generation,
        )
        self.assertEqual(self.records[-1]["pools"][0]["unattributed_occupied"], 0)

    def test_actual_abort_then_deferred_cleanup_waits_for_ack(self):
        req = self.request("held")
        self.pool.alloc([req])
        idx = self.metadata.alloc(req)
        manager = fixtures._make_manager()
        queue = fixtures._make_queue()
        queue.req_to_metadata_buffer_idx_allocator = self.metadata
        entry = fixtures._make_decode_req(req.bootstrap_room, idx, manager)
        entry.req = req
        generation = manager.register_deferred_abort_room(req.bootstrap_room)
        prepare_abort(req, "test timeout")
        queue._defer_release(entry)
        # The actual deferred cleanup owns the decision and metadata teardown;
        # only physical KV freeing is replaced with the real CPU request pool.
        with (
            patch("torch.distributed.get_world_size", return_value=1),
            patch.object(
                queue, "_release_request", side_effect=lambda dr: self.pool.free(dr.req)
            ),
        ):
            queue.resolve_deferred_releases()
            self.assertEqual(self.pool.available_size(), 0)
            recorder.summary({"deferred_release": 1}, force=True)
            self.assertEqual(
                self.records[-1]["oldest_holders"][0]["stage"], "cleanup_deferred"
            )
            manager.note_abort_ack(req.bootstrap_room, 0, generation)
            queue.resolve_deferred_releases()
        self.assertEqual(self.pool.available_size(), 1)
        self.assertEqual(self.metadata.available_size(), 1)
        recorder.summary({}, force=True)
        self.assertFalse(recorder.owners)
        self.assertFalse(recorder.entries)
        events = [e for r in self.records for e in r["events"]]
        self.assertEqual(sum(e["event"] == "resource_released" for e in events), 2)
        self.assertTrue(any(e.get("cleanup_outcome") == "drained" for e in events))

    def test_abort_before_reservation_and_retry_timestamps(self):
        req = self.request("unreserved")
        req.time_stats.set_decode_prealloc_queue_entry_time()
        prepare_abort(req, "cancel before reservation")
        recorder.summary({"preallocation": 0}, force=True)
        self.assertFalse(recorder.owners)
        aborted = next(e for e in self.records[-1]["events"] if e["event"] == "aborted")
        self.assertFalse(aborted["diagnostic"]["resources"])
        req = self.request("retry", 4)
        req.time_stats.set_prefill_bootstrap_queue_entry_time()
        req.time_stats.reset_prefill_retry_time()
        req.time_stats.set_wait_queue_entry_time()
        recorder.summary({}, force=True)
        events = [e["event"] for e in self.records[-1]["events"]]
        self.assertIn("prefill_retry", events)
        self.assertIn("execution_wait", events)


if __name__ == "__main__":
    unittest.main()
