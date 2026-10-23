"""CPU-only contracts for tuning telemetry.

Load the dependency-free recorder directly. For GPU-owning modules, execute
unmodified AST-extracted production methods/classes with fake resources. This
checks decisions and disabled/enabled equivalence without importing CUDA libs;
it does not replace a full serving/graph integration test.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tuning = load_file("tuning_under_test", SRT / "observability/tuning.py")


def production(path, names, namespace, owner=None):
    tree = ast.parse((SRT / path).read_text())
    body = (
        tree.body
        if owner is None
        else next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    )
    chosen = [
        n
        for n in body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert len(chosen) == len(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + chosen,
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(SRT / path), "exec"), namespace)
    return namespace


class RecorderTest(unittest.TestCase):
    def test_disabled_does_not_read_clock_or_emit(self):
        clock, emit = Mock(side_effect=AssertionError), Mock(side_effect=AssertionError)
        r = tuning.TuningRecorder(clock=clock, emit=emit)
        r.observe("not_even_valid", value=object())
        r.flush()
        self.assertFalse(r.counts)

    def test_window_summaries_and_reset(self):
        now = [0.0]
        records = []
        r = tuning.TuningRecorder(
            10, {"tp_rank": 1}, clock=lambda: now[0], emit=records.append
        )
        r.observe("decode.kv_budget", full_required=100, full_budget=90)
        now[0] = 10
        r.observe("decode.kv_budget", full_required=120, full_budget=80)
        self.assertEqual(len(records), 1)
        row = records[0]["events"][0]
        self.assertEqual(row["observations"], 2)
        self.assertEqual(
            row["values"]["full_budget"],
            dict(count=2, sum=170, min=80, max=90, last=80),
        )
        self.assertEqual(records[0]["window_seconds"], 10)
        self.assertEqual(json.loads(json.dumps(records)), records)
        self.assertFalse(r.counts)
        self.assertFalse(r.samples)
        r.flush()
        self.assertEqual(len(records), 1)

    def test_bounded_vocabulary_and_no_tensor_values(self):
        r = tuning.TuningRecorder(10)
        for event, values in [
            ("request-123", {}),
            ("prefill.request_limit", {"rid": 1}),
            ("decode.kv_budget", {"full_budget": float("nan")}),
            ("decode.kv_budget", {"full_budget": object()}),
        ]:
            with self.assertRaises(ValueError):
                r.observe(event, **values)
        self.assertFalse(r.counts)

    def test_invalid_interval(self):
        for v in [-1, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                tuning.TuningRecorder(v)


class AllocatorSnapshotTest(unittest.TestCase):
    def test_disabled_and_empty_windows_never_sample(self):
        sampler = Mock(side_effect=AssertionError("must not sample"))
        recorder = tuning.TuningRecorder(allocator_sampler=sampler)
        recorder.observe("prefill.input_budget")
        recorder.flush()
        tuning.TuningRecorder(10, allocator_sampler=sampler).flush()
        sampler.assert_not_called()

    def test_once_per_flush_and_diagnostic_failure_preserves_events(self):
        now, records = [0.0], []
        sampler = Mock(side_effect=[{"status": "not_initialized"}, RuntimeError])
        recorder = tuning.TuningRecorder(
            10, clock=lambda: now[0], emit=records.append,
            allocator_sampler=sampler,
        )
        for _ in range(20):
            recorder.observe("prefill.input_budget")
        sampler.assert_not_called()
        now[0] = 10
        recorder.observe("prefill.input_budget")
        self.assertEqual(sampler.call_count, 1)
        self.assertEqual(records[0]["events"][0]["observations"], 21)
        self.assertEqual(records[0]["allocator_snapshot"], {"status": "not_initialized"})
        recorder.flush()
        self.assertEqual(sampler.call_count, 1)
        now[0] = 20
        recorder.observe("prefill.chunk_budget")
        self.assertEqual(sampler.call_count, 2)
        self.assertEqual(records[1]["allocator_snapshot"], {"status": "error"})
        self.assertEqual(records[1]["events"][0]["event"], "prefill.chunk_budget")
        self.assertFalse(recorder.counts)
        self.assertEqual(json.loads(json.dumps(records)), records)

    def test_uninitialized_or_unsupported_never_reads_stats(self):
        stats = Mock(side_effect=AssertionError("must not read"))
        backend = Mock(side_effect=AssertionError("must not inspect backend"))
        cuda = NS(is_initialized=lambda: False, memory_stats=stats,
                  memory=NS(get_allocator_backend=backend))
        self.assertEqual(tuning.cuda_allocator_snapshot(cuda), {"status": "not_initialized"})
        backend.assert_not_called()
        cuda.is_initialized = lambda: True
        cuda.memory.get_allocator_backend = lambda: "cudaMallocAsync"
        self.assertEqual(tuning.cuda_allocator_snapshot(cuda), {"status": "unsupported_backend"})
        stats.assert_not_called()

    def test_native_snapshot_units_counters_and_missing_values(self):
        values = {
            "allocated_bytes.all.current": 100,
            "reserved_bytes.all.current": 200,
            "active_bytes.all.current": 110,
            "inactive_split_bytes.all.current": 30,
            "num_alloc_retries": 4,
            "num_ooms": 0,
            "unrelated.dynamic.key": 999,
        }
        # Deliberately expose no init/synchronize/empty_cache/reset methods.
        cuda = NS(is_initialized=lambda: True,
                  memory=NS(get_allocator_backend=lambda: "native"),
                  memory_stats=lambda: values)
        result = tuning.cuda_allocator_snapshot(cuda)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["values"], {
            "allocated_bytes": 100, "reserved_bytes": 200,
            "active_bytes": 110, "inactive_split_bytes": 30,
            "allocation_retries_total": 4, "ooms_total": 0,
        })
        del values["allocated_bytes.all.current"]
        values["reserved_bytes.all.current"] = -1
        values["num_alloc_retries"] = float("nan")
        values["num_ooms"] = True
        result = tuning.cuda_allocator_snapshot(cuda)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["values"], {"active_bytes": 110, "inactive_split_bytes": 30})

    def test_peaks_share_one_read_and_preserve_reset_scope(self):
        values = {
            "allocated_bytes.all.current": 100,
            "allocated_bytes.all.peak": 180,
            "reserved_bytes.all.peak": 250,
            "active_bytes.all.peak": True,
            "inactive_split_bytes.all.peak": -1,
        }
        stats = Mock(side_effect=lambda: dict(values))
        cuda = NS(is_initialized=lambda: True,
                  memory=NS(get_allocator_backend=lambda: "native"),
                  memory_stats=stats)
        first = tuning.cuda_allocator_snapshot(cuda)
        stats.assert_called_once_with()
        self.assertEqual(first["peaks"], {
            "scope": "native_allocator_since_last_peak_reset",
            "values": {"allocated_bytes": 180, "reserved_bytes": 250},
        })
        # A reset by another caller is observable; the sampler must not hide it
        # behind its own maximum or mislabel it a process-lifetime peak.
        values["allocated_bytes.all.peak"] = 100
        second = tuning.cuda_allocator_snapshot(cuda)
        self.assertEqual(second["peaks"]["values"]["allocated_bytes"], 100)
        self.assertEqual(first["peaks"]["values"]["allocated_bytes"], 180)
        values.clear()
        self.assertNotIn("peaks", tuning.cuda_allocator_snapshot(cuda))


class MoEBufferReceiptTest(unittest.TestCase):
    def test_only_cache_misses_emit_and_buffer_identity_is_unchanged(self):
        for enabled in (False, True):
            factory = Mock(side_effect=lambda *a, **k: object())
            emit = Mock()
            library = NS(get_num_sms=lambda: 132, get_symm_buffer_for_mega_moe=factory)
            ns = production(
                "layers/moe/mega_moe.py", {"_get_mega_moe_symm_buffer"},
                dict(_MEGA_MOE_SYMM_BUFFER={},
                     _configure_mega_moe_deep_gemm_num_sms=lambda _: nullcontext(),
                     envs=NS(SGLANG_TUNING_MOE_BUFFER_STATS=NS(get=lambda: enabled))),
            )
            group = NS(size=lambda: 4)
            with patch.dict(sys.modules, {
                "deep_gemm": library,
                "sglang.srt.observability.tuning": NS(log_mega_moe_buffer=emit),
            }):
                get = ns["_get_mega_moe_symm_buffer"]
                first = get(group, 384, 20480, 6, 5120, 2304, mma_type="fp8xfp4")
                again = get(group, 384, 20480, 6, 5120, 2304, mma_type="fp8xfp4")
                second = get(group, 128, 20480, 3, 5120, 2304, mma_type="fp8xfp4")
            self.assertIs(first, again)
            self.assertIsNot(first, second)
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(emit.call_count, 2 if enabled else 0)
            if enabled:
                self.assertEqual(emit.call_args.kwargs, {
                    "requested_token_cap": 20480, "group_size": 4, "num_sms": 132,
                })

    def test_cpu_metadata_records_aligned_cap_and_bytes_without_tensor_reads(self):
        buffer = NS(num_max_tokens_per_rank=256, num_experts=384, num_topk=6,
                    hidden=5120, intermediate_hidden=2304, num_shared_experts=1,
                    mma_type="fp8xfp4", activation="swiglu", buffer=NS(nbytes=4096))
        with patch.object(tuning.logger, "info") as emit:
            tuning.log_mega_moe_buffer(buffer, requested_token_cap=129, group_size=4, num_sms=132)
        record = json.loads(emit.call_args.args[1])
        self.assertEqual(record["status"], "ok")
        self.assertEqual(record["values"]["requested_token_cap"], 129)
        self.assertEqual(record["values"]["resolved_token_cap"], 256)
        self.assertEqual(record["values"]["buffer_bytes"], 4096)
        self.assertEqual(record["mma_type"], "fp8xfp4")
        self.assertNotIn("group_id", record)

    def test_missing_metadata_is_unknown_and_does_not_raise(self):
        with patch.object(tuning.logger, "info") as emit:
            tuning.log_mega_moe_buffer(object(), requested_token_cap=128, group_size=4, num_sms=132)
        record = json.loads(emit.call_args.args[1])
        self.assertEqual(record["status"], "unavailable")
        self.assertNotIn("values", record)


class BudgetTest(unittest.TestCase):
    def setup_budget(self, enabled, cls="PrefillBudget", full=100, swa=100):
        r = tuning.TuningRecorder(1000 if enabled else 0)
        ns = production(
            "mem_cache/prefill_budget.py",
            {
                "estimate_swa_kv_tokens",
                "PrefillBudget",
                "SWAPrefillBudget",
                "SharedSWAPrefillBudget",
            },
            dict(tuning_recorder=r, is_swa_req_ring=lambda a: False),
        )
        allocator = NS(
            page_size=1,
            size_swa=100,
            available_size=lambda: full,
            full_available_size=lambda: full,
            swa_available_size=lambda: swa,
        )
        cache = NS(
            supports_mamba=lambda: False,
            evictable_size=lambda: 0,
            full_evictable_size=lambda: 0,
            swa_evictable_size=lambda: 0,
            sliding_window_size=None,
        )
        return ns[cls](allocator, cache), r

    @staticmethod
    def check(b, tokens=10):
        return b.check_prefill(
            extend_input_len=tokens,
            total_tokens=tokens + 1,
            max_new_tokens=0,
            input_tokens=tokens,
            swa_host_hit_length=0,
            chunk_limit=20,
        )

    def test_full_boundary_and_observation_do_not_change_decision(self):
        for full in [0, 10, 11, 12, 100]:
            off, _ = self.setup_budget(False, full=full)
            on, r = self.setup_budget(True, full=full)
            self.assertEqual(self.check(off), self.check(on))
            self.assertEqual(r.counts["prefill.full_capacity"], int(full <= 11))

    def test_swa_pressure_distinct_from_full(self):
        for enabled in [False, True]:
            b, r = self.setup_budget(enabled, "SWAPrefillBudget", full=100, swa=5)
            self.assertEqual(self.check(b), (False, None))
            self.assertEqual(r.counts["prefill.swa_capacity"], int(enabled))
            self.assertEqual(r.counts["prefill.full_capacity"], 0)

    def test_swa_permanent_shortfall_can_reduce_chunk(self):
        b, r = self.setup_budget(True, "SWAPrefillBudget", full=100, swa=5)
        b.allocator.size_swa = 5
        ok, cap = self.check(b)
        self.assertTrue(ok)
        self.assertLess(cap, 20)
        self.assertEqual(r.counts["prefill.swa_chunk_cap"], 1)

    def test_shared_budget_does_not_invent_independent_pool_failure(self):
        b, r = self.setup_budget(True, "SharedSWAPrefillBudget")
        b.allocator.can_reserve = lambda *a, **kw: kw.get("empty_pool", False)
        self.assertEqual(self.check(b), (False, None))
        self.assertEqual(r.counts["prefill.shared_capacity"], 1)
        self.assertEqual(r.counts["prefill.full_capacity"], 0)


class GraphEligibilityTest(unittest.TestCase):
    def run_case(self, enabled, updates=None, inputs=None):
        r = tuning.TuningRecorder(1000 if enabled else 0)
        ns = production(
            "model_executor/runner/prefill_cuda_graph_runner.py",
            {"can_replay_locally"},
            dict(
                tuning_recorder=r,
                Backend=NS(BREAKABLE="breakable"),
                is_cuda=lambda: True,
                _MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR=4,
            ),
            owner="PrefillCudaGraphRunner",
        )
        runner = NS(
            enable_cp_bcg_capture=False,
            _qwen_bcg_hc_sidechannel=False,
            _qwen_bcg_mtp_draft=False,
            _is_full_backend=False,
            _fa4_prefill=False,
            _capture_req_slots=10,
            _captured_attn_metadata_max_bs=None,
            prefer_eager_mixed_prefill=False,
            prefill_backend_name="breakable",
            has_mha_companion_layers=False,
            _has_uncapturable_chunked_prefix=lambda p: False,
            capture_hidden_mode=1,
            _uses_eager_prefill_tail=lambda: True,
            max_context_size=16384,
            max_seq_len=None,
            max_num_tokens=8192,
            capture_num_tokens=[1024, 8192],
            _pad_to_bucket=lambda n, buckets: n,
        )
        runner.__dict__.update(updates or {})
        args = dict(
            batch_size=1,
            num_tokens=1024,
            input_embeds=None,
            replace_embeds=None,
            prefix_lens=[100],
            is_target_verify=False,
            capture_hidden_mode=0,
            return_logprob=False,
            batch_max_context_len=2048,
        )
        args.update(inputs or {})
        return ns["can_replay_locally"](runner, **args), r

    def test_eligible_and_each_relevant_fallback(self):
        cases = [
            ({}, {}, None),
            ({}, {"batch_max_context_len": 32768}, "context_coverage"),
            ({}, {"num_tokens": 16384}, "token_coverage"),
            ({"prefer_eager_mixed_prefill": True}, {"is_mixed": True}, "mixed_batch"),
            (
                {"_captured_attn_metadata_max_bs": 1},
                {"batch_size": 2},
                "metadata_capacity",
            ),
            (
                {"_is_full_backend": True, "_capture_req_slots": 1},
                {"batch_size": 2},
                "request_capacity",
            ),
            ({}, {"lora_ineligible": True}, "lora"),
            ({"_pad_to_bucket": lambda n, b: 10000}, {}, "padding"),
        ]
        for updates, inputs, reason in cases:
            with self.subTest(reason=reason):
                off, _ = self.run_case(False, updates, inputs)
                on, recorder = self.run_case(True, updates, inputs)
                self.assertEqual(off, on)
                self.assertEqual(on, reason is None)
                self.assertEqual(
                    dict(recorder.counts),
                    {} if reason is None else {"graph.prefill." + reason: 1},
                )


class DecodeAdmissionTest(unittest.TestCase):
    def run_gate(self, enabled, gate):
        r = tuning.TuningRecorder(1000 if enabled else 0)

        class Abort:
            pass

        ns = production(
            "disaggregation/decode.py",
            {"pop_preallocated"},
            dict(
                tuning_recorder=r,
                FINISH_ABORT=Abort,
                get_disagg=lambda: NS(disaggregation_decode_host_receive_threshold=1),
                ensure_mamba_capacity=lambda *a: gate != "mamba_capacity",
            ),
            owner="DecodePreallocQueue",
        )
        req = NS(finished_reason=None)
        entry = NS(req=req, waiting_for_input=True)
        q = NS(
            pp_size=1,
            queue=[entry],
            pending_reqs=[],
            _resolve_pending_reqs=lambda: None,
            _update_handshake_waiters=lambda *a: None,
            _uses_swa_tail_prealloc=lambda: False,
            _uses_swa_reservation=lambda: False,
            _allocatable_token_budgets=lambda **kw: 100,
            _hicache_pending_restore_tokens=lambda: 0,
            req_to_metadata_buffer_idx_allocator=NS(
                available_size=lambda: 0 if gate == "metadata_slots" else 1
            ),
            req_to_token_pool=NS(
                available_size=lambda: 0 if gate == "request_slots" else 1
            ),
            tree_cache=None,
            scheduler=NS(
                running_batch=NS(reqs=[]),
                enable_priority_scheduling=False,
                enable_hisparse=False,
                enable_lora=False,
            ),
        )
        result = ns["pop_preallocated"](q)
        return result, q, r

    def test_blocked_requests_retained_with_telemetry_on_and_off(self):
        for gate in ["metadata_slots", "request_slots", "mamba_capacity"]:
            for enabled in [False, True]:
                result, q, r = self.run_gate(enabled, gate)
                self.assertEqual(result, ([], []))
                self.assertEqual(len(q.queue), 1)
                self.assertEqual(
                    dict(r.counts), {"decode." + gate: 1} if enabled else {}
                )


class DispatchContractTest(unittest.TestCase):
    def test_target_and_draft_paths_have_separate_counters(self):
        r = tuning.TuningRecorder(1000)
        ns = production(
            "model_executor/model_runner.py",
            {"_observe_tuning_dispatch"},
            dict(tuning_recorder=r, logger=Mock()),
            owner="ModelRunner",
        )
        for draft in (False, True):
            for path in ("decode_runner", "prefill_graph", "eager", "split_prefill"):
                ns["_observe_tuning_dispatch"](
                    NS(is_draft_worker=draft), path, NS(batch_size=3)
                )
        self.assertEqual(len(r.counts), 8)
        self.assertTrue(all(v == 1 for v in r.counts.values()))


class DecodeIntervalTest(unittest.TestCase):
    def test_interval_decisions_unchanged_with_observation(self):
        for enabled in (False, True):
            r = tuning.TuningRecorder(1000 if enabled else 0)
            ns = production(
                "managers/scheduler.py", {"_should_defer_prefill"},
                dict(tuning_recorder=r), owner="Scheduler",
            )
            state = NS(_prefill_decode_interval_remaining=2,
                       waiting_queue=[object(), object()], chunked_req=object())
            actual = [ns["_should_defer_prefill"](state) for _ in range(4)]
            self.assertEqual(actual, [True, True, False, False])
            self.assertEqual(state._prefill_decode_interval_remaining, 0)
            self.assertEqual(dict(r.counts), {"prefill.decode_interval": 2} if enabled else {})
            if enabled:
                self.assertEqual(r.samples[("prefill.decode_interval", "remaining_interval")]["sum"], 1)
                self.assertEqual(r.samples[("prefill.decode_interval", "waiting_requests")]["last"], 2)
                self.assertEqual(r.samples[("prefill.decode_interval", "chunk_pending")]["last"], 1)

    def test_idle_deferral_is_not_reported_as_queued_work(self):
        r = tuning.TuningRecorder(1000)
        ns = production("managers/scheduler.py", {"_should_defer_prefill"},
                        dict(tuning_recorder=r), owner="Scheduler")
        state = NS(_prefill_decode_interval_remaining=1, waiting_queue=[], chunked_req=None)
        self.assertTrue(ns["_should_defer_prefill"](state))
        for name in ("waiting_requests", "chunk_pending", "remaining_interval"):
            self.assertEqual(r.samples[("prefill.decode_interval", name)]["last"], 0)


class ConfigGraphTest(unittest.TestCase):
    def test_sources_contracts_and_inventory(self):
        tools = load_file(
            "config_graph_tools", SRT / "observability/config_graph/tools.py"
        )
        report = tools.check(tools.load_graph())
        self.assertGreater(report["mapped_fields"], 50)
        selected = tools.lookup(tools.load_graph(), "--chunked-prefill-size")
        self.assertIn("mechanism.prefill_tokens", {n["id"] for n in selected["nodes"]})
        self.assertEqual(
            tools.lookup(tools.load_graph(), "unknown")["status"],
            "not_causally_audited",
        )


if __name__ == "__main__":
    unittest.main()
