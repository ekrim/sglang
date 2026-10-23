"""CPU return-site contracts; no GPU/cache readiness re-evaluation."""
import ast
from collections import namedtuple
from enum import Enum
from types import SimpleNamespace as NS
from unittest.mock import Mock
import unittest

from test_tuning_observability import SRT, production
from test_policy_diagnostics import sample, request

E = Enum("AddReqResult", "CONTINUE NO_TOKEN OTHER")


def adder_methods():
    namespace = dict(
        AddReqResult=E, tuning_recorder=NS(enabled=True, observe=lambda *a, **k: None),
        get_schedule=lambda: NS(schedule_policy="shortest-prefill-first"),
        CLIP_MAX_NEW_TOKENS=4096,
        _PrefillAdmission=namedtuple("Admission", (
            "prefix_len extend_len max_new_tokens is_chunked replay_len")),
    )
    production("managers/schedule_policy.py",
               ("budget_state", "_select_prefill_admission", "ceil_paged_tokens"),
               namespace, owner="PrefillAdder")
    return type("Adder", (), {k: namespace[k] for k in (
        "budget_state", "_select_prefill_admission", "ceil_paged_tokens")})


class PrefillStopSitesTest(unittest.TestCase):
    def test_all_direct_stop_returns_have_bounded_site_literals(self):
        tree = ast.parse((SRT / "managers/schedule_policy.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "PrefillAdder")
        sites = []
        for parent in ast.walk(cls):
            for _, body in ast.iter_fields(parent):
                if not isinstance(body, list):
                    continue
                for i, node in enumerate(body):
                    if not (isinstance(node, ast.Return)
                            and isinstance(node.value, ast.Attribute)
                            and isinstance(node.value.value, ast.Name)
                            and node.value.value.id == "AddReqResult"
                            and node.value.attr != "CONTINUE"):
                        continue
                    self.assertGreater(i, 0)
                    gate = body[i - 1]
                    self.assertIsInstance(gate, ast.If)
                    self.assertEqual(ast.unparse(gate.test),
                                     "getattr(self, '_tuning_capture_stop', False)")
                    self.assertEqual(len(gate.body), 1)
                    assignment = gate.body[0]
                    self.assertIsInstance(assignment, ast.Assign)
                    self.assertEqual(ast.unparse(assignment.targets[0]),
                                     "self._tuning_stop_site")
                    self.assertIsInstance(assignment.value, ast.Constant)
                    self.assertIs(type(assignment.value.value), str)
                    sites.append(assignment.value.value)
        self.assertEqual(len(sites), 26)
        self.assertEqual(len(set(sites)), 26)

    def test_actual_budget_order_and_call_counts_are_unchanged(self):
        Adder = adder_methods()
        for capacity, slots, input_left, chunk_left, dllm, dllm_left, expected in (
            (False, None, 0, 0, None, None, "pass_capacity_or_mamba_slots"),
            (True, 0, 0, 0, None, None, "pass_capacity_or_mamba_slots"),
            (True, None, 0, 0, None, None, "pass_input_budget"),
            (True, None, 1, 0, None, None, "pass_chunk_budget"),
            (True, None, 1, None, object(), 0, "pass_dllm_budget"),
            (True, None, 1, 1, None, None, None),
        ):
            outcomes = []
            for enabled in (False, True):
                a = Adder(); a.memory_budget = NS(has_capacity=Mock(return_value=capacity))
                a.rem_mamba_slots = slots; a.rem_input_tokens = input_left
                a.rem_chunk_tokens = chunk_left; a.dllm_config = dllm
                a.rem_dllm_tokens = dllm_left; a._tuning_capture_stop = enabled
                result = a.budget_state()
                a.memory_budget.has_capacity.assert_called_once_with()
                outcomes.append(result)
                self.assertEqual(getattr(a, "_tuning_stop_site", None),
                                 expected if enabled else None)
            self.assertIs(outcomes[0], outcomes[1])

    def test_positive_residual_chunk_slot_and_other_shape_stops(self):
        Adder = adder_methods()
        for case, expected in (
            ("chunk_slot", "existing_chunk_slot"),
            ("alignment", "chunk_alignment_empty"),
            ("replay", "replay_chunk_budget"),
            ("capacity", "prefill_shape_capacity"),
            ("unchunked", "unchunked_input_budget"),
            ("dllm", "dllm_extend_empty"),
            ("admit", None),
        ):
            outcomes = []
            for enabled in (False, True):
                a = Adder(); a._tuning_capture_stop = enabled
                a.page_size = 256; a.rem_input_tokens = 256
                a.rem_chunk_tokens = None if case == "unchunked" else 256
                a.exact_chunk_fill = False; a.dllm_config = object() if case == "dllm" else None
                a.can_run_list = [object()]; a.rem_total_tokens = 100000
                calls = []
                def replay(*args):
                    calls.append("replay"); return 512 if case == "replay" else 0
                def budget(*args, **kwargs):
                    calls.append("capacity"); return (case != "capacity", a.rem_chunk_tokens)
                a._replay_len = replay; a._check_prefill_budget = budget
                a._get_dllm_extend_len = lambda *args: 0
                a._check_prefill_tile_budget = lambda tokens: calls.append("tile")
                req = NS(prefix_len=0, full_untruncated_fill_ids=range(1000),
                         sampling_params=NS(max_new_tokens=20))
                result = a._select_prefill_admission(
                    req, total_tokens=1020, host_hit_length=0, swa_host_hit_length=0,
                    truncation_align_size=512 if case == "alignment" else None,
                    has_chunked_req=case == "chunk_slot")
                outcomes.append((result, calls))
                self.assertEqual(getattr(a, "_tuning_stop_site", None),
                                 expected if enabled else None)
                self.assertEqual(a.rem_input_tokens, 256)
                self.assertEqual(a.rem_chunk_tokens, None if case == "unchunked" else 256)
            self.assertEqual(outcomes[0], outcomes[1], case)

    def test_end_preserves_admitted_vs_rejected_and_unknown(self):
        for admitted in (False, True):
            p, rows = sample(n=1); p.visit(request("candidate"), "admitted" if admitted else "not_admitted_other")
            p.end("adder_result", adder_result="other", candidate_admitted=admitted,
                  adder_stop_site="pass_input_budget" if admitted else "existing_chunk_slot")
            p.finish()
            self.assertIs(rows[0]["pass_end"]["candidate_admitted"], admitted)
            self.assertEqual(rows[0]["pass_end"]["adder_stop_site"],
                             "pass_input_budget" if admitted else "existing_chunk_slot")
        p, rows = sample(); p.end("adder_result", adder_result="other")
        p.finish(); self.assertIsNone(rows[0]["pass_end"]["adder_stop_site"])


if __name__ == "__main__":
    unittest.main()
