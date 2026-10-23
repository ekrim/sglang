"""CPU contracts for real Prometheus graph-counter baselines.

Execute the unmodified production initialization/increment methods with a
minimal collector and runtime context. This checks export and rank selection,
not full scheduler startup or multiprocess serving integration.
"""
import unittest
from types import SimpleNamespace as NS

from prometheus_client import CollectorRegistry, Counter, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from test_tuning_observability import production


class GraphCounterBaselinesTest(unittest.TestCase):
    def setUp(self):
        self.registry = CollectorRegistry()
        self.observability = NS(
            enable_metrics=True,
            enable_metrics_for_all_schedulers=False,
            kv_events_config=None,
            extra_metric_labels={"campaign": "test"},
        )
        self.parallel = NS(
            attn_tp_rank=0, attn_cp_rank=0, tp_rank=0, pp_rank=0,
            moe_ep_rank=0, dp_rank=0,
        )
        self.disagg = NS(enable_pd_role_switch=False, disaggregation_mode="null")
        self.serving = NS(served_model_name="model", enable_streaming_session=False)
        registry = self.registry

        class Collector:
            metric = None
            constructions = 0

            def __init__(self, labels, **kwargs):
                type(self).constructions += 1
                self.labels = labels
                if type(self).metric is None:
                    type(self).metric = Counter(
                        "sglang:cuda_graph_passes_total", "passes",
                        list(labels) + ["mode"], registry=registry,
                    )
                self.cuda_graph_passes_total = type(self).metric

        namespace = dict(
            get_observability=lambda: self.observability,
            get_parallel=lambda: self.parallel,
            get_disagg=lambda: self.disagg,
            get_serving=lambda: self.serving,
            DisaggregationMode=NS(to_engine_type=lambda mode: "unified"),
            resolve_collector_class=lambda role, cls: cls,
            STAT_LOGGER_ROLE_SCHEDULER="scheduler",
            SchedulerMetricsCollectorContext=NS,
        )
        methods = (
            "init_new", "initialize_cuda_graph_passes",
            "increment_decode_cuda_graph_pass", "increment_prefill_cuda_graph_pass",
        )
        production("observability/metrics_collector.py", methods, namespace,
                   owner="SchedulerMetricsCollector")
        for name in methods:
            setattr(Collector, name, namespace[name])
        self.Collector = Collector

    def create(self, priority=False):
        return self.Collector.init_new(
            server_args=NS(), enable_priority_scheduling=priority,
            enable_lora=False, enable_hierarchical_cache=False,
        )

    def samples(self):
        # Check public text exposition rather than Counter private values.
        return [s for f in text_string_to_metric_families(
            generate_latest(self.registry).decode()) for s in f.samples
            if s.name == "sglang:cuda_graph_passes_total"]

    def test_reporting_rank_exports_four_zero_modes_with_final_labels(self):
        self.disagg.enable_pd_role_switch = True
        self.observability.extra_metric_labels["model_name"] = "override"
        ctx = self.create(priority=True)
        self.assertTrue(ctx.current_scheduler_metrics_enabled)
        samples = self.samples()
        self.assertEqual(len(samples), 4)
        self.assertEqual({s.labels["mode"] for s in samples}, {
            "decode_cuda_graph", "decode_none", "prefill_cuda_graph", "prefill_none"})
        for s in samples:
            self.assertEqual(s.value, 0)
            self.assertEqual(s.labels["model_name"], "override")
            self.assertEqual(s.labels["engine_type"], "dynamic")
            self.assertEqual(s.labels["campaign"], "test")
            self.assertEqual(s.labels["priority"], "")
            self.assertEqual(s.labels["dp_rank"], "0")

    def test_disabled_metrics_do_not_construct_or_export(self):
        self.observability.enable_metrics = False
        self.assertIsNone(self.create().collector)
        self.assertEqual(self.Collector.constructions, 0)
        self.assertEqual(self.samples(), [])

    def test_nonreporting_tp_or_cp_rank_does_not_export_zero_children(self):
        for attr in ("attn_tp_rank", "attn_cp_rank"):
            setattr(self.parallel, attr, 1)
            self.assertFalse(self.create().current_scheduler_metrics_enabled)
            self.assertEqual(self.samples(), [])
            setattr(self.parallel, attr, 0)

    def test_all_scheduler_override_enables_nonleader_baselines(self):
        self.parallel.attn_tp_rank = 1
        self.observability.enable_metrics_for_all_schedulers = True
        self.assertTrue(self.create().current_scheduler_metrics_enabled)
        self.assertEqual(len(self.samples()), 4)

    def test_rank_isolation_increments_and_reinitialization(self):
        collectors = []
        for rank in range(4):
            self.parallel.tp_rank = self.parallel.dp_rank = rank
            collectors.append(self.create().collector)
        collectors[2].increment_decode_cuda_graph_pass(True)
        collectors[2].increment_decode_cuda_graph_pass(True)
        collectors[1].increment_decode_cuda_graph_pass(False)
        collectors[0].increment_prefill_cuda_graph_pass(True)
        collectors[3].increment_prefill_cuda_graph_pass(False)
        for collector in collectors:
            collector.initialize_cuda_graph_passes()
        samples = self.samples()
        self.assertEqual(len(samples), 16)
        nonzero = {(s.labels["dp_rank"], s.labels["mode"]): s.value
                   for s in samples if s.value}
        self.assertEqual(nonzero, {
            ("2", "decode_cuda_graph"): 2, ("1", "decode_none"): 1,
            ("0", "prefill_cuda_graph"): 1, ("3", "prefill_none"): 1,
        })


if __name__ == "__main__":
    unittest.main()
