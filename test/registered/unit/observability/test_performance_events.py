"""CPU source-hook and real pipe tests; not native model/overhead qualification."""
import ast
import json
import os
import subprocess
import sys
import time
import unittest
from enum import IntEnum
from types import SimpleNamespace as NS
from unittest.mock import Mock

from test_tuning_observability import SRT, load_file, production

pe = load_file("performance_events_test", SRT / "observability/performance_events.py")
reader_module = load_file("performance_event_reader_test", SRT / "observability/performance_event_reader.py")


class PerformanceEventsTest(unittest.TestCase):
    def recorder(self, **kwargs):
        self.now = [0.0]
        self.records = []
        return pe.EventRecorder(True, clock=lambda: self.now[0], emit=self.records.append, **kwargs)

    def test_disabled_reads_no_clock_or_payload(self):
        recorder = pe.EventRecorder(clock=Mock(side_effect=AssertionError))
        recorder.event("invalid", payload=object())
        recorder.queue_depth("invalid", object())
        recorder.arrival(object())
        recorder.tick()
        recorder.decision(object())
        self.assertEqual(recorder.seq, 0)

    def test_short_pressure_episode_and_hysteresis(self):
        r = self.recorder(high=4, low=1)
        for now, depth in [(0, 0), (.01, 4), (.02, 6), (.03, 2), (.04, 3), (.05, 1)]:
            self.now[0] = now
            r.queue_depth("execution_wait", depth)
        self.assertEqual([x["event"] for x in self.records],
                         ["queue.initial", "queue.pressure_enter", "queue.pressure_exit"])
        self.assertEqual(self.records[-1]["values"]["peak_depth"], 6)
        self.assertAlmostEqual(self.records[-1]["values"]["duration_seconds"], .04)
        self.assertLess(self.records[-1]["occurred_monotonic"], .1)

    def test_surge_baseline_recovery_and_idle_window(self):
        r = self.recorder(surge_min=4)
        r.tick()
        for count in [2, 2, 10, 10, 2, 0]:
            for _ in range(count):
                r.arrival(100)
            self.now[0] += 1
            r.tick()
        events = [x for x in self.records if "surge" in x["event"]]
        self.assertEqual([x["event"] for x in events], ["arrival.surge_enter", "arrival.surge_exit"])
        self.assertEqual(events[0]["values"]["baseline_rate"], 2)
        self.assertEqual(events[1]["values"]["peak_rate"], 10)
        self.assertEqual(self.records[-1]["values"]["requests"], 0)
        self.assertEqual(self.records[-1]["values"]["input_tokens"], 0)

    def test_rate_limit_loss_and_invalid_payload(self):
        r = self.recorder(max_per_second=2)
        for _ in range(5):
            r.event("decode.retraction", requests=1, aborted=0, tokens_gained=2, reason="kv_capacity")
        r.event("cache.host_evict", prompt="secret")
        self.assertEqual(r.loss, {"rate_limited": 3, "invalid_event": 1})
        self.assertEqual(len(self.records), 2)
        self.now[0] = 1
        r.event("decode.retraction", requests=0, aborted=0, tokens_gained=0, reason="test")
        self.assertEqual(self.records[-1]["seq"], 7)
        self.assertEqual(self.records[-1]["loss_total"]["rate_limited"], 3)

    def test_repeat_decision_is_not_blocked_duration(self):
        r = self.recorder()
        for _ in range(4): r.decision("prefill.request_slots")
        self.now[0] = .1
        r.decision("prefill.admitted")
        self.assertEqual(len(self.records), 2)
        self.assertEqual(self.records[-1]["values"]["repeats"], 4)
        self.assertEqual(self.records[-1]["values"]["previous"], "prefill.request_slots")

    def test_bounded_pending_and_sink_failure_are_explicit(self):
        r=self.recorder(max_pending=2)
        r.emit=None  # exercise nonblocking producer queue without a draining sink
        for _ in range(5):
            r.event("decode.retraction", requests=0, aborted=0, tokens_gained=0, reason="test")
        self.assertEqual(r.pending.qsize(),2)
        self.assertEqual(r.loss["queue_full"],3)
        self.assertEqual(r._status()["loss_total"]["queue_full"],3)
        r.emit=Mock(side_effect=OSError("private details"))
        r.event("decode.retraction", requests=0, aborted=0, tokens_gained=0, reason="test")
        self.assertEqual(r.loss["sink_errors"],1)
        self.assertNotIn("private",str(r._status()))

    def test_reader_rejects_bad_event_timestamp_and_unknown_vocabulary(self):
        reader=reader_module.EventReader()
        for event,stamp in [("decode.retraction",None),("unbounded.new.key",0)]:
            record=dict(schema_version=1,kind="event",context={"incarnation":"x"},
                        delivery_seq=1,seq=1,event=event,occurred_monotonic=stamp,values={})
            reader.feed(pe.PREFIX+json.dumps(record))
        self.assertEqual(reader.report()["transport_gaps"],{"malformed":2})
        self.assertEqual(set(pe.FIELDS),reader_module.EVENT_NAMES)

    def test_actual_host_eviction_distinguishes_zero_and_component(self):
        class CT(IntEnum): FULL=0; SWA=1
        for enabled in [False, True]:
            r = self.recorder(); r.enabled = enabled
            ns = dict(performance_events=r, time=NS(perf_counter=lambda: self.now[0]), BASE_COMPONENT_TYPE=CT.FULL)
            production("mem_cache/unified_radix_cache.py", ["evict_host"], ns, owner="UnifiedRadixCache")
            cache = NS(host_memory_mode="normal", tree_core=NS(drive_host_eviction=Mock(return_value=NS(tracker={CT.SWA: 0}))), _apply_eviction_result=Mock())
            self.assertEqual(ns["evict_host"](cache, 128, CT.SWA), 0)
            cache.tree_core.drive_host_eviction.assert_called_once_with(CT.SWA, 128)
            cache._apply_eviction_result.assert_called_once()
            if enabled:
                self.assertEqual(self.records[0]["event"], "cache.host_evict")
                self.assertEqual(self.records[0]["values"]["component"], "SWA")
                self.assertEqual(self.records[0]["values"]["freed"], 0)
            cache.host_memory_mode="buffer_only"
            before=len(self.records)
            self.assertEqual(ns["evict_host"](cache, 128, CT.SWA), 0)
            self.assertEqual(len(self.records), before)  # no eviction operation executed

    def test_actual_device_eviction_and_backup_without_metrics(self):
        class CT(IntEnum): FULL=0; SWA=1; MAMBA=2
        r = self.recorder()
        ns = dict(performance_events=r, time=NS(perf_counter=lambda: self.now[0]),
                  BASE_COMPONENT_TYPE=CT.FULL, ComponentType=CT, EvictResult=lambda **kw: kw)
        production("mem_cache/unified_radix_cache.py", ["_evict", "_log_write_ack_metrics"], ns, owner="UnifiedRadixCache")
        def evict(request, tracker, **kw): tracker[CT.FULL]=64; tracker[CT.SWA]=32
        cache=NS(disable=False, tree_components=[CT.FULL, CT.SWA], cache_controller=None,
                 _evict_request_by_type=lambda p: {CT.FULL:128, CT.SWA:32},
                 _evict_components=Mock(side_effect=evict), update_eviction_metrics=Mock(), metrics_collector=None)
        result=ns["_evict"](cache, object())
        self.assertEqual(result["num_tokens_evicted"], 64)
        self.assertEqual([x["values"]["freed"] for x in self.records], [64,32])
        ns["_log_write_ack_metrics"](cache, NS(num_bytes=4096, node_ids=[1,2]))
        self.assertEqual(self.records[-1]["event"], "cache.backup_complete")
        cache._evict_components.assert_called_once()

    def test_actual_transfer_queue_mutation_between_snapshots(self):
        r=self.recorder(high=2, low=0)
        ns=dict(performance_events=r)
        production("disaggregation/decode.py", ["add"], ns, owner="DecodeTransferQueue")
        q=NS(queue=[])
        ns["add"](q, object()); ns["add"](q, object())
        self.assertEqual(self.records[-1]["event"], "queue.pressure_enter")
        self.assertEqual(self.records[-1]["values"]["depth"], 2)

    def test_queue_offer_deduplicates_requeues_and_excludes_retraction(self):
        class Mode: NULL=0; PREFILL=1; DECODE=2
        r=self.recorder()
        ns=dict(performance_events=r, DisaggregationMode=Mode)
        production("managers/scheduler.py", ["_add_request_to_queue"], ns, owner="Scheduler")
        scheduler=NS(_set_or_validate_priority=lambda req: True, disaggregation_mode=Mode.NULL,
                     _abort_on_queued_limit=lambda req: False, _prefetch_kvcache=lambda req: None,
                     waiting_queue=[], processed_tokens_counter=0)
        req=NS(origin_input_ids=[0]*100, time_stats=NS(set_wait_queue_entry_time=lambda: None))
        for _ in range(3): ns["_add_request_to_queue"](scheduler,req)
        ns["_add_request_to_queue"](scheduler,NS(origin_input_ids=[0],time_stats=req.time_stats), True)
        self.now[0]=1; r.tick()
        self.assertEqual(self.records[-1]["values"]["requests"],1)
        self.assertEqual(self.records[-1]["values"]["input_tokens"],100)

    def test_real_concurrent_atomic_pipe_and_closed_consumer(self):
        path=str(SRT / "observability/performance_events.py")
        code='''import importlib.util,sys,threading,os
spec=importlib.util.spec_from_file_location("pe",sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def noise():
 for i in range(100): os.write(2,b"ordinary partial ")
t=threading.Thread(target=noise);t.start()
recorders=[m.EventRecorder(True, context={"incarnation":"writer"+str(i)},max_per_second=2000) for i in range(3)]
for r in recorders:
 for i in range(100): r.event("decode.retraction",requests=1,aborted=0,tokens_gained=8,reason="test")
for r in recorders:r.close()
t.join()
'''
        run=subprocess.run([sys.executable,"-c",code,path],capture_output=True,text=True,timeout=10)
        self.assertEqual(run.returncode,0,run.stderr)
        reader=reader_module.EventReader()
        for line in run.stderr.splitlines(): reader.feed(line)
        report=reader.report()
        self.assertEqual(report["event_counts"],{"decode.retraction":300})
        self.assertEqual(len(report["streams"]),3)
        self.assertTrue(all(x["coverage"]=="closed_contiguous" for x in report["streams"]))
        # A missing record must defeat zero/complete claims.
        reader=reader_module.EventReader()
        lines=run.stderr.splitlines()
        for line in lines:
            if '"seq":50,' not in line:reader.feed(line)
        self.assertTrue(all(x["coverage"]=="incomplete_or_live" for x in reader.report()["streams"]))

    def test_bounded_consumer_and_unclosed_stream(self):
        reader=reader_module.EventReader(max_processes=1,max_episodes=1)
        reader.feed(pe.PREFIX+'{"broken":true}')
        record=dict(schema_version=1, kind="heartbeat", delivery_seq=1,
                    context={"incarnation":"a"},loss_total={},last_event_seq=0)
        reader.feed(pe.PREFIX+json.dumps(record))
        record["context"]["incarnation"]="b"
        reader.feed(pe.PREFIX+json.dumps(record))
        report=reader.report()
        self.assertEqual(report["streams"][0]["coverage"],"incomplete_or_live")
        self.assertEqual(report["transport_gaps"],{"malformed":1,"process_limit":1})


if __name__ == "__main__": unittest.main()
