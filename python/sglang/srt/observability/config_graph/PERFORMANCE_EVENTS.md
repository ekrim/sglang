# Performance operation events and episodes

The event layer records operations and condition transitions when they occur. It complements aggregate counters and PD request ownership; it does not replace them. `event_producer_audit.json` inventories all 33 campaign graph mechanisms, their controllable knobs, existing observations, candidate events and remaining gaps. `performance_event_contracts.json` is the machine-readable definition. Native qualification remains separate.

Enable `SGLANG_PERFORMANCE_EVENTS=1` on every intended scheduler process. Defaults: 1,024 pending records, 200 events per monotonic-second bucket, queue high/low thresholds 128/64 requests. Override rate and thresholds with `SGLANG_PERFORMANCE_EVENTS_RATE`, `SGLANG_PERFORMANCE_EVENTS_QUEUE_HIGH`, `SGLANG_PERFORMANCE_EVENTS_QUEUE_LOW`. These thresholds describe an analytical episode, not physical capacity. Existing guarded admission/dispatch producers also need a nonzero tuning observation interval. Disabled queue hooks avoid length/timestamp reads; no event code reads tensor values, adds GPU synchronization, scans queues, or calls readiness again.

| Producer | Trigger | Useful knob/mechanism | Interpretation limit |
|---|---|---|---|
| Queue initial/pressure-enter/pressure-exit | Actual mutation of execution, bootstrap, preallocation, transfer, retraction or inflight queue | Running requests, admission budgets, cadence, PD reservation | Recovery may be cancellation or flush; join holders and completions |
| Arrival window/surge-enter/surge-exit | Scheduler tick/first queue offer; at least one-second windows | Concurrency, worker distribution, workload mix | Backend offers, not upstream HTTP arrivals; subsecond shape unresolved |
| Device eviction | Completed unified-cache eviction, per component | Device capacity, reservation, shared pools | Host copy may survive; repeated inclusive duration is not additive |
| Host eviction | Completed host eviction, including zero freed | HiCache ratio/size, working set | Logical component units, not physical bytes; buffer-only bypass emits no operation |
| Backup/loadback complete | Existing cache acknowledgment | Host retention, transfer/admission | No per-request or exposed-transfer timing |
| Decode retraction | Actual retraction result | Reservation, running requests, device pools | Distinguishes KV pressure/test; not proof of leak |
| Decision changed | Change in an observed admission/forward domain | Budget, cadence, graph dispatch | Last observed decision, not a blocked-time span or per-kernel replay count |

Records have occurrence monotonic/wall timestamps, attempted event sequence, process incarnation and ranks. The incarnation is shared with the PD ledger. A bounded daemon emits small single-write records plus independent one-second sink heartbeats. Delivery sequence, cumulative rate/queue/serialization/sink losses, and a closure receipt support completeness checks. A heartbeat establishes sink activity only. Abrupt termination or a blocked sink can prevent the last loss/closure report. The consumer must also verify the expected producer set against actual workers; an absent process cannot certify zero events. Pipe atomicity requires native validation through the actual merged collector.

Run `python python/sglang/srt/observability/performance_event_reader.py LOG [LOG ...]` on one chronologically ordered log sequence per process (do not double-feed live and archived copies). It retains bounded condition episodes and marks missing, duplicate, malformed, unclosed, or lossy streams incomplete. Its output proposes temporal candidates; it does not update a causal claim automatically.

After each run:

1. Establish activation, producer cardinality, clock/rank scope, loss and closure before interpreting absence.
2. Retain pre/during/post windows around each episode: arrival volume/mix, queue depth/peak/recovery, cache operations, dispatch/admission observations and same-incarnation PD holders/stages/releases. Keep original evidence immutable.
3. Compare competing explanations and matched non-event windows. Rank-local ordering supports a candidate mechanism; coincidence alone does not establish causality.
4. Append an empirical explanation revision with outcome/prediction/evidence references, prior claims and counterevidence. Strengthen only within the tested conditions; don't overwrite old findings or treat one recent run as universal.
5. If the event cannot distinguish mechanisms, record a specific diagnostic experiment or observation gap. Change source graph/contracts only when producer semantics/code changes, and requalify the changed path before another dependent performance run.

Retained gaps include router dispatch permits (a high-priority dependency for `decode_dispatch_limit` attribution), upstream offered traffic/subsecond burst timing, physical shared-KV ownership, wire progress, dynamic Engram backing and per-layer replay cost. PP/CP/PDMux/DLLM and non-unified caches are not covered by the current campaign-path qualification. The existing R23 frozen source does not contain this candidate.
