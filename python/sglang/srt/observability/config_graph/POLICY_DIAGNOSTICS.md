# Bounded scheduling-policy diagnostics

Status: implemented and default off. Full R8b/R9/R10 validate all-rank sampled policy coverage; R9/R10 also validate pass endings. R11 adds stored budget residuals, with initial all-rank warmup coverage verified. The current revision passes32 actual ARM contracts across tuning and policy tests. Isolated serving overhead and full R11 measured-phase coverage remain unvalidated. These records are absent from R6/R7. This refines gap G003 and the `prefill-ordering-and-aged-queues` rule; preserve each run's frozen source when interpreting fields.

The question is whether old requests are repeatedly bypassed by ordering, held by admission readiness, or progressing through long continuation work. Current first-forward histograms omit requests that are still waiting and later continuation/retraction waits. Gate encounter counts do not measure exposed wait.

## Source facts and population boundaries

- `SchedulePolicy.calc_priority` runs before per-candidate budget, LoRA, prefetch and staging checks. The sorted head and the non-deferred prefix group are not proven eligible requests.
- `ReqTimeStats.set_wait_queue_entry_time` uses `time.perf_counter` and overwrites the entry timestamp on reentry. A live difference is age in the current queue episode, not accumulated lifetime wait.
- Unified `Scheduler._add_request_to_queue` snapshots `processed_tokens_counter` on entry, including reentry. `run_batch` increments that counter by `extend_num_tokens` before model execution. HRRN age measures submitted prefill work, not wall time or completed GPU work.
- HRRN uses `max(0, len(origin_input_ids) - num_matched_prefix_tokens)`. SPF uses `max(1, len(origin_input_ids) + len(output_ids) - num_matched_prefix_tokens)`. An output-bearing candidate can therefore have different estimates. Normal fresh requests have no output yet; do not attribute a run's result to this distinction without establishing the relevant population.
- HRRN gives fully cached, non-deferred work priority; temporary prefix deferral still wins over that priority. Exact ratio ties use request-ID lexical order internally. Telemetry must not expose those IDs.
- `shortest_prefill_chunk_limit` reserves page-rounded tokens for shorter requests only under SPF; HRRN returns no policy-specific cap. Neither absence of that cap nor absence of `prefill.chunk_slot` proves spare capacity.

CPU examples executed the actual four sorting/cost methods and verified their ASTs match frozen source4e6e7f8. The output-bearing example is synthetic, not observed workload incidence. Scheduler lifecycle/ordering statements above are source audits, not runtime timing measurements.

## Implemented observations

| Observation | Population and collection point | Fields and limits |
|---|---|---|
| Queue snapshot | One sampled existing sorting pass; reuse sort-key evaluation to accumulate summaries without a second queue walk. | Queue count, known/unknown current-episode age count, oldest episode age, fixed age/work bands, already-forwarded and output-bearing counts, temporary-prefix-deferred count. Record selected head's age and the policy's exact work estimate after sorting. Never label these requests eligible. |
| Candidate visit | Candidates actually visited by the existing admission loop after sorting. | Bounded reason vocabulary for budget stop, LoRA skip, prefetch/staging pending and admission. Episode age/work summaries are visit-weighted, not request-weighted; skipped/unvisited candidates remain outside the population. Use existing readiness results, never call a side-effectful check again. |
| Continuation reservation | Existing policy-cap calculation and continuation admission call. | Configured/effective budget, whether a policy cap applies, returned cap when present, reserved tokens when calculated, and actual continuation tokens admitted if available at the existing call site. `None` means no policy cap, not zero tokens. Do not infer reservation from configuration alone. |

Keep numeric fields finite CPU scalars with fixed names and bounded bands. Omit unknown age/work rather than treating it as zero. Do not log request IDs, prompts, cache keys, process-group addresses or unbounded shape labels. Retain scheduler/rank/process context and producer-enable status.

Sampling is opt-in and rate-limited independently of per-candidate visits. Use one perf_counter timestamp for a sampled pass. Disabled mode must not read clocks or request fields for diagnostics. A window containing no sampled sort is unknown: earlier cadence/capacity gates may prevent sorting. No background heartbeat or whole-queue rescans are proposed.

Aggregates lose correlation between maximum age and maximum work. Selected-head paired values help, but do not identify which oldest request was bypassed. If paired oldest-age/work is needed, accumulate it during the same sampled key evaluation; do not add a separate scan. A non-deferred or visited request is still not proof of eventual admission or execution.

## Validation and deployment

Use production-method contracts to compare exact request order, return values, cache-touch counts and continuation limits with diagnostics off/on across empty queues, equal-key ties, duplicate-prefix deferral, cached work, output-bearing/retracted cases and page-budget boundaries. Unknown timestamps and reset counters must not manufacture age. Check that cancellation removes a request from later snapshots and that reentry starts a new episode without claiming lifetime wait.

Measure sampled-pass overhead across realistic queue sizes and ensure disabled mode adds no allocation, clock or scan. Then run a separate declared serving instrumentation trial, retaining source/image/workload identity and canonical replay rules. GPU/full-serving overhead and effect on ordering are unvalidated until that trial; unit checks alone are insufficient.

These observations would support an explanation, not uniquely prove causation. For example, old non-deferred candidates plus earlier inexpensive admissions can motivate a policy trial; old candidates repeatedly failing staging checks instead motivate cache-readiness investigation. Compare finalized tails, cancellations, trajectory progress and full-run throughput before accepting either diagnosis.

## Runtime contract

Set SGLANG_TUNING_POLICY_INTERVAL=10 for a sampled pass at most once per10seconds per scheduler. A tuning_policy_config startup line identifies the resolved policy and interval. tuning_policy_pass is a standalone JSON record retaining paired fields from one pass; it is not aggregated into TuningRecorder windows. Context identifies the process/ranks. Policies other than SPF/HRRN are unsupported and silent. The main unified prefill admission path is instrumented; specialized paths are not covered.

Queue age/work bands use exclusive upper bounds: age1/10/60/300/1200/inf seconds, work1/1024/16384/65536/inf tokens. Omitted bands contain no members in that sampled population; missing samples/unknown ages are not zero. Oldest age is paired with work and prefix-deferral state. Per-reason candidate oldest age/work is also paired. Visits evaluate work using the same formula at visit completion; prefix initialization may have updated the estimate since sorting.

Candidate reasons: pass_budget_stop, lora_skip, pooling_skip, request_slots_stop, prefetch_pending, prefetch_reissued, staging_pending, admitted, not_admitted_continue/no_token/other. Adder result categories alone do not identify the underlying gate. Each visited candidate gets one outcome; stopping leaves the remainder unvisited. Admission is established by actual appended object identity, not the adder return value. A continuing request is separately checked against the adder list and reports actual extend_len. Returned policy cap/reserved tokens come from the existing calculation.

CPU tests execute production sorting and continuation methods plus the actual admission loop with fake resources. They check ties, fully cached/output-bearing/deferred requests, unchanged order/readiness calls, all skip/stop categories, episode reset/cancellation/unknown age, cap boundaries, disabled-clock silence, sampling cadence and log-sink failure isolation. These do not substitute for full serving integration or measure total application overhead.

## CPU overhead check

On the login CPU,200 synthetic SPF sorts per case compared frozen4e6e7f8, diagnostics disabled, and one sampled pass including JSON serialization. At queue32/128/512, baseline medians were16.14/60.82/257.77us; disabled16.11/62.59/262.05us; sampled113.61/410.51/1602.47us. Sampled p95 was234.37/683.98/2058.39us. These are local synthetic measurements, not ARM scheduler or end-to-end overhead. Actual sink I/O and candidate loop are excluded. Full replay must establish coverage and practical overhead before a tuning decision.

## Effective-policy fallback

HRRN/LPM fall back to FCFS when the waiting queue exceeds128. SPF does not. Every sampled pass records configured policy and effective_policy after the existing policy selection. queue_snapshot_available=false indicates no supported sort-key population was observed; selected-head and age/work bands are unavailable, not zero. Candidate outcomes still describe actually visited requests, but their work estimates use the configured-policy formula, not an FCFS priority. The128/129boundary is covered by a production-method contract. This source behavior alone does not prove its incidence or performance impact in R7.


### Admission is distinct from ending a pass

R8b's first quarter exposed an interpretation hazard: `visits.admitted` establishes that the request was appended, but the same `add_one_req` call can return `OTHER` or `NO_TOKEN` and end the pass. `pass_budget_stop` counts only the earlier pre-candidate gate. These counts must not be read as all pass endings.

A later development refinement adds optional `pass_end`: cause (`pass_budget_stop`, `request_slots_stop`, `adder_result`, `queue_scan_exhausted`), the existing adder result and admitted boolean when applicable, plus visited/unvisited candidate counts. It adds no queue/readiness scan, GPU call or decision. A candidate counted at an early stop has been visited, even if readiness is unknown. Exhausting the scan is not admitting the whole queue. Adder result categories do not identify which underlying memory/compute resource caused a stop.

Thirty CPU contracts pass in total, including actual production-loop equivalence for admitted-and-stop outcomes and explicit unknowns when no scan completed. Later actual ARM contracts and full R9/R10 sampled pass-end coverage pass; frozen R8b1422f5c has no pass_end field. Isolated serving overhead remains unvalidated.


## Optional pass-end budget residuals (prepared after R10 freeze)

`pass_end.budget_remaining` copies stored CPU scalar `rem_input_tokens`, `rem_chunk_tokens`, and `rem_dllm_tokens` immediately after the existing admission loop. It does not re-evaluate budget, memory, or readiness checks. Missing fields are unknown; null preserves an unset value; negative/zero residuals remain distinct from positive values. Input and chunk exhaustion may coexist. These residuals do not uniquely attribute OTHER or NO_TOKEN, and should be read with the recorded stop site and existing gate observations.

R9 established that1172of1177sampled adder-result stops followed admission; a simple admitted count hid these endings. R9 and frozenR10 lack residual fields. The refinement passes31CPUcontracts, including production loop/finalization equivalence and no custom/tensor conversion; the later32-contract ARM suite passes. Initial R11 startup/warmup has residuals in50/50sampled passes acrossfourranks, with44input/chunk co-exhaustions. Full measured-phase coverage and unique bottleneck attribution remain pending.


R11 Q2 provides a counterexample to treating OTHER as exhausted input/chunk: one of330sampled OTHER endings retains256tokens in each budget, rejects the final candidate, and has a tracked continuation. The other329OTHER endings and14pre-candidate budget stops co-exhaust both. The one-chunk rule is a source-supported alternative, not a confirmed same-call reason. A future reason receipt should reuse the first predicate actually reached in that add call, without re-evaluating gates; preserve no-reason/legacy unknowns. All344Q2passes carry residuals. Full replay and isolated overhead remain pending.


## Returned admission sites (prepared after frozen R11)

`pass_end.adder_stop_site` records which annotated base `PrefillAdder` return branch produced a sampled `pass_budget_stop` or `adder_result` ending. There are26bounded literal sites covering direct OTHER/NO_TOKEN returns. The scheduler enables capture only for a sampled pass and clears the site before each budget check and add-request call. It copies stored metadata into the ending; it does not call a capacity/readiness function again. Admitted-and-stop remains separate from rejected-and-stop. Calls that record no annotated site emit null; older records lack the field and remain unknown. Custom adders that recover from annotated stops require a separate propagation audit; these contracts cover the base adder.

Examples: `pass_input_budget` identifies the ordered input check; with both residuals exhausted this still does not identify input as the only control. `existing_chunk_slot` identifies the SPF one-continuation return despite potentially positive raw token budgets. `chunk_alignment_empty`, `replay_chunk_budget`, `prefill_delayer`, `prefill_request_limit` and load-back sites distinguish other paths. `pass_capacity_or_mamba_slots` and `prefill_shape_capacity` deliberately retain compound scope: no extra inspection is performed to choose a physical pool. Sites are return-branch evidence, not blocked time, unique rejected-request counts or necessarily the first underlying resource failure.

Four new CPU tests execute production budget/shape-selection methods and real sample finalization; paired enabled/disabled cases preserve results and capacity-call counts. A static contract checks all26direct base return sites have literal annotations. The existing production admission-loop test additionally checks an ignored budget NO_TOKEN site cannot leak into an unannotated custom add result. Actual ARM/full serving coverage and isolated overhead are pending. No active R11 executable or historical record has been changed.

All41local observability contracts pass (22tuning,10policy,5graph-baseline,4return-site); this does not replace actual ARM and serving validation.


Qualified R11 (frozen b6df8ee) completes measured field coverage: all1383sampled policy passes carry stored residuals, with1432successful native snapshots acrossfourranks. Full output/accounting/archive qualification passes; total341,312.95590TPS/GPU is5.9902%belowbestR6. R11 lacks the newly prepared returned sites and initialized graph counters. Deployment/overhead validation for those remains separate from this historical full-run coverage.
