# Experiment #72: speculative state transactions

> Session closeout, 2026-10-25: the user authorized preserving research/results and closing experimental code PRs #74/#78/#81 without adoption. Earlier approval/pending/publication statements below describe the historical stage, not a current request to continue. The later #80 diagnostic20 includes the frozen MTP reference. See the [final disposition and combined summary](SPECULATIVE-DECODE-SESSION-CLOSEOUT-20261025.md). No acceptance threshold or original measurement was changed.

Evaluation completed with service-level failures; not all predeclared acceptance criteria were met. [Experiment #72](https://github.com/LittleLittleCloud/Infer.NET/issues/72) was explicitly approved by the user on 2026-10-24. [Code PR #74](https://github.com/LittleLittleCloud/Infer.NET/pull/74) implements the default-off synthetic proposal correctness reference; it is not a real DFlash2 draft and has no acceleration claim. This independent results branch starts from main and can be reviewed separately from implementation.

Baseline main: `4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3`. Candidate: `afdb36fd1106cf29b19b66ac94745dda481fc429`. Windows is not participating. Model remains fixed affine8 revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, FP32 KV/state, fully local Metal, fused-fp32 attention, managed prefill chunk 32 and scratch 128. No system GPU-limit or precision change was made. The original worker was restored and independently verified after evaluation.

Implementation and corrective review used GitHub Copilot CLI 1.0.103, usage-confirmed `claude-sonnet-5.5`. Independent CPU validation passed 430 unit tests and 70 HTTP/process integration checks. Measured fixture/cohort results, failures, hashes and restoration evidence follow below. No missing metric is filled with zero or an estimate.

## Retained setup failures

[Setup-failure summary](evaluations/issue72/setup-failures.json):

- An early parent scheduler treated the harness's incremental `summary.json` as proof the first Pi20 run had exited. Fixed-output fixture requests entered the same serialized worker queue. The extra fixture client was stopped; that cohort and interrupted fixtures are retained and invalid for formal A/B attribution. The cohort completed 19/20 Pi tasks with one stream-finish failure on HumanEval/149. No official-assertion result is implied by Pi completion.
- The next attempt started before the observer coordinator registered the ready worker. Three initial requests returned 503 without inference. The attempt was interrupted and retained. The replacement cohort started only after authenticated worker and coordinator readiness plus a real streamed warmup are verified. Raw observations are selected by explicit run windows, preserving the excluded startup/warmup rows.

These are parent evaluation orchestration failures, not evidence of a candidate kernel regression. They are retained alongside later runs. [Current evaluation status](evaluations/issue72/status.json).

## Measurement boundaries

For clean baseline/candidate-off, the measured aggregates are ΣinputTokens/ΣprefillSeconds and Σ(N−1)/ΣdecodeSeconds. Report coverage, official assertions, first-valid-output client TTFT, request/task counts, Pi wall time, failures and per-request resources. Enabled synthetic runs retain cycle/resource/emission-lifecycle diagnostics; their standard decode metric remains unavailable when generation-equivalent timestamps are absent.

Fixed token fixtures compare tokens, terminal and checkpoint state digests/logits, rejection offsets, EOS/output-cap boundaries, cancellation/reset and poisoned invisible KV suffixes. Diagnostic copy/replay times are not acceleration measurements. Acceptance remains exactly the preapproved Issue plan, including 20/20 official assertions, disabled-mode regression/resource limits and exact deterministic continuation. Failure or partial completion will remain explicit. HumanEval does not substitute for the fixed 128K goals.

No merge or adoption decision has been made. Results are submitted for human review; code remains a draft candidate.

## Formal main reference

The replacement cohort passed Pi 20/20 and official assertions 20/20, with zero task timeouts/errors. All 98 worker requests reconcile input/output token totals with Pi. [Audited baseline](evaluations/issue72/baseline.json).

| Metric | Measured |
|---|---:|
| Worker prefill | 270.829915 tok/s |
| Worker decode, N−1 intervals | 67.449858 tok/s |
| Client TTFT median | 10940.3325 ms |
| Pi wall time | 1148.529100 s |
| Response-sampled Metal allocation maximum | 34.729813 GiB |
| Largest per-model-request system swap-out | 377.062500 MiB |

All request-boundary power samples were AC. The model process footprint and recorded peak were 35.1G in vmmap. Metal figures are response samples, not a continuous allocator high-water trace; observed allocation and every per-request swap bracket were below their limits, without claiming unmeasured peaks. [Whole-cohort swap bracket](evaluations/issue72/baseline-resource-bracket.json) is separate from request-level sums. The original adopted baseline selection is unchanged.

## Real-model state validation

Candidate afdb36f completed 95 model checks with zero failures: 45 at fixed 128 outputs, 5 at fixed 4096 outputs, and 45 at natural EOS (128 output cap, 86 emitted tokens for this prompt). Counts include warmup, formal and lifecycle rows; fixed-output configurations retain one excluded warmup plus three formal runs. [Hardware summary](evaluations/issue72/hardware-summary.json), [fixed input/reference identity](evaluations/issue72/fixture-reference.json).

All rejection offsets 0–7, all/none acceptance and mixed patterns were exercised at 128 outputs with block 8. The 4096 mixed/block 4 case repeatedly covers every offset 0–4 including all/none acceptance; it is not an exhaustive long-output cross product. Natural EOS cases verify exact terminal position, pending token, state and logits, including EOS inside an accepted block. Poisoned invisible KV suffixes, cancellation/reset and short requests after a long state are also covered. CPU and native-buffer tests cover the remaining block/cap combinations.

The three short fixed-output formal runs show approximately 1.05–1.07 ms per snapshot capture and 1.02–1.05 ms per restore for the selected patterns. These are diagnostic fixture averages, not an unperturbed DRAM benchmark or a DFlash2 speed prediction. The snapshot holds 65,863,680 host bytes (62.8125 MiB). Raw elapsed times include correctness instrumentation and periodic footprint inspection.

Whole-process swap counters in the hardware summary include loading/warmup and cannot be assigned to individual formal fixture runs. Loading produced substantial system swap, so no zero-swap claim is made. Service-cohort per-request resources are measured separately.

The enabled standard decode value is null because this reference does not collect a generation-equivalent first/last-ready clock. This is an instrumentation limitation of this harness, not a claim that block decoders can never report a valid N−1 aggregate. A future native verifier must define that clock explicitly; network block intervals and lifecycle totals must not silently substitute for it.

## Candidate disabled formal cohort

The default-off cohort attempted all 20 tasks: Pi completed 19/20 with one stream-finish error, while official assertions passed 20/20, including the saved HumanEval/149 solution. No answer was regenerated or dropped. All 104 worker requests have measured prefill/decode timings and request-resource brackets; only 103/104 have a first valid output and TTFT. The failed stream had 3,270 input and 102 output tokens in worker evidence but no Pi usage, so Pi/worker token totals do not reconcile. Both totals are retained. [Audited disabled cohort](evaluations/issue72/off.json).

| Metric | Main | Candidate off | Relative change |
|---|---:|---:|---:|
| Worker prefill tok/s | 270.829915 | 271.061100 | +0.0854% |
| Worker decode tok/s | 67.449858 | 67.390122 | −0.0886% |
| Client TTFT median ms | 10940.332500 | 11018.170583 | +0.7115% |
| Pi wall seconds | 1148.529100 | 1190.371060 | +3.6431% |
| Response-sampled Metal GiB | 34.729813 | 34.729813 | 0 |
| Largest request swap-out MiB | 377.062500 | 159.937500 | — |

These aggregate timing changes are inside the proposed 5% bounds (prefill time per input token improved approximately 0.0853%). Separate Pi agent histories differ; this is not a paired identical-token kernel speed experiment. The missing TTFT and Pi stream failure prevent an unconditional acceptance claim. The disabled point is marked failed in the trend source while retaining its measured timings and official pass rate.

All eight disabled fixed-output fixtures (128 and 4096 outputs, one warmup plus three formal runs each) matched the saved main token IDs. Every baseline/off fixture reported exactly 37,290,852,352 Metal bytes: no additional disabled-mode GPU buffers were observed. This evidence does not measure a continuous allocator high-water mark.

## Reproduction and evidence

[Exact fixture argument vectors](evaluations/issue72/hardware-commands.json) preserve the measured matrix, warmup/formal counts and native profile, with local paths replaced by named placeholders. `LOCAL_REFERENCE_JSON` contains the saved 32 input IDs and 4096 main greedy output IDs; its prompt/input identity and canonical output hash are published in the fixture reference. The source harness `scripts/run-speculative-state.py` and `docs/SPECULATIVE-STATE.md` in code PR #74 describe its JSON format and first-divergence exports. The measured matrix is the smaller explicit matrix above, not the script's full default matrix.

[Main package manifest](evaluations/issue72/baseline-package-manifest.json) and [candidate package manifest](evaluations/issue72/candidate-package-manifest.json) retain all nine file hashes. Every file was verified, and authenticated health matched the expected immutable build SHA before and after each cohort. The public cohort summaries bind the harness, dataset, runtime health, official assertion output, task/observation logs and package manifest by SHA256. Raw logs, generated solutions and reference tokens remain in local ignored artifacts; hashes do not imply public download access.

Service cohorts use the same fixed Pi command: `python3 scripts/benchmark-pi-humaneval.py --output NEW_OUTPUT_DIRECTORY --token-file PRIVATE_TOKEN_FILE --endpoint http://127.0.0.1:5280 --cohort cohort --seed 0 --timeout 300 --dataset HumanEval.jsonl.gz --context-window 32768 --max-tokens 4096 --endpoint-build MEASURED_SHA --model-revision e06a74e6236a60c8367e1a3214e83d8b61b637b0`. Pilot runs use `--cohort pilot`. No other model client runs concurrently. Authenticated readiness, observer registration and a real streamed warmup precede each window; candidate modes use a 4,213-token capacity warmup to reserve 6,144 KV slots. Main already retained the same capacity after excluded startup work; every measured response in all three cohorts reports capacity 6,144. Request state resets and prefix hits remain zero.

The enabled worker adds `--speculative-synthetic mixed --speculative-block-tokens 4 --kv-transfer final`. Main/off retain the existing `kv-transfer incremental` option, unused in the explicitly local path: there is no handoff or Windows execution. Both candidate modes use the same executable/native library. Two loopback observation forwards collect worker results and first valid output; Pi-to-observer dispatch is outside the TTFT boundary. Official assertions use `scripts/eval-pi-humaneval-docker.py` with the SOP-pinned Python image and a failing negative control; no external tests are supplied to Pi.

[Environment observations](evaluations/issue72/environment.json) record AC/High Power and default wired limit. Thermal samples report that no warning level was recorded, rather than an unavailable temperature. Memory pressure was sampled late in the disabled cohort and during enabled evaluation; it was not traced continuously. The supplemental two-second service sampler starts midway through the disabled formal run. Response-sampled allocation, intermittent vmmap process peaks and whole-system swap counters do not establish an unobserved continuous allocator peak or attribute every swapped page to this worker. These resource-observability limits remain open in acceptance.

## Enabled synthetic formal cohort

Both candidate pilot runs completed 5/5 with zero Pi errors/timeouts. The enabled formal cohort attempted all 20 tasks: Pi completed 19/20, and official assertions passed 19/20. HumanEval/94 ended without `finish_reason` and produced no `solution.py`; it remains in the denominator as an error/missing solution. There were no task timeouts or selected answer retries. The enabled run completed HumanEval/149, which had failed at the stream boundary in the disabled run. This distribution alone does not establish the cause or exclude a regression. Exact paired application requests were not retained, so the cause of the stream failures remains unresolved. [Audited enabled cohort](evaluations/issue72/on.json).

All 100 worker requests have separate prefill/resource diagnostics; 99/100 have a valid first output/TTFT and matching Pi token counts. The failed request records 1,822 input and 566 output tokens in worker evidence, absent from Pi usage. Standard decode intervals/seconds/throughput remain null. Raw worker-prefill diagnostics aggregate to 271.612400 tok/s; they are retained in JSON and excluded from standard speed plots for this distinct synthetic correctness configuration. TTFT median is 17,547.334833 ms and Pi wall time is 1,957.882213 s. These serial oracle/replay costs do not measure an accelerated draft/verifier.

Response-sampled Metal allocation remains 34.729813 GiB. Largest request swap-out is 245.187500 MiB; all 100 resource brackets are present and power samples are AC. A snapshot holds 65,863,680 host bytes. All three model processes report physical footprint/recorded process peak as the literal vmmap value `35.1G`; that is not a continuous Metal allocation measurement. [Resource brackets](evaluations/issue72/resource-brackets.json) separate whole-cohort system swap from request sums: main 690.171875 MiB, off 299.968750 MiB, on 245.187500 MiB. Loading/warmup are excluded from these cohort brackets and retained separately in raw artifacts.

## Acceptance and next work

[Criterion-by-criterion comparison](evaluations/issue72/comparison.json) records exact fixture continuation as passed and observed disabled timing deltas within the 5% bounds. The disabled TTFT coverage is incomplete (103/104), enabled official correctness is 19/20, and the continuous allocator high-water criterion is not established. The experiment therefore does **not** meet all predeclared acceptance criteria. The code remains default-off and unadopted in draft PR #74; the independent results PR #75 preserves the failed outcome. There is no acceleration, fixed 128K acceptance, baseline-selection or production-deployment change.

Before adoption, isolate streaming failures using captured paired application requests and collect the missing allocation high-water evidence. The later speculative path remains: exact hidden-tap matching, real BF16 draft forward, multi-row target logits with numerical-order validation, efficient rollback, then integrated DFlash2 A/B. Each later experiment requires its own approved proposal. S0 confirms selected state-recovery cases; it does not resolve the Splash decode gap.

## Service restoration

The controller's finalizer restored the original LaunchAgent. Independent verification confirms original build `0.3.0+cd6f7da401fadf044dac13e87b415d27bb5eadfb`, worker PID 71597 in Ready, healthy coordinator registration and two successful real chat probes across controller/independent checks. All nine original package hashes match. Known experiment ports 5280/5281/5282/5290/5291/5292 are closed, owned experiment processes and the supplemental sampler are stopped, and `iogpu.wired_limit_mb` is 0. Windows was not modified. [Restoration evidence](evaluations/issue72/restoration.json).

Validation: 430 unit tests, 70 managed HTTP/process checks, 70 Native AOT integration checks, 587 native Metal-buffer transaction checks and 95 real-model fixture checks passed. The native buffer test uses CPU mutations over real Metal buffers and is not a numerical model/operator oracle; real-model digest/logit checks are separate. [Validation log hashes](evaluations/issue72/validation.json), [Copilot model provenance](evaluations/issue72/copilot-provider.json). Code CI passed on macOS, Linux and Windows. Site verification covers 26 Python and 27 offline DOM checks plus the site build; it does not claim browser layout acceptance. No merge or live Pages publication was performed.

[Frozen raw evidence hashes](evaluations/issue72/evidence.json) bind the retained logs after all experiment writers stopped. No worker crash/OOM was recorded, all 302 observed cohort requests used real metal-local execution, and CPU weight offload was not configured. Full per-request tensor/logit finiteness traces were not retained; selected checkpoint equality is not a substitute for such a trace. NaNs deliberately placed in invisible KV suffixes are fixture inputs, not a claimed runtime numerical failure.
