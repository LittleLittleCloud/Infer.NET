# Experiment #72: speculative state transactions

Evaluation is in progress. [Experiment #72](https://github.com/LittleLittleCloud/Infer.NET/issues/72) was explicitly approved by the user on 2026-10-24. [Code PR #74](https://github.com/LittleLittleCloud/Infer.NET/pull/74) implements the default-off synthetic proposal correctness reference; it is not a real DFlash2 draft and has no acceleration claim. This independent results branch starts from main and can be reviewed separately from implementation.

Baseline main: `4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3`. Candidate: `afdb36fd1106cf29b19b66ac94745dda481fc429`. Windows is not participating. Model remains fixed affine8 revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, FP32 KV/state, fully local Metal, fused-fp32 attention, managed prefill chunk32 and scratch128. No system GPU-limit or precision change was made. The original worker will be restored after evaluation.

Implementation and corrective review used GitHub Copilot CLI 1.0.103, usage-confirmed `claude-sonnet-5.5`. Independent CPU validation passed 430 unit tests and 70 HTTP/process integration checks. GPU/model fixture results, fixed 5/20 cohorts, official assertions, package hashes and service restoration will be appended after they complete; no missing metric is filled with zero or an estimate.

## Retained setup failures

[Setup-failure summary](evaluations/issue72/setup-failures.json):

- An early parent scheduler treated the harness's incremental `summary.json` as proof the first Pi20 run had exited. Fixed-output fixture requests entered the same serialized worker queue. The extra fixture client was stopped; that cohort and interrupted fixtures are retained and invalid for formal A/B attribution. The cohort completed 19/20 Pi tasks with one stream-finish failure on HumanEval/149. No official-assertion result is implied by Pi completion.
- The next attempt started before the observer coordinator registered the ready worker. Three initial requests returned 503 without inference. The attempt was interrupted and retained. A replacement cohort starts only after authenticated worker and coordinator readiness plus a real streamed warmup are verified. Raw observations are selected by explicit run windows, preserving the excluded startup/warmup rows.

These are parent evaluation orchestration failures, not evidence of a candidate kernel regression. They will not be silently replaced by later successful runs. [Current evaluation status](evaluations/issue72/status.json).

## Measurement boundaries

For clean baseline/candidate-off, collect complete worker request timings and aggregate ΣinputTokens/ΣprefillSeconds and Σ(N−1)/ΣdecodeSeconds. Report coverage, official assertions, first-valid-output client TTFT, request/task counts, Pi wall time, failures and per-request resources. Enabled synthetic runs retain cycle/resource/emission-lifecycle diagnostics; their standard decode metric remains unavailable when generation-equivalent timestamps are absent.

Fixed token fixtures compare tokens, terminal and checkpoint state digests/logits, rejection offsets, EOS/output-cap boundaries, cancellation/reset and poisoned invisible KV suffixes. Diagnostic copy/replay times are not acceleration measurements. Acceptance remains exactly the preapproved Issue plan, including 20/20 official assertions, disabled-mode regression/resource limits and exact deterministic continuation. Failure or partial completion will remain explicit. HumanEval does not substitute for the fixed 128K goals.

No merge or adoption decision has been made. This record will be completed with actual results and restoration evidence before requesting review.

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

The three short fixed-output formal runs show approximately 1.05–1.07ms per snapshot capture and 1.02–1.05ms per restore for the selected patterns. These are diagnostic fixture averages, not an unperturbed DRAM benchmark or a DFlash2 speed prediction. The snapshot holds 65,863,680 host bytes (62.8125 MiB). Raw elapsed times include correctness instrumentation and periodic footprint inspection.

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

Service cohorts use the same fixed Pi command: `python3 scripts/benchmark-pi-humaneval.py --output NEW_OUTPUT_DIRECTORY --token-file PRIVATE_TOKEN_FILE --endpoint http://127.0.0.1:5280 --cohort cohort --seed 0 --timeout 300 --dataset HumanEval.jsonl.gz --context-window 32768 --max-tokens 4096 --endpoint-build MEASURED_SHA --model-revision e06a74e6236a60c8367e1a3214e83d8b61b637b0`. Pilot runs use `--cohort pilot`. No other model client runs concurrently. Authenticated readiness, observer registration and a real streamed warmup precede each window; a 4,213-token capacity warmup fixes retained KV capacity at 6,144 before measurements.

The enabled worker adds `--speculative-synthetic mixed --speculative-block-tokens 4 --kv-transfer final`. Main/off retain the existing `kv-transfer incremental` option, unused in the explicitly local path: there is no handoff or Windows execution. Both candidate modes use the same executable/native library. Two loopback observation forwards collect worker results and first valid output; Pi-to-observer dispatch is outside the TTFT boundary. Official assertions use `scripts/eval-pi-humaneval-docker.py` with the SOP-pinned Python image and a failing negative control; no external tests are supplied to Pi.

[Environment observations](evaluations/issue72/environment.json) record AC/High Power and default wired limit. Thermal samples report that no warning level was recorded, rather than an unavailable temperature. Memory pressure was sampled late in the disabled cohort; it was not traced continuously. The supplemental two-second service sampler starts midway through the disabled formal run. Response-sampled allocation, intermittent vmmap process peaks and whole-system swap counters do not establish an unobserved continuous allocator peak or attribute every swapped page to this worker. These resource-observability limits remain open in acceptance.
