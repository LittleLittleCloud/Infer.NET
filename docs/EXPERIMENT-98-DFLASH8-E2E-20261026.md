# Experiment 98: real DFlash8 E2E, power-profile rerun in progress

Status: a complete diagnostic rerun is in progress after verifying a 140W adapter and High Power in System Settings. The earlier run stopped after 57 complete fixed-control requests (42 formal and 15 excluded warmups), with one additional main request interrupted. The known independent draft numerical gate failed. The declared High Power condition was not adequately verified, and the battery fell from 72% to 6% while connected to a reported 35W adapter. **No qualified main-versus-native E2E conclusion or positive code-adoption recommendation is available.** HumanEval was not run. Existing timing observations, slow trials and incomplete attempts are retained.

Approved [Issue #98](https://github.com/LittleLittleCloud/Infer.NET/issues/98), [candidate code PR #99](https://github.com/LittleLittleCloud/Infer.NET/pull/99), independent [results PR #100](https://github.com/LittleLittleCloud/Infer.NET/pull/100). Main is `4f51f241e87999c8cc3f79d733cd95ade9c711cd`; frozen E2E candidate is `9c073ff81082fe79b1cd7a2d9471d8c2b8f3deaa`. The baseline and production defaults remain unchanged. Neither PR is merged.

## Power correction and stop

The earlier interim label “AC / High Power” was inferred from numeric `pmset powermode=2`; that was insufficient verification. On 2026-10-26 around 19:49 UTC, the system reported an Apple 35W USB-C adapter connected and battery charge at 6%. The request-boundary battery readings show sustained depletion from 72% at the corrected run's start. The final native 8K/4096 trial slowed from about 28.71 to 23.03 token/s, with unchanged output tokens, cycle/acceptance counts and histogram, and zero new swap-out. GPU command time also increased.

`system_profiler` reports AC `LowPowerMode=Yes` / `HighPowerMode=No`, while a separate Foundation query after stopping reports `isLowPowerModeEnabled=false`. The contradiction persists in the after-stop capture. **Effective Low Power during earlier timings is not established.** The exact cause of the slow trial is also not established; no thermal or power-throttling cause is inferred solely from these snapshots. At the earlier stop, the Mac was locked, preventing System Settings verification; no lock bypass or power-setting change was attempted.

The owned controller was interrupted, all experiment workers/coordinators were cleaned up, and the original LaunchAgent and package/configuration/plist were restored unchanged. The original address-bind failure predates this experiment; post-restore health was Loading/non-Ready, and no real-request restoration is claimed. Temporary sleep prevention and allocation monitoring ended with the controller. See [power evidence and battery trajectory](evaluations/issue98/power-condition.json), [runtime/restoration](evaluations/issue98/runtime-and-restoration.json) and [attempt ledger](evaluations/issue98/attempt-ledger.json).

Resumption requires adequate sustained supply, battery recovery and actual energy-mode verification. A changed supply requires fresh affected comparison groups under one recorded condition; existing slow/failed observations must not be replaced or combined into a qualified result.

## Power-profile restart, 20:12 UTC

The user requested continued verification after the physical power adjustment. Read-only checks now identify an Apple 140W adapter connected to AC and battery charge increasing from 14%. Both Energy Mode controls in System Settings show High Power. The agent made no power-setting change. `system_profiler` mode fields still contradict the UI; retain this discrepancy rather than inferring earlier effective Low Power or a causal explanation for the slow trial.

Run `e2e-20261026T201255Z` repeats the entire preregistered five-shape/four-arm matrix, with one excluded warmup and three formal trials per arm/shape, using the same frozen packages and precision. Earlier trials remain separate, including the slow and interrupted requests. Request boundaries and a separate periodic watcher verify AC, the 140W adapter, and charge not below the 14% restart level; the watcher also checks the unchanged numeric AC mode setting. Existing allocation and swap stop conditions remain unchanged. Fixed controls precede the approved pilot5 A/B/C/D and diagnostic20 A/B/D cohorts. This continuation does not waive the failed numerical gate; rerun and HumanEval summaries are pending.

## Frozen implementation and workload

One M5 Pro 20-core GPU / 48 GiB, full-local Metal, concurrency one, no prefix cache, `iogpu.wired_limit_mb=0`; no Windows/CUDA work. Target `mlx-community/Qwen3.6-35B-A3B-8bit@e06a74e6236a60c8367e1a3214e83d8b61b637b0`, affine8/group64, FP32 operands/KV/recurrence. Draft `incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b6923ad6c14cb1bb41c37a9041446496ab6`, BF16 checkpoint matrices with FP32 arithmetic/context. All target shards/configuration/tokenizer and draft hashes were reverified after GPU timing; [model provenance](evaluations/issue98/model-provenance.json) retains the exact checksums.

A is fresh scalar main. B is candidate-off. C uses real DFlash8 proposals, actual scalar verification and snapshot/replay. D uses the same draft width/selection mechanism with native eight-row verification and direct prefix commit. No prerecorded proposals or padded draft blocks impersonate generation; both enabled controls use scalar output tails. Native D performs no accepted-prefix target replay.

All arms use explicit FP32 GEMM/KV, managed chunk32/scratch128, matrix0/sequential DeltaNet, fused-FP32 decode and local threshold262144. This fixed application profile is not Splash's default fast-prefill/Q8 profile. Draft-enabled combined capacity is 36864 for the approved 32768+4096 control; planned Pi context remains 32768. Immutable AOT package/runtime hashes are retained, not inferred from checkout HEAD.

The planned shapes were 21/128, 8192/128, 32768/128, 8192/4096 and 32768/4096. Shape-wise arm order was fixed before timing as ABCD, DCBA, BADC, CDAB, ABCD. Each arm/shape has one excluded warmup and three retained formal trials. This counterbalances shape-level load order rather than randomizing individual trials. State is reset for every request; capacity may remain allocated after warmup growth. All actual input counts and fixture hashes are retained in [fixed controls](evaluations/issue98/fixed-controls.json).

The same two loopback observation hops are used in all arms. Worker decode is sum(N−1)/sum(first-to-last generated-token readiness seconds), including draft/verify/commit work; tokens in one verified block share one timestamp. Client request wall includes dispatch, tokenization, reset, prefill, complete generation and response delivery, with fixture reading/JSON preparation outside t0. Native token-ID receipt TTFT is separate from compatible text/tool TTFT. These local worker prefill observations do not establish 128K distributed acceptance.

## Observed controls, not qualified High Power results

Values below retain all completed formal trials. Worker throughput uses aggregate token/time sums; client wall is the median. The power-verification limitation applies to every row. A long-main warmup is not substituted for missing formal main trials.

|Input/output|Arm|Formal trials|Prefill token/s|Decode token/s|Median request s|Max new swap-out MiB|
|---|---|---:|---:|---:|---:|---:|
|21/128|Main|3|171.628|66.315|2.038|0.000|
|21/128|Candidate off|3|172.931|65.987|2.049|0.000|
|21/128|DFlash8 scalar/replay|3|165.011|20.762|6.248|0.000|
|21/128|DFlash8 native/commit|3|157.659|46.052|2.899|0.000|
|8192/128|Main|3|170.698|60.191|50.107|0.000|
|8192/128|Candidate off|3|170.748|60.234|50.098|0.000|
|8192/128|DFlash8 scalar/replay|3|166.355|13.783|59.027|0.000|
|8192/128|DFlash8 native/commit|3|163.649|25.540|55.042|0.000|
|32768/128|Main|3|155.758|45.922|213.159|2.812|
|32768/128|Candidate off|3|155.750|45.941|213.169|0.000|
|32768/128|DFlash8 scalar/replay|3|155.622|13.740|219.824|133.672|
|32768/128|DFlash8 native/commit|3|155.581|22.008|216.415|0.000|
|8192/4096|Main|Not Obtained|Not Obtained|Not Obtained|Not Obtained|Not Obtained|
|8192/4096|Candidate off|Not Obtained|Not Obtained|Not Obtained|Not Obtained|Not Obtained|
|8192/4096|DFlash8 scalar/replay|3|171.124|15.539|311.401|0.000|
|8192/4096|DFlash8 native/commit|3|169.450|26.528|190.512|0.000|
|32768/4096|Main|Not Obtained|Not Obtained|Not Obtained|Not Obtained|Not Obtained|
|32768/4096|Candidate off|Not Obtained|Not Obtained|Not Obtained|Not Obtained|Not Obtained|
|32768/4096|DFlash8 scalar/replay|Not Obtained|Not Obtained|Not Obtained|Not Obtained|Not Obtained|
|32768/4096|DFlash8 native/commit|Not Obtained|Not Obtained|Not Obtained|Not Obtained|Not Obtained|

All completed output token sequences agree across the observed arms/trials for each shape. The complete 4096-token native outputs also agree with the excluded scalar-main warmup; that warmup does not establish formal main performance.

The observed 8K/4096 native aggregate is 70.71% faster than scalar DFlash, including the slow third trial. Native request walls were 190.512, 190.503 and 227.173 seconds; worker decode was 28.710, 28.711 and 23.026 token/s. No slow sample was removed or retried. Native final-quarter/whole-decode ratios were approximately 1.105, 1.105 and 1.081. These are candidate worker readiness clocks; main scalar profiling windows time forwards and cannot reconstruct equivalent worker readiness, so baseline worker-tail performance is Not Obtained. Client receipt tail ratios remain a separate clock.

For the first native long formal trial, target verification took 117.970 seconds, draft 22.108, ingest 1.359 and direct commit 1.055, with zero replay. The corresponding scalar-DFlash request used 53.572 seconds of rejection replay. Both recorded 1395 cycles, 9765 proposals and 2694 accepted proposals with identical acceptance histograms. The observed mechanism removes replay and reduces verification cost; it does not establish an improvement over formal scalar main under qualified power conditions.

## Numerical, state and resource checks

See [gates](evaluations/issue98/gates.json). Native Metal compilation, independent target-state checks and all 19 short full-cycle cases pass. Actual capacity growth, ring eviction, invisible-slot poisoning, cancellation/resumption, natural EOS and exact greedy output/final target-state comparisons at positions 2047, 4093, 8192 and 32768 pass. Independent shared embedding/head, fresh context and ring checks pass. Release/Native AOT builds pass, with 171 CPU tests passed, 13 opt-in model facts skipped and 38 controlled AOT protocol checks passed.

The independent draft oracle fails at `HOut layer 1`: relative L2 `8.291167981301167e-7`, maximum absolute error `0.0107421875`, under unchanged relative L2 <1e-4 AND maxabs <0.01 criteria. Native activation magnitude at that checkpoint reaches about 11192; this does not waive the absolute-error gate. Reference assertions after the first failure are not claimed passed. Issue #98 permits diagnostic E2E continuation for this known draft-only failure; qualification remains failed.

The actual native library SHA-256 `a3d606266947a6b44214a2396307e3173c2331448b94c79a0bad5e00cffb734b` matches short gates, long gates and the frozen AOT package. Managed backend/native product sources have no diff from the SHA-identified short-gate backend through frozen E2E. The long tool's informational version lacks an embedded commit (`0.3.0`); its clean-controller SHA is separate evidence and is not represented as an embedded runtime version.

No OOM or target-token mismatch was observed. All completed formal request swap-out brackets stay below 512 MiB; the maximum is 133.672 MiB. The 32K native sampled allocation peak is 37.485 GiB, under 38.5 GiB but above the device's recommended 37.440 GiB working set, with a nonzero submission-above-working-set counter. Those counters do not prove physical paging, residency loss or the cause of slow throughput. In the first short-gate component, 1454.484375 MiB system swap-out occurred before readiness; post-load checks added none. Load/warmup and steady request records remain separate.

Traced C/D formal GPU submission increments match interface submission counts. GPU milliseconds/submissions include draft and ingest native calls, while their dispatch counts are not exported. Allocator sampling is not continuous driver peak coverage. Process footprint peak, continuous driver residency/peak and numeric temperature measurements are Not Obtained. Native draft active-finiteness checks cover selected small final outputs, not every intermediate throughout all requests. Snapshot/replay wall timers overlap: replayMs is included in legacy commitMs and must not be added twice.

## Transport, missing application evaluation and retained interruptions

A separately committed fake-worker regression reproduces an uncaught malformed-tool-call exception. The compatibility layer now returns an explicit model-output error; malformed calls remain failures. This does not prove that all historical truncated streams had that cause. Terminal candidate worker clocks retain generated-token readiness on failure/cancellation; they do not invent client receipt or successful application completion.

Pilot5 A/B/C/D and diagnostic20 A/B/D were **not run** after the power stop. HumanEval prefill/decode, Pi wall, official assertions/negative controls, request reconciliation, compatible TTFT and task pass rate are Not Obtained or unverified. They are not filled with zeros or inferred from fixed native stress requests. No HumanEval trend point is fabricated.

Retained controller failures include an out-of-range 3600-second deadline causing HTTP 500 before the first excluded warmup, a superseded run interrupted to move client t0 after input preparation, and a preflight-only TCP TIME_WAIT bind-probe failure before model loading/service pause. The corrected control deadline is 1800 seconds; planned Pi timeout remains 300 seconds. Finally, the power stop interrupted one main 8K/4096 formal request; no complete worker-result timing is inferred for it. See the [attempt ledger](evaluations/issue98/attempt-ledger.json).

## Disposition and evidence

Keep code PR #99 unadopted and the experiment Issue open for manual review/resumption. Results PR #100 preserves this failed/interrupted evaluation independently of candidate code. No merge, production-default change, numerical relaxation, memory-limit change, unrelated process/container termination or Windows work was performed.

Raw logs, full states, prompts and output text remain restricted local artifacts; the public [evidence inventory](evaluations/issue98/evidence.json) provides relative names, sizes and hashes, not download links. The original service's pre-existing readiness failure and the unresolved power condition remain explicit. Correct these conditions and rerun fresh affected groups before making any qualified E2E or adoption claim.
