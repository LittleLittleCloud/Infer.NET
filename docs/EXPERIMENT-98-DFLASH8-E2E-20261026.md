# Experiment 98: native DFlash8 E2E, fixed controls complete

Status: the verified-power fixed rerun completed 80 requests (60 formal and 20 excluded warmups), with exact target output agreement and no new target/state/resource stop. Both long-output tiers are slower than frozen main. The independent draft numerical gate remains failed. HumanEval run `human-20261027T003048Z` is in progress; complete application metrics are pending. The candidate remains unadopted, and all earlier interrupted/slow/failed observations are retained separately.

Approved [Issue #98](https://github.com/LittleLittleCloud/Infer.NET/issues/98), [candidate code PR #99](https://github.com/LittleLittleCloud/Infer.NET/pull/99), independent [results PR #100](https://github.com/LittleLittleCloud/Infer.NET/pull/100). Main is `4f51f241e87999c8cc3f79d733cd95ade9c711cd`; frozen E2E candidate is `9c073ff81082fe79b1cd7a2d9471d8c2b8f3deaa`. The baseline and production defaults remain unchanged. Neither PR is merged.

## Power correction and stop

The earlier interim label “AC / High Power” was inferred from numeric `pmset powermode=2`; that was insufficient verification. On 2026-10-26 around 19:49 UTC, the system reported an Apple 35W USB-C adapter connected and battery charge at 6%. The request-boundary battery readings show sustained depletion from 72% at the corrected run's start. The final native 8K/4096 trial slowed from about 28.71 to 23.03 token/s, with unchanged output tokens, cycle/acceptance counts and histogram, and zero new swap-out. GPU command time also increased.

`system_profiler` reports AC `LowPowerMode=Yes` / `HighPowerMode=No`, while a separate Foundation query after stopping reports `isLowPowerModeEnabled=false`. The contradiction persists in the after-stop capture. **Effective Low Power during earlier timings is not established.** The exact cause of the slow trial is also not established; no thermal or power-throttling cause is inferred solely from these snapshots. At the earlier stop, the Mac was locked, preventing System Settings verification; no lock bypass or power-setting change was attempted.

The owned controller was interrupted, all experiment workers/coordinators were cleaned up, and the original LaunchAgent and package/configuration/plist were restored unchanged. The original address-bind failure predates this experiment; post-restore health was Loading/non-Ready, and no real-request restoration is claimed. Temporary sleep prevention and allocation monitoring ended with the controller. See [power evidence and battery trajectory](evaluations/issue98/power-condition.json), [runtime/restoration](evaluations/issue98/runtime-and-restoration.json) and [attempt ledger](evaluations/issue98/attempt-ledger.json).

Resumption requires adequate sustained supply, battery recovery and actual energy-mode verification. A changed supply requires fresh affected comparison groups under one recorded condition; existing slow/failed observations must not be replaced or combined into a qualified result.

## Power-profile restart, 20:12 UTC

The user requested continued verification after the physical power adjustment. Read-only checks now identify an Apple 140W adapter connected to AC and battery charge increasing from 14%. Both Energy Mode controls in System Settings show High Power. The agent made no power-setting change. `system_profiler` mode fields still contradict the UI; retain this discrepancy rather than inferring earlier effective Low Power or a causal explanation for the slow trial.

Run `e2e-20261026T201255Z` repeats the entire preregistered five-shape/four-arm matrix, with one excluded warmup and three formal trials per arm/shape, using the same frozen packages and precision. Earlier trials remain separate, including the slow and interrupted requests. Request boundaries and a separate periodic watcher verify AC, the 140W adapter, and charge not below the 14% restart level; the watcher also checks the unchanged numeric AC mode setting. Existing allocation and swap stop conditions remain unchanged. Fixed controls precede the approved pilot5 A/B/C/D and diagnostic20 A/B/D cohorts. This continuation does not waive the failed numerical gate; rerun and HumanEval summaries are pending.

## Completed fixed rerun with verified power conditions

Run `e2e-20261026T201255Z` completed all 80 requests: 60 formal and 20 excluded warmups. Every arm/shape has three formal trials. All returned target tokens and common prefixes across output lengths agree. These are FP32 diagnostic observations; the known draft numerical gate is not waived.

|Input/output|Main decode token/s|Candidate-off decode token/s|Scalar DFlash8 decode token/s|Native DFlash8 decode token/s|
|---|---:|---:|---:|---:|
|21/128|48.171|48.337|15.157|34.290|
|8192/128|52.105|52.254|9.869|18.873|
|32768/128|41.264|34.770|13.757|19.451|
|8192/4096|60.363|60.349|15.524|28.702|
|32768/4096|45.577|45.614|10.952|17.263|

Rates use total emitted decode intervals divided by total worker decode seconds. Loading and excluded warmups do not contribute. The full-request long-output comparison is:

|Input/output|Arm|Median client wall s|Worker decode token/s|Formal trials|Max request swap-out MiB|
|---|---|---:|---:|---:|---:|
|8192/4096|Main|115.518|60.363|3|0.000|
|8192/4096|Candidate off|115.504|60.349|3|0.000|
|8192/4096|Scalar DFlash8|311.647|15.524|3|0.000|
|8192/4096|Native DFlash8|190.568|28.702|3|0.000|
|32768/4096|Main|300.119|45.577|3|0.000|
|32768/4096|Candidate off|300.122|45.614|3|0.000|
|32768/4096|Scalar DFlash8|584.542|10.952|3|0.000|
|32768/4096|Native DFlash8|447.618|17.263|3|0.000|

At 8192/4096, native versus scalar DFlash improves decode by 84.88% and reduces median client wall by 38.85%. Relative to main, native decode is 52.45% lower and median client wall is 64.97% higher.

At 32768/4096, native versus scalar DFlash improves decode by 57.63% and reduces median client wall by 23.42%. Relative to main, native decode is 62.12% lower and median client wall is 49.15% higher.

Both long tiers therefore fail the positive main-versus-native E2E criterion. Candidate-off closely matches main in the completed long tiers. C/D final-quarter worker-readiness ratios all exceed 0.9; equivalent baseline worker-tail clocks are Not Obtained because legacy scalar windows time only forwards. Client-receipt tail ratios remain separate.

At 8K/4096, C and D have identical acceptance histograms in every formal trial: 1395 cycles, 9765 proposals, 2694 accepted proposals, and six AR tail steps. Each cycle emits an average of 2.931 new tokens. Native verifies 11160 target rows plus six AR rows to produce 4095 post-anchor outputs, about 2.73 times the target row evaluations of AR; batching does not remove rejected speculative work. Native replay count is zero versus 1275 scalar replay cycles. Across native formal decode time, verification accounts for 82.71%, draft 15.50%, ingest 0.95%, and commit 0.73%. This explains why native can improve scalar DFlash while still losing to AR. Scalar replayMs is included in its legacy commitMs, so these fields must not be added as disjoint components.

The 32K/128 main formal client walls are 272.804, 257.166, and 215.480 seconds; decode rates are 35.178, 45.141, and 45.202 token/s. This time-ordered variation prevents attributing apparent prefill or short-request wall improvements solely to code. AC supply, the 140W adapter, and the numeric power setting remained unchanged; no thermal warning was reported at request boundaries. The cause remains unresolved. The user reported no obvious competing graphics workload during the queried earlier period; CPU snapshots do not establish per-process GPU contention. Raw slow rounds are retained. Shape-wise arm order is counterbalanced, with consecutive trials per arm; it is not randomized per-trial interleaving.

Per-request values, minimum/maximum/mean/median times, loading/warmup swap counters, artifact hashes, speculation counters, and sampled memory are in [the fixed rerun summary](evaluations/issue98/power-rerun/fixed-controls.json). [Prefix verification](evaluations/issue98/power-rerun/target-prefix-check.json) covers all 80 completed requests. The full restricted raw inventory will be refreshed after HumanEval stops. Earlier observations below remain separate.

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

In the fresh fixed run, no OOM, worker crash, or target-token mismatch was observed. The maximum steady formal request swap-out is 257.515625 MiB (32K/128 main trial 2), below 512 MiB; all formal 4096-output requests added zero. Loading is materially different: the 20 loading/readiness intervals added 63830.109375 MiB cumulative whole-machine swap-out, with a maximum of 5497.6875 MiB per interval. All 20 excluded benchmark warmups added zero. These loaded-state timings are not a cold-start or no-swap claim. The complete per-load counters are published separately.

The maximum traced C/D allocation sample is 37.69146728515625 GiB (32K/4096 native), below 38.5 GiB. Native 32K allocations exceed the device's recommended 37.440 GiB working set and have nonzero submission-above-working-set counters; these do not prove paging or explain the observed throughput. Earlier 35W completed formal requests had a maximum of 133.672 MiB swap-out and a native sampled peak of 37.485 GiB; the first short-gate load added 1454.484375 MiB before readiness. These intervals remain distinct.

Traced C/D formal GPU submission increments match interface submission counts. GPU milliseconds/submissions include draft and ingest calls, while their dispatch counts are not exported. The fixed-run resource stream contains 29786 worker RSS samples; RSS is not the kernel physical-footprint metric or continuous Metal-driver peak. System wired/compressor maxima include loading and occur at different times. Numeric chip temperature, physical-footprint peak, continuous driver residency/peak, and memory-pressure color are Not Obtained. Active draft finite checks cover selected small final outputs, not every intermediate tensor throughout every request.

## Transport, application evaluation and retained interruptions

A separately committed fake-worker regression reproduces an uncaught malformed-tool-call exception. The compatibility layer now returns an explicit model-output error; malformed calls remain failures. This does not prove that all historical truncated streams had that cause. Terminal candidate worker clocks retain generated-token readiness on failure/cancellation; they do not invent client receipt or successful application completion.

Pilot5 and diagnostic20 were not run after the earlier 35W power stop. The fresh HumanEval run `human-20261027T003048Z` has now started, with pilot order A/B/C/D followed by diagnostic20 order D/B/A. It uses the pinned standard harness at `349400b92b9de5efb88d73ce8cf5931f94f9e7b7`, Pi 1.4.0, seed 0, context 32768, output cap 4096, and task timeout 300 seconds. Actual Python positive and HumanEval/156 assertion-negative preflight controls passed (exit 0 / exit 1). Complete cohort worker metrics, Pi totals, official quality, TTFT, and reconciliation remain pending; no incomplete cohort speed or trend point is invented.

Retained controller failures include an out-of-range 3600-second deadline causing HTTP 500 before the first excluded warmup, a superseded run interrupted to move client t0 after input preparation, and a preflight-only TCP TIME_WAIT bind-probe failure before model loading/service pause. The corrected control deadline is 1800 seconds; planned Pi timeout remains 300 seconds. Finally, the power stop interrupted one main 8K/4096 formal request; no complete worker-result timing is inferred for it. See the [attempt ledger](evaluations/issue98/attempt-ledger.json).

## Disposition and evidence

Keep code PR #99 unadopted and the experiment Issue open for manual review/resumption. Results PR #100 preserves the complete fixed rerun and all failed/interrupted records independently of candidate code; HumanEval is still running. No merge, production-default change, numerical relaxation, memory-limit change, unrelated process/container termination or Windows work was performed.

Raw logs, full states, prompts and output text remain restricted local artifacts; the public [evidence inventory](evaluations/issue98/evidence.json) provides relative names, sizes and hashes, not download links. The original service files were restored unchanged after fixed controls, with the pre-existing readiness failure still recorded. Verified power conditions allowed the complete fresh fixed matrix and the current HumanEval continuation; the earlier power uncertainty remains attached to the earlier run. Known numerical failure and negative fixed E2E results prevent adoption. The inventory will be refreshed after all active evaluation processes stop.
