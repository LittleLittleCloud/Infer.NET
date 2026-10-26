# Experiment 98: real DFlash2 with native eight-row target verification

Status: diagnostic evaluation in progress. The unchanged draft numerical gate failed; code adoption is not recommended. Held-out E2E controls and fixed HumanEval cohorts remain pending. No missing metric is represented as zero.

Approved [Issue #98](https://github.com/LittleLittleCloud/Infer.NET/issues/98), implementation [PR #99](https://github.com/LittleLittleCloud/Infer.NET/pull/99). Baseline main is `4f51f241e87999c8cc3f79d733cd95ade9c711cd`; frozen E2E candidate is `9c073ff81082fe79b1cd7a2d9471d8c2b8f3deaa`. Results are independent of the candidate branch and do not update the adopted baseline.

## Method and configuration

One M5 Pro 20-core GPU / 48 GiB, AC / High Power, full-local Metal, concurrency one, no prefix cache, `iogpu.wired_limit_mb=0`. Target `mlx-community/Qwen3.6-35B-A3B-8bit@e06a74e6236a60c8367e1a3214e83d8b61b637b0`, affine8/group64, FP32 operands/KV/recurrence. Draft `incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b6923ad6c14cb1bb41c37a9041446496ab6`, BF16 checkpoint matrices and FP32 arithmetic/context. No Windows/CUDA run or setting/precision change.

A is fresh scalar main; B is the same candidate with DFlash off; C uses real DFlash block8 with actual scalar verification and snapshot/replay; D uses the same proposals/width with native block verification and direct prefix commit. All arms use explicit chunk32/scratch128, matrix0/sequential DeltaNet, FP32 GEMM/KV, fused-FP32 decode, and local threshold262144. Component verification time is not emitted-token throughput. The common two-hop loopback observer keeps native token receipt TTFT separate from compatible text/tool TTFT and worker clocks.

Held-out controls are 21/128, 8192/128, 32768/128, 8192/4096 and 32768/4096. Before observing timing, arm order was fixed by shape as ABCD, DCBA, BADC, CDAB, ABCD. Each arm/shape has one excluded warmup followed by three retained formal trials; this balances shape-level load order, not per-trial randomization. All exact output tokens are compared across arms. Pi 1.4.0 seed0 pilot5 is planned in A/B/C/D, then diagnostic20 in A/B/D; context32768, output4096, timeout300s and the fixed dataset/hash remain unchanged.

## Numerical and state screening

See [gates.json](evaluations/issue98/gates.json). Native Metal compilation, independent target-state checks and all 19 short full-cycle cases passed. Growth, ring eviction, suffix poisoning, cancellation/resumption, natural EOS and exact output/final target-state comparisons at positions 2047, 4093, 8192 and 32768 passed. Independent shared embedding/head, fresh context and ring tests passed.

The independent draft oracle failed at `HOut layer 1`: relative L2 `8.291167981301167e-7`, maximum absolute error `0.0107421875`. The required criteria remain relative L2 <1e-4 AND maxabs <0.01. Later checkpoints after the first failing assertion are not claimed passed. Issue #98 explicitly permits diagnostic E2E continuation for this known draft-only failure; qualification remains failed.

The first short-gate component added 1454.484375 MiB whole-system swap-out before runtime readiness, with no additional swap-out during its post-load checks. This is a model-load observation, not a formal request result. Continuous driver peak coverage is not assumed from sampled process memory.

## Implementation and transport validation

Release and Native AOT builds pass, with 171 CPU tests passed and 13 opt-in model facts skipped. The frozen AOT package passes 38 controlled protocol checks. A separate fake-worker regression reproduces malformed tool output causing an uncaught exception. A distinct fix returns an explicit API error, retaining malformed calls as failures. It does not prove that all historical stream failures had that cause.

The first E2E attempt (`e2e-20261026T174029Z`) failed before any formal trial: the controller configured a 3600-second worker deadline, beyond the supported maximum 1800, causing HTTP500 on A's excluded warmup before inference. This controller error and its raw evidence are retained. The corrected stress deadline is 1800 seconds; Pi timeout remains 300 seconds. A subsequent run (`e2e-20261026T174243Z`) was deliberately interrupted to move client t0 after fixture reading and JSON construction, before HTTP dispatch. Its completed short-input trials and interrupted 8K request remain superseded diagnostic evidence. A preflight-only port probe then encountered TCP TIME_WAIT with no listeners; SO_REUSEADDR corrected the probe before any service pause/model load. The complete corrected run is `e2e-20261026T174911Z`. Product code was not changed after freezing.

## Completed short-output controls (interim)

The corrected run `e2e-20261026T174911Z` has completed all four arms for all three 128-output shapes. Values below use all three formal trials; each preceding warmup is excluded. Worker throughput is sum(N−1)/sum(first-to-last readiness seconds). Client wall is the median of complete request intervals with input reading and JSON construction excluded. These diagnostic measurements retain the failed numerical gate; they are not HumanEval results.

|Input/output|Arm|Worker prefill token/s|Worker decode token/s|Median client wall s|Maximum new swap-out MiB|
|---|---|---:|---:|---:|---:|
|21/128|Main|171.628|66.315|2.038|0.000|
|21/128|Candidate off|172.931|65.987|2.049|0.000|
|21/128|DFlash8 scalar/replay|165.011|20.762|6.248|0.000|
|21/128|DFlash8 native/commit|157.659|46.052|2.899|0.000|
|8192/128|Main|170.698|60.191|50.107|0.000|
|8192/128|Candidate off|170.748|60.234|50.098|0.000|
|8192/128|DFlash8 scalar/replay|166.355|13.783|59.027|0.000|
|8192/128|DFlash8 native/commit|163.649|25.540|55.042|0.000|
|32768/128|Main|155.758|45.922|213.159|2.812|
|32768/128|Candidate off|155.750|45.941|213.169|0.000|
|32768/128|DFlash8 scalar/replay|155.622|13.740|219.824|133.672|
|32768/128|DFlash8 native/commit|155.581|22.008|216.415|0.000|

All observed output token hashes match across arms/trials. Every short-output formal request stays within the 512 MiB whole-system new swap-out gate; nonzero 32K observations are retained in the table. Native DFlash improves decode over scalar DFlash, while all three short-output fixtures remain slower than main. Sustained 4096-token generation and Pi application results are still pending; no conclusion about those unmeasured loads follows from this table.

For 21/128, native decode is 121.81% faster than scalar DFlash but 30.56% slower than main. For 8192/128, it is 85.30% faster than scalar DFlash and 57.57% slower than main; median complete request wall increases by 9.85%. Native replay count is zero. The 8K fixture accepts 75/329 proposals across 47 cycles, or about 2.60 emitted outputs per cycle before scalar tails; target verification still takes about 4.0 seconds.

The 32K native allocation sampling peak is 37.485 GiB, below the 38.5 GiB gate but slightly above Apple's recommended 37.440 GiB working set. Its submission-above-working-set counter is nonzero, with zero observed new swap-out in these native formal requests. These counters do not measure continuous physical driver residency or establish the cause of the throughput gap. They are disclosed separately from endpoint allocation and the unchanged system setting. Continuous driver peak remains Not Obtained.

GPU submission counts reconcile with the allocator's per-request submission increments for every traced C/D formal request. GPU milliseconds/submissions include native draft and ingest calls; their dispatch counts are not exported, so the dispatch field is incomplete. Main A/B endpoint counts are retained without claiming equivalent allocation-trace coverage. Per-component wall times are not independently additive: replayMs is contained in commitMs for the legacy control.

See [all retained corrected-run trials](evaluations/issue98/fixed-controls.json). Main scalar profiling windows time forwards; they do not reconstruct the same worker readiness tail clock as the candidate's joint-block windows. Client receipt tail ratios and worker readiness remain separate.

## Pending measurements and recovery

Worker prefill/decode, client TTFT, request wall time, sustained final quarter, speculation counters, request resources, pilot5 and diagnostic20 are Not Obtained while evaluation is running. The original service had an address-bind failure before this experiment. Its package/plist/configuration are restored unchanged after each controller; Ready/real-request restoration is not claimed without evidence. No code merge or production-default change is authorized.
