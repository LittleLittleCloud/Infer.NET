# Native target block verification and prefix commit

Status: Evaluation in progress; candidate unadopted. No merge, production default change, or site publication.

Experiment [#95](https://github.com/LittleLittleCloud/Infer.NET/issues/95), approved plan revision 1, code [PR #96](https://github.com/LittleLittleCloud/Infer.NET/pull/96). Baseline main: `4f51f241e87999c8cc3f79d733cd95ade9c711cd`. Transaction measurements use `cda86c0dfc16b5e98484b74e4b72d5345132dd98` on M5 Pro 20-core GPU / 48 GiB, AC power, unchanged affine8/group64 target weights, FP32 KV/operands/recurrence, no draft, concurrency one and no prefix cache. Windows is not used or modified.

## Mechanism and measurement boundary

The native verifier executes 1/4/8 consecutive positions through a layer-major target graph. Dense vector-four projections reuse packed affine weights; stateless reductions have a row axis. Routed experts retain separate per-row expert IDs and FP32 dots. Each recurrent SIMD lane writes the required prefix states while performing the recurrence. Live recurrent state stays immutable until commit; commit selects a captured state and updates convolution history, without accepted-prefix replay. Tentative KV suffix rows remain invisible behind the committed position. The maximum eight-row buffers add 519127072 bytes (approximately 495 MiB). The original scalar path remains the default.

The controlled comparison executes all candidate input rows in both arms and commits the same selected prefix. The scalar reference runs the unchanged scalar graph and snapshots/restores the selected prefix if a suffix is rejected. Its device-copy overhead is included. Wall-clock transaction time, GPU time, commit wall time, dispatches and submissions are separate. These values are not autoregressive emitted-token throughput and are not divided by eight to manufacture a decode rate. No DFlash draft, acceptance-rate estimate, or comparison with Splash's historical 148.914 token/s is included in this measurement.

Numerical gates use scratch128/managed chunk32 with FP32 vector prefill. Controlled transaction measurements use scratch2048/managed chunk2048, FP32 matrix/grouped expert prefill in both arms, and fused FP32 attention. Application HumanEval uses its existing diagnostic worker profile separately. Different prefill profiles are not combined into a speedup claim.

## Retained failure and correction

The first full-model attempt at `b6123fa` failed the unchanged FP32 logits criterion: relative L2 0.0994789523509562 and maximum absolute error 2.316425323486328. One-row prefixes passed; multiple rows used distinct RoPE positions but accidentally shared the KV write position. This is an implementation bug, not quantization drift or a reason to relax numerical thresholds. The failed logs remain retained.

Commit `871d3f6` corrects every derived head position and adds an independent KV-address check with poisoned neighboring rows across the 2048 boundary. Independent FP32 recurrence, convolution, decay, beta and history checks passed for two fixtures and rows1/4/8. Full-model short-context checks passed all16 prefix cases (rows1/4/8; accepted0..rows), per-row logits/layer traces, immutable live state, all committed model states, continuation tokens, pending-operation guards, explicit growth and reset. Longer-context validation passed all 24 cases at 2047, 4093, 8192 and 32768 tokens, accepted prefixes 1/4/8, in block and scalar-transaction arms. Each row of logits and each layer trace was checked separately, along with every committed KV/Delta/convolution state, pending recurrent prefix slots and three continuation tokens. [Long validation](evaluations/issue95/long-validation.json).

Native/managed Release builds and131 managed tests passed. The offline `xcrun metal` command is unavailable on this host; actual native Metal runtime shader compilation and numerical dispatches passed. Missing compiler support is not described as a shader failure or an offline compile pass.

## Controlled results

The complete held-out controls finished: 144 trials, including 36 excluded warmups and 108 formal trials across 18 comparisons. Every arm/shape has one warmup and three formal trials, with alternating order. The table shows median milliseconds per transaction; reductions use aggregate wall time across all three formal trials. The scalar control shares the candidate binary and allocated workspace, executes every input row through the unchanged scalar graph, and includes selected-prefix snapshot/restore costs. It is not a separately measured main executable.

| Context | Input rows | Committed rows | Scalar ms/transaction | Block ms/transaction | Aggregate reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 21 | 1 | 1 | 14.200 | 17.376 | -22.29% |
| 21 | 4 | 1 | 56.936 | 34.583 | +39.28% |
| 21 | 4 | 4 | 56.891 | 34.906 | +38.59% |
| 21 | 8 | 1 | 111.678 | 60.393 | +45.94% |
| 21 | 8 | 4 | 113.362 | 60.822 | +46.34% |
| 21 | 8 | 8 | 113.736 | 62.009 | +45.43% |
| 8192 | 1 | 1 | 16.354 | 19.621 | -19.99% |
| 8192 | 4 | 1 | 67.063 | 44.997 | +32.89% |
| 8192 | 4 | 4 | 65.420 | 44.018 | +32.69% |
| 8192 | 8 | 1 | 132.553 | 81.403 | +38.58% |
| 8192 | 8 | 4 | 132.834 | 80.686 | +39.23% |
| 8192 | 8 | 8 | 130.772 | 80.768 | +38.22% |
| 32768 | 1 | 1 | 21.683 | 25.034 | -15.74% |
| 32768 | 4 | 1 | 88.555 | 66.586 | +24.80% |
| 32768 | 4 | 4 | 86.721 | 65.936 | +23.57% |
| 32768 | 8 | 1 | 175.528 | 125.220 | +28.62% |
| 32768 | 8 | 4 | 175.879 | 124.343 | +29.02% |
| 32768 | 8 | 8 | 173.466 | 124.653 | +28.12% |

All six predeclared eight-row 8K/32K comparisons exceed the 5% transaction-cost reduction criterion. One-row verification is 15.7–22.3% slower than the original scalar transaction, so the scalar path remains the default. This does not measure draft cost or establish speculative-generation throughput. [All trials and GPU/commit counters](evaluations/issue95/transactions.json).

Maximum sampled Metal allocation was 36.855743 GiB; every formal request had zero new whole-system swap-out. One excluded warmup incurred 406.625 MiB; the complete benchmark component, including loading/warmups, recorded 406.625 MiB. The corrected short numerical component separately recorded 159.125 MiB across loading and checks. These are whole-machine counters, not worker attribution. Continuous driver/residency peaks and numeric temperatures were not obtained. AC/High Power were recorded; the late thermal gauge reported no warnings, and resource-monitor gaps stayed below 0.667 seconds during the benchmark.


The fixed HumanEval pilot5 completed with official assertions 5/5 in both arms and an actually failing negative control. Worker metrics are complete: main16/16 and candidate15/15 requests. The target block path is disabled in both application arms. Explicit native settings are FP32 KV/GEMM, managed chunk32, matrix0, sequential DeltaNet, scratch128 and fused-FP32 decode attention. This profile is separate from the matrix/chunk2048 transaction controls and must not be compared directly with historical native-prefill rates.

| Pilot5 arm | Prefill token/s | Decode token/s | TTFT median ms | Official | Worker clocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| main | 171.589 | 67.800 | 15503.938 | 5/5 | 16/16 |
| candidate | 171.705 | 67.826 | 15251.636 | 5/5 | 15/15 |

Both pilot arms have complete valid-output TTFT coverage and zero bracketed new whole-system swap-out per request. Task trajectories differ, so the lower candidate Pi time is not attributed to the verifier. These are unchanged-default integration controls, not application acceleration from target batching. Complete cohort20 is pending. [Main pilot5](evaluations/issue95/main-pilot5.json), [candidate pilot5](evaluations/issue95/candidate-pilot5.json).

The first long numerical component recorded 3645.6875 MiB of whole-system new swap-out while reading complete CPU state arrays and writing reference fixtures. It is not a single steady-state product request or a worker-attributed count. This result is retained; a repeat with a predeclared 1 GiB diagnostic managed-heap limit is in progress. The heap limit changes neither model precision nor the native verifier, and it is not applied to application throughput runs.

## Existing service failure and restoration limits

Before any candidate process started, the original production worker was already non-Ready and repeatedly failed with SocketException49, “Can't assign requested address.” Exclusive GPU controllers verified its package identity, paused only the identified Mac LaunchAgent, and re-enabled the original LaunchAgent with unchanged executable/native/plist/configuration hashes. Ready and real-request restoration are Not Obtained while that pre-existing binding failure remains. The experiment does not change the LAN address, Windows service, authentication or system settings; wired_limit_mb remains0. This condition is distinct from candidate correctness and cannot be reported as successful Ready restoration.

## Evidence and decision

Raw logs, model states and private configurations remain in local ignored artifacts under `artifacts/dual-host/issue95-target-block-commit/`, with public sanitized summaries and hashes to follow. Local paths are not public downloads. Formal continuous driver peaks,128K acceptance and complete application results are not yet obtained.

Decision pending complete evidence and manual review. Successful target transaction timing alone will not adopt speculative generation or establish a product speed baseline.
