# Native target block verification and prefix commit

Status: Evaluation in progress; candidate unadopted. No merge, production default change, or site publication.

Experiment [#95](https://github.com/LittleLittleCloud/Infer.NET/issues/95), approved plan revision 1, code [PR #96](https://github.com/LittleLittleCloud/Infer.NET/pull/96). Baseline main: `4f51f241e87999c8cc3f79d733cd95ade9c711cd`. Transaction measurements use `cda86c0dfc16b5e98484b74e4b72d5345132dd98` on M5 Pro 20-core GPU / 48 GiB, AC power, unchanged affine8/group64 target weights, FP32 KV/operands/recurrence, no draft, concurrency one and no prefix cache. Windows is not used or modified.

## Mechanism and measurement boundary

The native verifier executes 1/4/8 consecutive positions through a layer-major target graph. Dense vector-four projections reuse packed affine weights; stateless reductions have a row axis. Routed experts retain separate per-row expert IDs and FP32 dots. Each recurrent SIMD lane writes the required prefix states while performing the recurrence. Live recurrent state stays immutable until commit; commit selects a captured state and updates convolution history, without accepted-prefix replay. Tentative KV suffix rows remain invisible behind the committed position. The maximum eight-row buffers add 519127072 bytes (approximately 495 MiB). The original scalar path remains the default.

The controlled comparison executes all candidate input rows in both arms and commits the same selected prefix. The scalar reference runs the unchanged scalar graph and snapshots/restores the selected prefix if a suffix is rejected. Its device-copy overhead is included. Wall-clock transaction time, GPU time, commit wall time, dispatches and submissions are separate. These values are not autoregressive emitted-token throughput and are not divided by eight to manufacture a decode rate. No DFlash draft, acceptance-rate estimate, or comparison with Splash's historical 148.914 token/s is included in this measurement.

Numerical gates use scratch128/managed chunk32 with FP32 vector prefill. Controlled transaction measurements use scratch2048/managed chunk2048, FP32 matrix/grouped expert prefill in both arms, and fused FP32 attention. Application HumanEval uses its existing diagnostic worker profile separately. Different prefill profiles are not combined into a speedup claim.

## Retained failure and correction

The first full-model attempt at `b6123fa` failed the unchanged FP32 logits criterion: relative L2 0.0994789523509562 and maximum absolute error 2.316425323486328. One-row prefixes passed; multiple rows used distinct RoPE positions but accidentally shared the KV write position. This is an implementation bug, not quantization drift or a reason to relax numerical thresholds. The failed logs remain retained.

Commit `871d3f6` corrects every derived head position and adds an independent KV-address check with poisoned neighboring rows across the 2048 boundary. Independent FP32 recurrence, convolution, decay, beta and history checks passed for two fixtures and rows1/4/8. Full-model short-context checks passed all16 prefix cases (rows1/4/8; accepted0..rows), per-row logits/layer traces, immutable live state, all committed model states, continuation tokens, pending-operation guards, explicit growth and reset. Longer-context validation is pending.

Native/managed Release builds and131 managed tests passed. The offline `xcrun metal` command is unavailable on this host; actual native Metal runtime shader compilation and numerical dispatches passed. Missing compiler support is not described as a shader failure or an offline compile pass.

## Controlled results

The complete held-out21/8192/32768-token, rows1/4/8, accepted1/4/8 transaction controls are in progress. Each arm/shape has one excluded warmup and three formal trials; order is counterbalanced. Numerical and resource failures and valid slow trials will be retained. Final aggregate values, complete coverage and resource status are pending; no partial short-context gain is substituted for the8K/32K adoption criterion.

HumanEval pilot5 and conditional cohort20: Not Obtained. No application throughput, TTFT or official task-pass point is plotted for this target-only diagnostic.

## Existing service failure and restoration limits

Before any candidate process started, the original production worker was already non-Ready and repeatedly failed with SocketException49, “Can't assign requested address.” Exclusive GPU controllers verified its package identity, paused only the identified Mac LaunchAgent, and re-enabled the original LaunchAgent with unchanged executable/native/plist/configuration hashes. Ready and real-request restoration are Not Obtained while that pre-existing binding failure remains. The experiment does not change the LAN address, Windows service, authentication or system settings; wired_limit_mb remains0. This condition is distinct from candidate correctness and cannot be reported as successful Ready restoration.

## Evidence and decision

Raw logs, model states and private configurations remain in local ignored artifacts under `artifacts/dual-host/issue95-target-block-commit/`, with public sanitized summaries and hashes to follow. Local paths are not public downloads. Formal continuous driver peaks,128K acceptance and complete application results are not yet obtained.

Decision pending complete evidence and manual review. Successful target transaction timing alone will not adopt speculative generation or establish a product speed baseline.
