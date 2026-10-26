# Native target block verification and prefix commit

Status: Evaluation complete; adoption gates fail. Candidate unadopted. No merge, production default change, or site publication.

Experiment [#95](https://github.com/LittleLittleCloud/Infer.NET/issues/95), approved plan revision 1, code [PR #96](https://github.com/LittleLittleCloud/Infer.NET/pull/96). Baseline main: `4f51f241e87999c8cc3f79d733cd95ade9c711cd`. Transaction measurements use `cda86c0dfc16b5e98484b74e4b72d5345132dd98` on M5 Pro 20-core GPU / 48 GiB, AC power, unchanged affine8/group64 target weights, FP32 KV/operands/recurrence, no draft, concurrency one and no prefix cache. Windows is not used or modified.

## Mechanism and measurement boundary

The native verifier executes 1/4/8 consecutive positions through a layer-major target graph. Dense vector-four projections reuse packed affine weights; stateless reductions have a row axis. Routed experts retain separate per-row expert IDs and FP32 dots. Each recurrent SIMD lane writes the required prefix states while performing the recurrence. Live recurrent state stays immutable until commit; commit selects a captured state and updates convolution history, without accepted-prefix replay. Tentative KV suffix rows remain invisible behind the committed position. The maximum eight-row buffers add 519127072 bytes (approximately 495 MiB). The original scalar path remains the default.

The controlled comparison executes all candidate input rows in both arms and commits the same selected prefix. The scalar reference runs the unchanged scalar graph and snapshots/restores the selected prefix if a suffix is rejected. Its device-copy overhead is included. Wall-clock transaction time, GPU time, commit wall time, dispatches and submissions are separate. These values are not autoregressive emitted-token throughput and are not divided by eight to manufacture a decode rate. No DFlash draft, acceptance-rate estimate, or comparison with Splash's historical 148.914 token/s is included in this measurement.

Numerical gates use scratch 128/managed chunk 32 with FP32 vector prefill. Controlled transaction measurements use scratch 2048/managed chunk 2048, FP32 matrix/grouped expert prefill in both arms, and fused FP32 attention. Application HumanEval uses its existing diagnostic worker profile separately. Different prefill profiles are not combined into a speedup claim.

## Retained failure and correction

The first full-model attempt at `b6123fa` failed the unchanged FP32 logits criterion: relative L2 0.0994789523509562 and maximum absolute error 2.316425323486328. One-row prefixes passed; multiple rows used distinct RoPE positions but accidentally shared the KV write position. This is an implementation bug, not quantization drift or a reason to relax numerical thresholds. The failed logs remain retained.

Commit `871d3f6` corrects every derived head position and adds an independent KV-address check with poisoned neighboring rows across the 2048 boundary. Independent FP32 recurrence, convolution, decay, beta and history checks passed for two fixtures and rows 1/4/8. Full-model short-context checks passed all 16 prefix cases (rows 1/4/8; accepted 0..rows), per-row logits/layer traces, immutable live state, all committed model states, continuation tokens, pending-operation guards, explicit growth and reset. Longer-context validation passed all 24 cases at 2047, 4093, 8192 and 32768 tokens, accepted prefixes 1/4/8, in block and scalar-transaction arms. Each row of logits and each layer trace was checked separately, along with every committed KV/Delta/convolution state, pending recurrent prefix slots and three continuation tokens. [Long validation](evaluations/issue95/long-validation.json).

Native/managed Release builds and 131 managed tests passed. The offline `xcrun metal` command is unavailable on this host; actual native Metal runtime shader compilation and numerical dispatches passed. Missing compiler support is not described as a shader failure or an offline compile pass.

## Controlled results

Each trial evaluates 128 candidate input rows. A partial-prefix trial commits 128×accepted/rows positions, so it is not a 128-emitted-token generation run. The complete held-out transaction controls finished: 144 trials, including 36 excluded warmups and 108 formal trials across 18 comparisons. Every arm/shape has one warmup and three formal trials, with alternating order. The table shows median milliseconds per transaction; reductions use aggregate wall time across all three formal trials. The scalar control shares the candidate binary and allocated workspace, executes every input row through the unchanged scalar graph, and includes selected-prefix snapshot/restore costs. It is not a separately measured main executable.

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

Maximum sampled Metal allocation was 36.855743 GiB; every formal request had zero new whole-system swap-out. One excluded warmup incurred 406.625 MiB; the complete benchmark component, including loading/warmups, recorded 406.625 MiB. The corrected short numerical component separately recorded 159.125 MiB across loading and checks. These are whole-machine counters, not worker attribution. Continuous driver/residency peaks and numeric temperatures were not obtained. AC/High Power were recorded; the late thermal gauge reported no warnings, and resource-monitor gaps stayed below 0.667 seconds during the benchmark. A candidate AOT package build overlapped the excluded initial 8K warmup only; all formal 8K/32K trials began after that build completed. No competing GPU benchmark ran during formal measurement.


The fixed HumanEval pilot 5 completed with official assertions 5/5 in both arms and an actually failing negative control. Worker metrics are complete: main 16/16 and candidate 15/15 requests. The target block path is disabled in both application arms. Explicit native settings are FP32 KV/GEMM, managed chunk 32, matrix 0, sequential DeltaNet, scratch 128 and fused-FP32 decode attention. This profile is separate from the matrix/chunk 2048 transaction controls and must not be compared directly with historical native-prefill rates.

| Pilot5 arm | Prefill token/s | Decode token/s | TTFT median ms | Official | Worker clocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| main | 171.589 | 67.800 | 15503.938 | 5/5 | 16/16 |
| candidate | 171.705 | 67.826 | 15251.636 | 5/5 | 15/15 |

Both pilot arms have complete valid-output TTFT coverage and zero bracketed new whole-system swap-out per request. Task trajectories differ, so the lower candidate Pi time is not attributed to the verifier. These are unchanged-default integration controls, not application acceleration from target batching. [Main pilot5](evaluations/issue95/main-pilot5.json), [candidate pilot5](evaluations/issue95/candidate-pilot5.json).

## Formal application control

Fresh fixed cohort20 completed under strict admission in run `cohort-20261026T092918Z`. Both arms use the same local FP32 vector-prefill application profile and keep the verifier disabled.

| Arm | Prefill token/s | Decode token/s | TTFT median ms | Official | Pi complete | Worker clocks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| main | 144.384 | 60.753 | 18727.987 | 20/20 | 20/20 | 92/92; reconciled=True |
| candidate | Not Obtained | Not Obtained | 20526.432 | 19/20 | 18/20 | 89/90; reconciled=False |

Main has complete Pi/worker token reconciliation. Candidate retains the incomplete HumanEval/149 run: its worker terminal result reports EOS and 331 generated tokens, but no valid output token reaches the observer; Pi reports `Stream ended without finish_reason` and records zero usage. One additional request interrupted by the HumanEval/47 timeout has no terminal worker clock; that task had already written a solution that passes the official assertions, but its Pi run remains a timeout. The token mismatch and missing clock make whole-cohort candidate prefill/decode **Not Obtained**. The failed request is retained rather than discarded. This is a new application error versus fresh main and fails the predeclared no-new-error condition; no qualified <=5% default-path regression conclusion is available. Because the verifier is disabled in this control, this outcome does not prove verifier causation.

Both initial official assertion invocations executed their checks but failed to save JSON to a nonexistent pilot5 output directory. Their logs/nonzero exits remain retained. After all GPU timing ended, the same unchanged solution files were verified again with the corrected absolute cohort20 output path. Solution hashes before/after match; negative controls actually fail. No answer is regenerated or selectively retried.

Main: observed worker input/output totals 205116/18838; Pi task wall sum 1736.615 s; errors/timeouts 0/0; valid-output TTFT 92/92; externally bracketed per-request new system swap-out maximum 21.59375 MiB (92/92 requests). Maximum sampled Metal allocation 34.643204 GiB. Sampled RSS is disclosed in the JSON; process-footprint/continuous-driver peaks are Not Obtained.

Candidate: observed worker input/output totals 205519/19043; Pi task wall sum 2004.407 s; errors/timeouts 1/1; valid-output TTFT 88/90; externally bracketed per-request new system swap-out maximum 0.00000 MiB (90/90 requests). Maximum sampled Metal allocation 34.725388 GiB. Sampled RSS is disclosed in the JSON; process-footprint/continuous-driver peaks are Not Obtained.

Retained candidate execution failures: HumanEval/149, HumanEval/47. No timed-out or incomplete task is removed from the 20-task denominator.

Task trajectories differ, so Pi wall-time differences are not attributed to target batching. [Main cohort20](evaluations/issue95/main-cohort20.json), [candidate cohort20](evaluations/issue95/candidate-cohort20.json).


The first long numerical component recorded 3645.6875 MiB of whole-system new swap-out while reading complete CPU state arrays and writing reference fixtures. It is not a single steady-state product request or a worker-attributed count. A repeat with a configured 1 GiB diagnostic managed-heap limit passed all numerical checks, but its short component recorded 65977.21875 MiB of new system swap-out; the long component recorded 493.25 MiB. The failed startup resource outcome is retained separately from correctness. [Bounded diagnostics](evaluations/issue95/bounded-validation.json).

The large increment occurred before the short tool wrote its initial runtime manifest, during model loading. Before that load the machine had only 0.58 GiB free, 14.53 GiB wired and 17.18 GiB in the compressor. Numerical execution after runtime creation took 20.49 seconds, compared with 20.81 seconds in the earlier clean short run. The previous controller's process-name check did not establish resource quiescence. No specific residual process or driver mechanism is proven by this evidence.

The controller now waits for the verified original PID to end and requires three model-free samples with at least 35 GiB free+inactive+speculative, at most 6 GiB wired and at most 4 GiB compressor before loading a model. These are stricter startup-admission conditions, not relaxed result/precision thresholds. The same native binary, inputs and diagnostic heap setting then completed the short gate in 28.750 seconds, with 372.5625 MiB of whole-component swap-out including loading and all numerical cases passing. Its last admission sample had 42,421,141,504 bytes available and no model processes. [Admission retest](evaluations/issue95/startup-admission.json).

A first cohort20 control was manually interrupted after two completed tasks/eight complete worker clocks while this startup issue was reviewed. No application request resource failure was established in that partial run. The two available solutions pass official assertions; 18 tasks were not attempted. Whole-cohort throughput is Not Obtained. An initial manual assertion command used a relative Docker bind path and failed with exit 125 before Python execution; that infrastructure attempt is retained separately and corrected absolute-path validation is reported. The complete new main/candidate cohort reported above ran under the stricter admission conditions; individual answers were not selectively retried or substituted. [Interrupted control](evaluations/issue95/interrupted-cohort.json).

## Existing service failure and restoration limits

Before any candidate process started, the original production worker was already non-Ready and repeatedly failed with SocketException 49, “Can't assign requested address.” Exclusive GPU controllers verified its package identity, paused only the identified Mac LaunchAgent, and re-enabled the original LaunchAgent with unchanged executable/native/plist/configuration hashes. Ready and real-request restoration are Not Obtained while that pre-existing binding failure remains. The experiment does not change the LAN address, Windows service, authentication or system settings; wired_limit_mb remains 0. This condition is distinct from candidate correctness and cannot be reported as successful Ready restoration.

## Reproduction and source identity

Use the candidate's `docs/TARGET-BLOCK-VERIFY.md` and `tools/TargetBlockBench` for the independent operator checks and `gate`, `long-gate`, and `bench` modes. `gate` uses scratch 128 and explicit FP32 vector prefill. `long-gate`/`bench` use scratch 2048, native matrix 1, grouped experts, chunked DeltaNet and fused-all FP32 prefill attention. The measured benchmark source is `cda86c0dfc16b5e98484b74e4b72d5345132dd98`; later commits update the diagnostic harness and documentation without changing `native/` or `src/`. Runtime manifests distinguish the native library hash from the diagnostic assembly hash.

The Pi harness and observer are pinned by source commit and file SHA256 in each public application summary. Run `scripts/benchmark-pi-humaneval.py` with `--cohort pilot` or `--cohort cohort`, `--seed 0 --timeout 300 --context-window 32768 --max-tokens 4096`, the pinned dataset and a fresh output directory/session/workspace. Use separate immutable main/candidate packages, explicit native matrix 0/sequential DeltaNet settings and identical loopback observers. Authentication remains enabled; credential paths are supplied locally. Official checks use `scripts/eval-pi-humaneval-docker.py` with absolute results/output paths and the pinned offline Docker image. Worker timings come from actual terminal `ServiceEvent.result.timings`; normalization does not turn them into natively emitted log records.

Model/configuration/tokenizer hashes, artifact/runtime identities, restoration limitations, and restricted raw-evidence hashes are recorded separately. Large state fixtures and application transcripts remain local. No raw prompt, generated solution, weight file or credential is included in the results PR.

## Evidence and decision

Raw logs, model states and private configurations remain in local ignored artifacts under `artifacts/dual-host/issue95-target-block-commit/`, with [public sanitized evidence hashes](evaluations/issue95/evidence.json), [model/tokenizer hashes](evaluations/issue95/model-provenance.json), and [runtime/package/restoration identities](evaluations/issue95/runtime-and-restoration.json). Local paths are not public downloads. Continuous driver/process-footprint peaks, a standalone JIT shader binary hash, qualified whole-cohort candidate throughput, and 128K acceptance are Not Obtained. All eight checkpoint shards were stream-rehashed after GPU timing and match their pinned download hashes; configuration/tokenizer hashes remain unchanged.

Recommendation: retain the results and **do not recommend merging the candidate code in this experiment**. All six predeclared eight-row 8K/32K transaction comparisons exceed the 5% benefit threshold, and corrected numerical checks pass, but candidate cohort20 has new incomplete/timed-out application runs and incomplete token reconciliation. The no-new-error and qualified default-path regression conditions are therefore not satisfied. The pre-existing service bind failure also prevents a Ready/real-request restoration claim. One-row verifier regressions and all diagnostic startup swap failures remain disclosed.

Code PR #96 remains draft/unadopted; independent results PR #97 is submitted for manual review. The baseline stays unchanged. Any subsequent investigation or integrated draft/verify/commit experiment must preserve this result; these target transaction gains do not establish emitted-token acceleration or 128K acceptance.

Local results validation: site generation passed (3 pages, 56 notes, 26 trend runs), 26 Python checks passed, and 27 JavaScript selection/render checks passed. The first JavaScript attempt exceeded Node's default 1 MiB child-output buffer after adding complete request records. Raising only that test subprocess limit to 16 MiB fixed the infrastructure failure; inference sources and benchmark outputs are unchanged. These are offline DOM/data checks, not new browser-layout measurements.

Docker Desktop was opened for offline assertions. At cleanup an unrelated container was running, so Docker was left active and that container was not stopped. Temporary HumanEval containers were removed by their exact generated names.
