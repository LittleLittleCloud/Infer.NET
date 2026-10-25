# Experiment 70: Metal decode shifts from linear layers to attention as context grows

2026-10-25 America/Los_Angeles. [Experiment #70](https://github.com/LittleLittleCloud/Infer.NET/issues/70), [code PR #90](https://github.com/LittleLittleCloud/Infer.NET/pull/90), [independent results PR #91](https://github.com/LittleLittleCloud/Infer.NET/pull/91).

**Diagnostic measurements are complete.** Short-context decode spends most sampled GPU-region time in MoE, dense projections and the LM head. At 32K, attention becomes the largest individual region. Arithmetic and precision match main; the candidate remains unadopted and the original deployed service is restored. This report establishes bottleneck priorities, not an optimization speedup or physical hardware utilization.

## Sampled GPU-region shares

Each shape has one excluded warm-up and three retained formal trials. The 75 formal sampled forwards cover every dispatch: 675 dispatches in 273 valid encoder regions per sample, zero invalid regions. Percentages divide each region's summed nanoseconds by the sum of all measured regions.

| Coarse region | 21-token prompt | 8192-token input | 32768-token input |
|---|---:|---:|---:|
| MoE block | 40.0% | 33.6% | 24.9% |
| Dense input/output projections | 37.9% | 31.8% | 23.9% |
| LM head | 13.6% | 11.3% | 8.4% |
| Attention preparation/core | 3.5% | 19.1% | 39.6% |
| Delta recurrence | 2.1% | 1.8% | 1.3% |
| Delta preparation/output gate | 1.4% | 1.2% | 0.9% |
| Other normalization/embedding/argmax | 1.5% | 1.3% | 1.0% |

MoE includes routing, shared/routed experts, activation, merge and the next input normalization. Its internal gate/up/down operations are not separately sampled in encoder mode. Dense projections include the input/output projections of both attention and DeltaNet: the recurrence percentage is not the cost of the entire DeltaNet block. Unused fine-grained tags in native records do not imply zero operation cost.

Measured-region sums account for 91.49%, 92.60% and 94.31% of sampled complete-command GPU durations. The remainder stays unattributed; it is not a measured barrier or dispatch cost. Fixed samples cover the first eight classifier forwards after reset and subsequent absolute positions divisible by 128: 27 short-request samples and 24 each at 8K/32K. They do not cover every decode interval. The first output normally comes from prefill and is outside these native decode samples.

The next optimization candidates are quantized linear projections/MoE/LM head and long-context attention. Splash's matrix-tile, scale handling and fusion ideas and Ignis's quantized attention design merit controlled experiments in those regions. These measurements do not determine whether each region is ALU-, matrix- or bandwidth-limited, or prove that INT8 beats a BF16/tiled alternative. Delta recurrence-only fusion has lower measured priority; its projections remain substantial work in the dense category.

## Configuration and actual versions

M5 Pro, 20 GPU cores / 48 GiB, AC/High Power. Pinned `mlx-community/Qwen3.6-35B-A3B-8bit@e06a74e6236a60c8367e1a3214e83d8b61b637b0`, affine8/group64, FP32 KV and recurrence, existing arithmetic, explicit fused-fp32 decode, native scratch128, managed prefill chunk32, concurrency one, fresh request state, no prefix cache or speculation. All measured inference is local Metal; the original Windows agent remains resident but performs no measured work. Fixed requests use 128 outputs with EOS ignored and a 900-second deadline; the application pilot handles EOS normally.

Main baseline is `4f51f241e87999c8cc3f79d733cd95ade9c711cd`; its difference from the original approved #70 baseline `4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3` has no runtime/native changes. Final measured encoder-mode code is `43c81faff69a5e503a0f04be9ca664c73d89cce8`. Runtime health, executable/native hashes and the actually mapped native library were checked separately from source HEAD. [Runtime manifests](evaluations/issue70/runtime-manifests.json).

The user resumed previously approved diagnostic #70 with “开始测各阶段占比”. Its earlier research/closeout remain historical records. The resumed scope and hardware correction were recorded in the Issue; no optimization or merge was authorized.

## Hardware sampling correction and controls

M5 reports dispatch-boundary counter sampling unsupported and encoder/stage-boundary sampling supported; timestamp-buffer creation succeeds. Phase A used main and `bd3daa6` default-off controls, each with one warm-up and three trials per shape. Its attempted dispatch-counter arm completed short/8K as host-only measurements with no valid stage coverage; 32K warm-up was interrupted after diagnosing the capability limitation. The original service was restored. The guarded pilot correctly refused to start after that interruption. No stage shares are inferred from this phase.

Earlier `de7c332` built the initial instrumentation but ran no full-model candidate cohort. `bd3daa6` clears diagnostic state on reset/prefill. The final `43c81fa` adds automatic encoder-boundary fallback and an independent output/timestamp/coverage hardware check. Phase B ran encoder-on then default-off, each with one warm-up and three formal trials per shape. All output IDs match across both modes and the earlier main controls.

Encoder splitting changes diagnostic scheduling. Per-region sampling uses the supported encoder boundary, while dispatch-capable devices retain barrier-based dispatch samples. CPU encode, commit and wait are recorded separately; wait includes GPU execution and must not be added to GPU duration. Native readiness stops before diagnostic log emission and managed/client output. These native records do not measure user-visible TTFT.

| Input/output | Off prefill median, s | On prefill median, s | Off decode, token/s | On decode, token/s | On decode change |
|---|---:|---:|---:|---:|---:|
| 21/128 | 0.084606 | 0.084584 | 70.180 | 69.765 | −0.590% |
| 8192/128 | 29.993710 | 30.000072 | 61.005 | 60.559 | −0.731% |
| 32768/128 | 139.756462 | 139.887223 | 46.005 | 45.462 | −1.181% |

Decode aggregates sum 127 intervals per request and divide by summed first-to-last output duration. Prefill includes producing the first output. The whole-request perturbation is diluted because most forwards are unsampled; it does not establish that sampled region shares equal uninstrumented shares exactly. Raw values, means, medians, slowest values, output hashes and coverage remain in [fixed requests](evaluations/issue70/fixed-requests.json) and [encoder summary](evaluations/issue70/encoder-fixed-summary.json).

For sampled forwards, median CPU encode is 0.804/0.837/0.836 ms at short/8K/32K; commit is 0.00383/0.00406/0.00423 ms. Complete-command GPU means are 14.502/17.114/22.621 ms. These diagnostic scopes cannot be substituted for clean throughput or used to bound possible CPU optimization gains.

Phase A is retained separately. A final CPU compilation overlapped part of main 32K trial3; the controller was briefly held between arms until packaging completed. That trial is retained and excluded from any clean formal adoption claim. No competing GPU benchmarks ran together. [Earlier phase summary](evaluations/issue70/phase-a-fixed-summary.json).

## Fixed HumanEval5 application diagnostic

Pi1.4.0, seed0, fixed IDs156/94/72/80/149, fresh session/workspace per task, context limit32768, output limit4096, timeout300 seconds. Official dataset revision `b9453e8f1865df0a0057a1383767caa400acfbcf`, SHA256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`. Counter-on local Metal with actual/observer coordinators and worker all on43c81fa; Windows absent from execution. The separately versioned evaluation harness and its exact file hash are recorded in the trend point/evidence.

- Pi completion and official Docker assertions: **5/5**, no timeouts/errors; failing negative control retained. Docker image `python@sha256:5fb4bc53dd0180955e31e5245b648bbb24a4ac4ab03a1de81513642325220bcd`.
- Complete worker timing/token reconciliation and valid-output TTFT coverage: **19/19 requests**.
- Input42203, output4623, decode intervals4604; summed prefill155.233991 seconds and decode68.180209 seconds.
- Aggregate **prefill271.867 token/s**, **decode67.527 token/s**; Pi task wall sum225.113466 seconds; median observer TTFT11818.036 ms.
- 186 application forwards have valid full dispatch coverage. Their measured-region shares include MoE37.44%, dense35.52%, LM head12.64%, attention preparation/core9.68%, Delta recurrence2.00%. Position-selected samples are not a time-weighted whole-cohort cost profile.

Worker durations come from actual streaming `ServiceEvent.result.timings`, explicitly normalized for the existing summarizer; no native `INFERENCE_METRIC` emission is claimed. TTFT starts immediately before the observer forwards to the experiment coordinator and ends at first valid text/tool-call output; role/usage/keepalive events are ignored. Pi-to-observer dispatch is excluded, two loopback observation hops are disclosed, and buffered tool calls may include much of decode before valid output arrives. The observer template's “main coordinator” wording is corrected in the public description: the measured coordinators are the candidate build.

This is a separate profiled configuration with no clean HumanEval main comparison. [Pilot evidence](evaluations/issue70/pilot5.json), [application stage summary](evaluations/issue70/application-stage-summary.json), [diagnostic trend point](https://littlelittlecloud.github.io/Infer.NET/#experiment=decode-profile-20261025&run=issue70-encoder-pilot5) (available after results merge and Pages publication). Fixed20 remains required before profiler adoption or a formal main performance comparison; it was not run here.

## Resources and 128K limit

The maximum completed fixed-request Metal allocation is35.880402 GiB; maximum formal per-request whole-machine new swap-out is305.5 MiB. Each retained fixed formal request passes those two recorded bounds. These are not continuous transient allocation or physical residency peaks. Process-footprint snapshots, one-second RSS/VM observations, thermal warning queries and memory-pressure records are retained; RSS does not replace footprint. No thermal warning was recorded by the sampled query.

The application pilot's maximum completed Metal allocation is34.647644 GiB. Whole-machine new swap-out is **3983.6875 MiB during loading/warmup**, then **0 bytes over the pilot cohort**. Loading swap is disclosed separately and cannot support a “no swap” claim. Swap is not attributed to one process. Continuous transient allocation and per-submission physical residency safety were not obtained, so full resource acceptance remains incomplete. [Resource coverage](evaluations/issue70/resources.json).

FP32 capacity preflight rejects128K under the current38.5 GiB specification: retaining observed non-KV allocation and replacing KV with135168-token FP32 capacity projects42,584,391,680 bytes (39.660 GiB), before further attention/workspace growth. This is an allocation projection, not a measured peak or proof of minimum hardware memory. No128K request was issued and no alternate precision or system limit was substituted. [Capacity preflight](evaluations/issue70/capacity-preflight.json).

## Verification, failures and restoration

131 managed checks passed. Native graph checks passed60 hardware cases plus existing profile/KV guards; legacy/fused attention passed independent double-reference checks at17/2048/8192/32768. The new two-encoder check independently validates identical outputs, timestamps and dispatch coverage. Fixed full-model outputs match; no new full-model layer/state L2 study was performed because arithmetic is unchanged. [Validation scope](evaluations/issue70/validation.json).

Retained failures include the initial native insertion/compile error, a missing-assets `--no-restore` invocation, unsupported dispatch counters, an exploratory division-by-zero analysis attempt on missing stage data, the intentional 32K interruption, the guarded pilot refusal, and initial readiness503 attempts. Corrected runs do not erase them. Some console stderr was not saved as a complete local file; this limitation is explicit. [Failures](evaluations/issue70/failures.json).

The final audit verifies original `cd6f7da401fadf044dac13e87b415d27bb5eadfb` deployment, unchanged executable/native/config/plist hashes, running Mac LaunchAgent, unchanged coordinator process, exactly one model process, closed experimental ports and a real deterministic CUDA-to-Metal split request. The Mac model PID stays stable across the final probe. Windows' original foreground agent was never stopped/rebuilt/reconfigured by this experiment. `iogpu.wired_limit_mb=0`, AC power retained. [Restoration and final audit](evaluations/issue70/restoration.json).

Raw evidence remains local-only in the candidate worktree's ignored `artifacts/dual-host/issue70-decode-profile-20261025/` and `issue70-encoder-profile-20261025/`. Public summaries omit prompts, solutions, credentials and private endpoints; [restricted evidence hashes](evaluations/issue70/evidence.json) identify the retained files without pretending to provide public downloads. Results and code require independent review. No main merge, deployed default change, optimization adoption or formal128K completion occurred.
