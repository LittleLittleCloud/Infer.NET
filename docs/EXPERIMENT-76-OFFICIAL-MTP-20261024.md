# Official Qwen3.6 MTP integration and evaluation

> Session closeout, 2026-10-25: the user authorized preserving research/results and closing experimental code PRs #74/#78/#81 without adoption. Earlier approval/pending/publication statements below describe the historical stage, not a current request to continue. The later #80 diagnostic20 includes the frozen MTP reference. See the [final disposition and combined summary](SPECULATIVE-DECODE-SESSION-CLOSEOUT-20261025.md). No acceptance threshold or original measurement was changed.

Experiment: [Issue #76](https://github.com/LittleLittleCloud/Infer.NET/issues/76). Implementation: [draft PR #78](https://github.com/LittleLittleCloud/Infer.NET/pull/78). Independent results: [PR #79](https://github.com/LittleLittleCloud/Infer.NET/pull/79). Research: [PR #77](https://github.com/LittleLittleCloud/Infer.NET/pull/77). DFlash is outside this experiment.

Status: implementation and both five-task pilots completed. G5 has no eligible width, so the approved plan stops before formal 20 tasks. Continuing with a diagnostic 20-task comparison requires approval of that decision-rule change. No adoption, merge, deployment update, or 128K acceptance is implied.

## Hypothesis and implementation

A trained official single-layer MTP drafter may amortize the target model's weight reads through multi-row verification. The implementation shares the target embedding and vocabulary head, keeps a separate draft attention cache, verifies all proposed rows, restores and replays the committed target prefix, and reconditions draft KV with accepted target hidden rows. Draft width is bounded to one or two tokens. The initial verified anchor remains unconsumed until the next cycle; EOS, cancellation, output caps, and the final autoregressive tail preserve the target's terminal state and pending token.

GitHub Copilot CLI 1.0.103 used `claude-sonnet-5.5` for implementation and review. The controlling agent independently audited weights, corrected implementation issues, and ran the hardware gates. These are separate activities; a Copilot report alone is not verification.

## Immutable versions and model artifacts

| Component | Version |
| --- | --- |
| Main control | `4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3` |
| Frozen application candidate | `17965ce1f43e37398441c6f192c0a331efab228d` |
| Target checkpoint | `mlx-community/Qwen3.6-35B-A3B-8bit@e06a74e6236a60c8367e1a3214e83d8b61b637b0` |
| Official draft source | `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Extracted sidecar SHA256 | `b822f11011b12e72f9e7c485c301bc5990f15745100b76438d7c025874b71ea8` |
| HumanEval revision | `b9453e8f1865df0a0057a1383767caa400acfbcf` |
| Dataset gzip SHA256 | `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef` |
| Assertion image | `python@sha256:5fb4bc53dd0180955e31e5245b648bbb24a4ac4ab03a1de81513642325220bcd` |

Only official BF16 shards 25 and 26 were downloaded: 6,064,305,104 bytes combined, both authenticated against their pinned complete LFS SHA256 values. The 19 source MTP tensors contain 844,640,768 parameters and 1,689,281,536 tensor bytes. Matrices retain their BF16 values. Seven norm scales use `rawBF16.float()+1` without rounding back to BF16. Splitting fused expert gate/up produces 20 sidecar tensors, with a runtime weight payload of 1,689,303,040 bytes. The independent audit checked every matrix, norm, and expert gate/up interleave. All 11 source-pinned target inference files were freshly verified; the target tokenizer was retained.

The Mac coordinator and worker use the corresponding immutable package for each arm, with authenticated runtime build checks and package hashes. Windows is absent. The results commit is not a measured inference binary.

## Configuration and measurement

Single M5 Pro, 20 GPU cores, 48 GiB; AC / High Power; one fully local Metal worker at a time. Target affine8/group64 formats remain unchanged; MTP matrices are BF16 with FP32 arithmetic, activations, routing, KV, and state. No FP16 staging, quantization change, CPU weight offload, or system memory-limit modification is used. Wired limit remains zero.

All arms use Pi 1.4.0, seed 0, context 32768, maximum 4096 output tokens per request, task timeout 300 seconds, zero prefix cache, fused FP32 decode attention, managed prefill chunk 32, suffix/scratch 128, short-prefill threshold 262144, and matching unused `kv-transfer final`. Each task receives a fresh Pi session and workspace. Agent tool trajectories can differ between arms; the task set alone does not guarantee identical prompts.

Pilot order: HumanEval/156, /94, /72, /80, /149. The frozen 20-task order adds /2, /35, /32, /146, /137, /97, /154, /6, /161, /115, /131, /162, /50, /47, /148. Pilot results are not formal 20-task results.

Main does not emit `INFERENCE_METRIC`. An authenticated streaming observer captures equivalent real per-request `ServiceEvent.result.timings`, including every measured request and API failure with a raw worker result. Prefill throughput is sum(input tokens)/sum(prefill seconds). Decode throughput is sum(N−1)/sum(worker decode seconds), including drafting, verification, reconditioning, rollback, and replay. Neither Pi wall time nor a mean of per-request rates substitutes for these clocks.

Client TTFT is measured from observer HTTP send to first valid text or tool-call output. The two loopback observation hops are included; Pi-to-observer dispatch is outside that interval. Tool-call buffering can make TTFT much longer than worker first-token readiness. Missing first output remains missing. Official assertions use the pinned offline restricted Docker image; a separately checked incorrect implementation produced an actual `AssertionError` and exit code 1.

## Correctness gates and retained failures

| Gate | Tested source | Outcome |
| --- | --- | --- |
| Official draft/reference, shared head, poisoned suffix | `34c7c95c29b6121ab4d4677664bcea20082f5cc9` | All three real-model facts passed |
| Per-row verifier and rollback | `cca0c4fcc94bc15c3e8543269a9fa5e569a4c57d` | Rows 1/2/3 logits bit-identical; all 80 state digests exact; every replay offset and poisoned suffix passed |
| Fixed 128/4096 and natural EOS | `31b01db9354386c07c1a398322c31f2d0eb1dcb9` | Both widths passed warm 1/formal 3, exact tokens, terminal state, pending token, and fresh teacher-forced draft KV |
| Generation-time cache growth | `17965ce1f43e37398441c6f192c0a331efab228d` | Both widths crossed target/draft capacity 2048→4096 during generation with exact target and draft-cache checks |

G4 uses an optional hidden observer and independent cache recomputation; its timings are correctness diagnostics, not serving-speed measurements. Main/off serving fixtures separately use the frozen target outputs with observer copies disabled. The ordinary unit/process suite passed 182 tests, with nine opt-in facts skipped; it includes small physical Metal operator tests. Seven converter tests and 14 harness tests passed, with two harness tests skipped. Three-platform candidate CI passed.

The initial actual sidecar load failed because the official expert-down tensor omits `.weight`; the loader was corrected and an actual-artifact regression test passed. The first multi-row verifier matched greedy argmax but failed the exact state digest check, with logits relative L2 3.360e-7. The fix retained batched projections while reusing actual decode reduction/state pipelines; thresholds were not loosened. Both failed attempts remain in the evidence inventory.

A temporary data-listen URI setup error occurred before model readiness or benchmark requests. It contributes no inference measurements; the original service was restored before the corrected run. A stale postprocessing command in the first main run referred to an absent summary script; raw worker observations were retained and reaggregated with the audited equivalent-clock collector.

## Resource evidence and acceptance limits

Per-request whole-system swap-out deltas and sampled Metal allocations are collected. Candidate traces also cover tracked allocation, growth, and submission boundaries, including residency registration. These are not a continuous Metal-driver peak. The required continuous peak is therefore **not obtained**, and complete resource acceptance cannot be claimed even if samples stay below 38.5 GiB. Loading/warmup are separate; the earliest main loading/warmup swap bracket was not collected and is unknown.

The predeclared gates remain unchanged: enabled decode at least 1.20× main; enabled prefill latency and client TTFT regressions at most 5%; off decode loss at most 5%, off prefill/TTFT regressions at most 5%; official and Pi 20/20; exact replay output/state; continuous Metal high-water at most 38.5 GiB and each request's whole-system new swap-out at most 512 MiB. Failed or missing gates prevent adoption.

## Application pilot results

All numbers below are the actual fixed **five-task pilot**, not the requested formal 20-task comparison. The aggregate worker columns include every measured request, including tokens from API-rejected output. They are diagnostic rates where application success/usage or TTFT coverage is incomplete. Different agent trajectories prevent interpreting these aggregate differences as an isolated MTP effect.

| Arm / run ID | Worker prefill tok/s | Worker decode tok/s | Requests with worker clocks | Client TTFT median ms / coverage | Pi / official passes | Pi wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main / `main-pilot5-r1` | 270.606 | 67.138 | 19/19 | 12428.998 / 18/19 | 4/5 / 4/5 | 243.947 |
| Candidate off / `off-pilot5` | 266.854 | 65.975 | 18/18 | 12076.883 / 18/18 | 5/5 / 5/5 | 221.677 |
| Width 1 / `k1-pilot5` | 250.282 | 69.416 | 16/16 | 12159.142 / 16/16 | 5/5 / 5/5 | 203.175 |
| Width 2 / `k2-pilot5` | 251.396 | 68.315 | 21/21 | 13727.156 / 20/21 | 4/5 / 4/5 | 305.799 |

Width 1 has the higher aggregate pilot decode rate: +3.39% relative to main and +5.22% relative to candidate off. This is not a qualifying selection: neither width satisfies the regression/resource evidence gates. Width 2's faster five-request replay does not override its slower aggregate pilot rate or application failure.

Main HumanEval/72 failed with `Stream ended without finish_reason`; the raw model tool call ended without its required closing wrapper and the coordinator rejected it as malformed. Candidate width 2 HumanEval/80 also failed with `Stream ended without finish_reason`. No generic formatter change or selective answer retry was made. The width 2 failure is retained as an unresolved application failure; fixed replay equality does not establish the cause of a different agent trajectory. Both failures remain in the five-task denominator. Main and width 2 have complete raw worker clocks, but failed API requests report zero Pi usage; the public summaries reconcile those differences explicitly. They do not claim strict successful application metric completeness. Off and width 1 have complete request/usage/TTFT reconciliation.

### Identical-request replay controls

Before candidate pilots, the first five measured main requests and the known malformed main response were frozen. Each candidate arm reproduced all six requests' prompt token counts, output token IDs, and finish reasons exactly. The five originally successful outer API requests provide timing/TTFT controls; the malformed case is replayed directly at the worker and is excluded only from this predefined five-request timing control. It remains in the original application pilot and the six-request exactness control.

| Arm | Decode tok/s | Decode change vs main | Prefill latency change | Client TTFT median change |
| --- | ---: | ---: | ---: | ---: |
| Main reference | 67.274 | — | — | — |
| Candidate off | 64.972 | −3.423% | +2.434% | **+5.034%** |
| Width 1 | 69.514 | +3.330% | **+8.125%** | +4.020% |
| Width 2 | 70.471 | +4.752% | **+6.284%** | +4.368% |

The off TTFT result is above 5%; rounding does not turn it into a pass. Both enabled prefill regressions exceed 5%, and neither decode gain approaches 20%. No threshold was adjusted after measurement. These are single fixed replay controls, not a statistical-noise estimate. Main/off's earlier fixed-output checks left target KV capacity at 6144; enabled service checks/replays used 2048→4096 capacity. Prompt/output identity is exact, but allocation history is not identical and limits attribution. A future fresh-start 20-task comparison would answer a broader diagnostic question without making these failures disappear.

Main/off's separate fixed 4096-output serving fixtures passed exact token checks, with warm 1/formal 3 aggregate decode 66.429 versus 65.468 tok/s (−1.447%). Fixed 128-output rates were 69.731 versus 68.385 tok/s. These fixtures do not substitute for application or enabled-mode acceptance.

### Why high draft acceptance did not produce a large speedup

MTP runs on the Metal GPU. `MetalMtpDrafter` calls the native device runner; `gpu_mtp.mm` allocates Metal buffers and dispatches BF16-weight GEMV, attention, routing, and expert kernels. CPU code coordinates acceptance and cache bookkeeping and copies some hidden rows. There is no CPU weight-compute fallback. Weight residency registration evidence is narrower than proof of physical residency for every KV/workspace buffer.

Width 1 accepted 2121/2176 proposed tokens (97.472%); width 2 accepted 3512/3718 (94.459%). Across the measured width 1 requests, host component timers recorded 51.343 s verification, 6.994 s draft-cache catch-up, 2.340 s snapshots, and 1.189 s replay; prompt bootstrap added 8.454 s to prefill. Width 2 recorded 60.347 s verification, 6.418 s catch-up, 5.587 s additional draft steps, and 3.441 s replay; bootstrap added 14.017 s. Component timers are a diagnostic breakdown, not independent additive end-to-end clocks. In width 1, ordinary draft computation is charged to catch-up/first-draft, so `draftMs=0` does not mean drafting is free.

The evidence points toward verification, cache maintenance, GPU/host synchronization, and prefill bootstrap as the next profiling targets. It does not demonstrate that CPU execution is the bottleneck, or establish which kernel optimization will recover the missing payoff.

Per-request system swap-out maxima were 0 for main/off/width 1 and 224,460,800 bytes (214.063 MiB) for width 2, below the 512 MiB request ceiling. Endpoint Metal allocation maxima were 37,290,852,352 bytes for main/off, 38,924,894,208 for width 1, and 39,021,527,040 for width 2. All are below 38.5 GiB, but are samples, not the missing continuous-driver high-water. Candidate allocation/submission traces are retained separately and must not be labeled continuous peaks.

## Decision and unrun work

G5 cannot nominate an eligible width. The approved Issue explicitly says, “If neither qualifies, stop and retain a negative result.” No formal HumanEval20 run was started, no 20-task gain is reported, and the code is not recommended for adoption. Implementation correctness checks passing does not satisfy the performance, application, or complete resource gates.

The user was shown both replay regressions and asked whether to continue a diagnostic main/off/selected-on 20-task comparison while preserving all original adoption thresholds. Their follow-up asked whether MTP was running on CPU; that question is not authorization to change G5. Pending explicit continuation approval, the negative pilot results are retained for review. The higher aggregate pilot-rate width would be width 1 for that diagnostic comparison, without calling it G5-qualified.

## Evidence and reproducibility

Raw evidence is retained in the local ignored `artifacts/dual-host/issue76-official-mtp/` collection. [Public sanitized per-request/per-task summaries, costs, replay, runtime/restoration records, and evidence hashes](evaluations/issue76/) accompany this report. This location is restricted local evidence, not a public download. Private configurations, credentials, HTTP bodies, prompts, generated solutions, and assertion traceback text are excluded from publication. Each application phase retains the exact collector helper snapshots and hashes. Code/package versions and individual gate versions above must not be conflated.

The phase controller command is `python3 artifacts/dual-host/issue76-official-mtp/evaluate.py LABEL MODE COHORT`, with `MODE=main|off|k1|k2` and `COHORT=pilot|cohort`. It requires the local private configuration and authenticated, verified weight/package artifacts; it is not a standalone public reproduction command. Public implementation entry points and converter instructions are in PR #78's `docs/OFFICIAL-MTP.md`.

After each phase, the controller stops its verified owned processes and restores the original worker. Restoration checks include Ready state, registration, two real chats, unchanged original package/configuration hashes, and wired limit zero. The experiment does not change the adopted main baseline or the original deployed package.
