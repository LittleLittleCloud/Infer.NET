# Native DFlash2 versus official MTP: diagnostic HumanEval20 results

> Session closeout, 2026-10-25: the user authorized preserving research/results and closing experimental code PRs #74/#78/#81 without adoption. Earlier approval/pending/publication statements below describe the historical stage, not a current request to continue. The later #80 diagnostic20 includes the frozen MTP reference. See the [final disposition and combined summary](SPECULATIVE-DECODE-SESSION-CLOSEOUT-20261025.md). No acceptance threshold or original measurement was changed.

Experiment started 2026-10-24 America/Los_Angeles; hardware checks continued on 2026-10-25. [Issue #80](https://github.com/LittleLittleCloud/Infer.NET/issues/80), [candidate code PR #81](https://github.com/LittleLittleCloud/Infer.NET/pull/81). The human requested implementation of DFlash2 and comparison with MTP. No merge, production adoption, or deployment update is authorized.

**All four diagnostic 20-task arms are complete.** The numerical gate remains failed; this record does not recommend adoption. The complete draft intermediate numerical gate failed. All nine separate exact target-state diagnostics passed, including fixed 128/4096-output checks. The human explicitly approved diagnostic 20-task continuation on 2026-10-25 while retaining the numerical failure; [approval record](https://github.com/LittleLittleCloud/Infer.NET/issues/80#issuecomment-6394804249). Application measurements are complete and reported below. Neither CI nor a passing final normalized output substitutes for the failed gate.

## Sources, implementation, and comparison envelope

| Component | Immutable identity |
| --- | --- |
| Main baseline | `4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3` |
| Initial DFlash2 candidate | `3e78b6fbfc1961bab2f010380f48790e21606c4f` |
| Reference-order correction and checkpoint evidence | `c706064676ad5ad263ab23f146c285726167aa35` |
| Existing MTP reference | `17965ce1f43e37398441c6f192c0a331efab228d`, [PR #78](https://github.com/LittleLittleCloud/Infer.NET/pull/78) |
| Target | `mlx-community/Qwen3.6-35B-A3B-8bit@e06a74e6236a60c8367e1a3214e83d8b61b637b0` |
| Draft | `incoai/Qwen3.6-35B-A3B-DFlash2@51ef7b6923ad6c14cb1bb41c37a9041446496ab6` |
| Algorithm reference | `z-lab/dflash@07ebd93db9f472af339b644bb70221ad8428328a` |

The complete local draft file is 1,052,513,456 bytes with SHA256 `d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d`. Its 96 BF16 tensors contain 526,251,520 parameters; every source value was checked finite. Config SHA256 is `86d2e8e57e7c70bb867240708d00c0535ae9ed3c828dbdcb21584471c7bbce59`. All 11 source-pinned target inference files were freshly verified, and the original tokenizer was retained.

The implementation runs native C#/Metal, with BF16 draft matrices and FP32 arithmetic, activations, accumulation, and context KV. It shares the target's affine8 embedding/head. The six-layer network includes its trained dynamic convolutions and top-16 candidate-path selector, not an MTP substitute. Post-layer raw residual taps come from zero-based target layers 1, 6, 11, 16, 22, 27, 32, 37. Proposal attention is bidirectional within an anchor-plus-proposals block; committed target context uses a 2047-row sliding cache with absolute positions.

Default remains off. The new branch starts from main; it explicitly ports the minimal verifier, state-transaction, and allocation-trace dependencies from the unmerged MTP source. This does not assume PR #74 or #78 was adopted. GitHub Copilot CLI 1.0.103 used `claude-sonnet-5.5` for implementation and resumed review; the controlling agent independently audited weights, fixed stale-tap bounds, added actual generation-time page-growth checks, and ran hardware validation. CLI output alone is not a passing test.

The controlled comparison ran on M5 Pro 20 GPU cores/48 GiB, AC/High Power, local Metal, concurrency one, unchanged affine8 target/FP32 target KV and state, BF16 drafts, greedy, zero prefix cache, context 32768 and maximum 4096 outputs/request. DFlash2 blocks 4 and 8 contain three/seven proposals; MTP width 1 is the existing higher-rate pilot reference. Different widths compare available implementations, not an isolated algorithm effect at equal draft width. Splash's Q8 KV, quantized draft, memory-limit configuration, and native-command timing remain separate historical context.

## Builds and independent review

Native and managed Release builds, compile-only Metal shader/pipeline checks, 37 focused CPU tests, and candidate CI on Linux/macOS/Windows passed. Twelve full-model facts were skipped in the CPU run; skipped facts are not hardware passes. The final site build, 26 Python checks and 27 offline DOM checks passed with all eight measured runs. The new complete main control updates the best verified worker decode observation while the explicitly adopted baseline remains unchanged; failed MTP quality and partial off clocks cannot become verified SOTA. No planned value is used as a measurement.

Review corrections included full-window RoPE capacity, actual tail-block clamping, graph/child safe-handle lifetime, avoiding double-counted whole-device allocation, explicit small-output finiteness checks, the 2047-row context window, stale tap source-position/valid-count rejection, and exact scalar-versus-batched target tests. Native target suffix poisoning and real 2048→4096 growth were added instead of assuming a small `Reset` request shrinks an existing allocation. The first full reference failure and the correction below remain separate evidence.

## Failed draft numerical gate

The fixed criterion is relative L2 below `1e-4` and maximum absolute error below `0.01`. Both conditions are required; no threshold was rounded or loosened.

| Attempt | Source | First failed checkpoint | Relative L2 | Maximum absolute error |
| --- | --- | --- | ---: | ---: |
| `attempt1` | `3e78b6f` | Attention residual, draft layer 1 | 1.0296030363e-6 | 0.01171875 |
| `attempt2` | `c706064` | Attention residual, draft layer 1 | 1.0163243004e-6 | 0.01025390625 |

The independent C# reference initially combined `(base+dynamic)*value`; the pinned upstream equation accumulates the base and dynamic contributions separately in FP32. Correcting that reference expression changed no serving/native code. The second attempt still failed and was retained.

A second independent NumPy 2.3.5 implementation used exact BF16 source values, FP32 equations, actual captured target taps, and independently read affine8 embeddings. It did not call product native code. Its RoPE table uses the declared native convention: double angle/cos/sin rounded once into FP32 table entries. Chained intermediate residual magnitudes reached about 36817.57; maximum absolute differences reached 0.130859375 in the final raw residual. The final normalized hidden rows passed, with relative L2 about 3.2765e-6 and maximum absolute error 5.6862831e-5.

Same-input sublayer comparisons were also retained. Several earlier modules met the original bounds, but draft layer 5 attention/MLP still exceeded the absolute bound, at 0.01953125/0.119140625. Therefore this is not merely an untested assertion that all isolated operators pass. Floating-point reduction order and amplified intermediate magnitudes are plausible contributors; no complete operator-accuracy acceptance is claimed.

The shared-head check compares actual native vocabulary logits with an independent affine8 CPU classifier using the native final hidden rows. That check passed. Its separate embedding assertion only checked vector length; it did not independently compare a captured native embedding vector, and must not be described as doing so. Full draft reference inputs use independently dequantized target embeddings, but a dedicated native embedding-value checkpoint remains a coverage limitation.

## Separate target/state diagnostics

These are correctness diagnostics at `c706064`, not serving-speed measurements. They do not erase the failed draft numerical gate.

| Check | Status |
| --- | --- |
| Rows 1/2/3/4/8, every forced rejection offset: actual scalar versus batched per-row logits/taps, full target state before restore, every replay-prefix state; poisoned target suffixes | Passed |
| Committed context window and poisoned invisible ring slots | Passed |
| Fresh target-conditioned context after rejection versus an independent reference of its advertised prefix | Passed |
| Actual generation-time target capacity 2048→4096 and context eviction, both blocks; poison stability after eviction | Passed |
| Absolute positions crossing 2048/4096, default 4200-token fixture | Passed; near-32768 fixture not run |
| Output caps, cancellation and reset | Passed |
| Natural EOS versus serial greedy | Passed |
| Serial versus batched rejection replay | Passed |
| Fixed 128/4096 outputs, warm one/formal three, both blocks, exact tokens and terminal state | Passed |

The raw ring-window test validates selected independently projected K/V rows and poison stability. It is not a full-window independent NumPy forward comparison. The dedicated growth test verifies an actual capacity increase while generating, rather than inferring growth from a reserve argument. Final unused draft context may legitimately lag the terminal target; its valid prefix is explicitly reported and resets before another request.

## Diagnostic pilots and frozen width

The completed five-task pilots use the same frozen target and dataset. The enabled DFlash2 rate below is an **all-request diagnostic worker rate**, not successful-application throughput: each enabled pilot retains one API failure, and strict Pi usage reconciliation omits failed API output until the recorded failure is explicitly included.

| Pilot | Worker decode token/s | Worker prefill token/s | Official assertions | Pi completed | Requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh main | 68.012 | 272.730 | 5/5 | 5/5 | 15 |
| Candidate off | 67.987 | 272.272 | 5/5 | 5/5 | 19 |
| DFlash2 block 8 | 53.127 | 265.692 | 4/5 | 4/5 | 15 |
| DFlash2 block 4 | 59.091 | 265.862 | 4/5 | 4/5 | 13 |

Block 4's failed HumanEval/149 request generated 327 worker tokens and ended at EOS, while Pi reported `Stream ended without finish_reason`; no valid client first-output timestamp was captured for that request. No answer retry or exclusion was performed. All 13 worker clocks are complete and token totals reconcile when the recorded API failure is included. All six frozen replay requests matched main target tokens and finish reasons; the five timing controls measured 61.668 token/s versus main 67.867 token/s. These are separate fixed-input controls, not substitutes for the HumanEval cohort.

Block 4 also passed all eight fixed128/4096 warm/formal controls and 15 serving checks. Its fixed4096 formal rates were 53.965, 53.979, and 53.967 token/s, with zero new swap-out. Block8 completed at 53.127 token/s, with 4/5 official passes, 15 complete worker clocks, and maximum per-request swap-out 283,131,904 bytes. The predeclared selection therefore froze block4 before any formal20 measurement; [decision](https://github.com/LittleLittleCloud/Infer.NET/issues/80#issuecomment-6395198735). Fresh four-arm HumanEval20 is complete below. The width selection rule remains the higher aggregate diagnostic worker rate among target/resource-safe pilots, tie block4; it is not an adoption decision.

## Complete diagnostic HumanEval20 comparison

All four arms attempted all 20 fixed tasks without selective answer retries. Rates below include **every measured model request**, including output rejected by the API. They are diagnostic worker clocks; failed application tasks remain failed and missing client first-token samples remain missing.

| Arm | Decode token/s | vs main | Prefill token/s | TTFT median ms (coverage) | Official assertions | Pi completed | Requests | Worker clocks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fresh main | 67.782 | +0.00% | 271.810 | 10917.017 (105/105) | 20/20 | 20/20 | 105 | 105/105 |
| Candidate off | Not obtained | Not obtained | Not obtained | 11124.568 (114/115) | 20/20 | 19/20 | 115 | 114/115 |
| Official MTP width1 | 69.563 | +2.63% | 255.429 | 11249.222 (87/88) | 19/20 | 19/20 | 88 | 88/88 |
| DFlash2 block4 | 57.981 | -14.46% | 265.762 | 11563.235 (92/92) | 20/20 | 20/20 | 92 | 92/92 |

DFlash2 block4 measured **0.834× MTP** (-16.65%) and **0.855× fresh main** (-14.46%). These figures compare the current implementations under this envelope; BF16 drafts, different widths, bootstrap attention and application trajectories prevent attributing the difference solely to the algorithms. They do not establish parity with Splash's separate Q8-KV/quantized-draft/memory-limit/native-command configuration.

| Arm | Known input / output tokens | Pi wall seconds | Errors / timeouts | Strict metrics complete | Max per-request system swap-out MiB | Sampled Metal GiB |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| main-human20 | 247915 / 21542 | 1237.393 | 0 / 0 | True | 0.000 | 34.730 |
| off-human20 | 294850 / 22496 | 1450.768 | 1 / 1 | False | 55.359 | 34.812 |
| mtp-human20 | 195854 / 17383 | 1025.562 | 1 / 0 | False | 83.781 | 36.342 |
| df-human20 | 209094 / 19706 | 1132.802 | 0 / 0 | True | 0.000 | 35.858 |

The candidate-off arm has one canceled request without a terminal worker clock after HumanEval/32 timed out. Its entire-cohort prefill and decode are **Not obtained**; the known subset alone measured 67.810 decode and 271.726 prefill token/s; the known114/115 clock subset remains separate in JSON and chart sample coverage, never substituted for the missing whole rate. Known token totals exclude that unknown request and are lower bounds, not zero-filled totals. Official assertions20/20 and Pi completion19/20 describe different outcomes. Strict `metricsComplete=false` is not hidden: Pi SDK usage may omit output on API failures. The separate worker diagnostic rates require complete unique worker clocks and a per-request reconciliation that explicitly accounts for recorded API errors. TTFT covers only requests with an actual first valid text/tool-call output; the median cannot repair missing samples. Per-task official status, request counts, wall time and tokens remain in the linked JSON files.

DFlash2's Pi wall time was lower than main despite slower worker decoding: it made 92 rather than105 requests and consumed209,094 rather than247,915 input tokens. Application trajectories differ; wall-time reduction is not a decode speedup. All six fixed-input replay controls remained exact, and DFlash2 also decoded more slowly on those controls.

### Same-input replay controls

These six frozen requests are separate diagnostics, never answer retries. Five use identical outer API bodies and one retains a known malformed-output baseline case. Every arm must match target tokens and finish reasons; five valid controls provide timing comparisons against a fresh main reference primed to the same capacity.

| Arm | Exact replay | Five-control decode token/s | vs main | Prefill latency change | TTFT median change |
| --- | ---: | ---: | ---: | ---: | ---: |
| main-human20 | 6/6 (True) | 67.800 | +0.00% | +0.00% | +0.00% |
| off-human20 | 6/6 (True) | 68.011 | +0.31% | -0.03% | +0.07% |
| mtp-human20 | 6/6 (True) | 70.640 | +4.19% | +6.51% | +2.78% |
| df-human20 | 6/6 (True) | 61.747 | -8.93% | +2.47% | +7.15% |

### Cost and adoption interpretation

DFlash2 accepted 14,025/17,091 proposals (82.061%); MTP accepted 8,564/8,875 (96.496%). Proposal widths differ, and terminal proposals may not all become emitted output, so this is not a quality score or a direct algorithm comparison. DFlash2 host timers recorded draft60.371s, verify229.447s and replay37.115s; MTP recorded verify206.711s, catch-up27.830s and replay6.683s. These totals cover different request/output counts. They point to draft work and frequent rejection restoration as additional costs, while verification dominates absolute time. Draft weights and execution remain on the Metal GPU; this result does not demonstrate CPU drafting.

[Component totals](evaluations/issue80/speculation-costs.json) retain proposal acceptance and host timers for draft, verification, ingest/catch-up, snapshot and replay. These timers are diagnostic and are not assumed independently additive. DFlash2 `draftMs` includes the six-layer network, shared head, top16 and trained selector. MTP may charge ordinary drafting to catch-up/first-draft rather than `draftMs`; zero `draftMs` does not imply free drafting.

**Do not adopt this candidate.** Both failed DFlash2 intermediate numerical attempts remain failed, and a continuous Metal-driver peak was not obtained. Exact target diagnostics, official task passes and diagnostic throughput cannot substitute for those missing acceptance conditions. Code PR81 and independent results PR82 remain unmerged; Issue80 remains open. Further optimization should profile verification/replay and draft/context costs with unchanged target semantics, and resolve numerical validation before any adoption proposal.

### Reproduction and evidence

The source hashes, authenticated actual builds, nine package file hashes per arm and final restoration checks are in [runtime-and-restoration.json](evaluations/issue80/runtime-and-restoration.json). Dataset revision `b9453e8f1865df0a0057a1383767caa400acfbcf`, gzip SHA256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`. Ordered tasks: 156,94,72,80,149,2,35,32,146,137,97,154,6,161,115,131,162,50,47,148. Pi1.4.0, seed0, timeout300s, context32768/output4096, fresh session/workspace; no prefix cache. Worker prefill is sum(inputs)/sum(prefill seconds); decode is sum(N−1)/sum(first-to-last output seconds). Observer client TTFT ends at the first valid text/tool-call output and excludes Pi-to-observer dispatch. The driver is frozen from MTP source `17965ce1f43e37398441c6f192c0a331efab228d`; per-phase helper SHA256 snapshots and raw manifest/task/request hashes are in [evidence.json](evaluations/issue80/evidence.json). The command shape is `python3 evaluation/evaluate.py LABEL main|off|mtp|df4 cohort`, executed serially under verified exclusive-worker maintenance with restore in `finally`. The helper sources and private model/chat logs remain restricted local ignored artifacts, not public downloads. Official Docker checks use all20 outputs and preserve missing solutions; the negative control actually fails.

The site source contains separate five-task and twenty-task configurations and selectable failed DFlash2 points. Proposed point: `/#experiment=dflash2-comparison-20261024&run=issue80-df-human20`. The results PR saves evidence only; it does not publish Pages or change the adopted baseline. No site publication is claimed before human-authorized merge and successful deployment.

## Resources, restoration, and evidence

No system memory limit, model precision, offload policy, Windows service, or production package was changed. Required boundaries remain Metal allocations at most 38.5 GiB, per-request whole-system new swap-out at most 512 MiB, zero OOM/crash/active NaN/Inf/CPU weight offload, and wired limit zero. Loading/warmup and unit-gate resource samples are separate from formal per-request resource acceptance. Allocation-boundary/submission samples are not a continuous Metal-driver peak; that continuous peak remains not obtained.


| Arm | Endpoint footprint / OS-reported process peak (raw vmmap units) | Sampled minimum system free MiB | Sampled maximum compressor MiB |
| --- | --- | ---: | ---: |
| main-human20 | 35.1G / 35.1G | 14.328 | 2911.578 |
| off-human20 | 35.2G / 35.2G | 50.453 | 2523.578 |
| mtp-human20 | 37.0G / 37.3G | 43.750 | 3972.500 |
| df-human20 | 36.2G / 36.2G | 48.828 | 3134.797 |

[Resource samples](evaluations/issue80/resource-summary.json) are restricted to each HumanEval observation window and exclude loading/fixed-control warmup. Every sampled power status was AC. System free/compressor/wired values are whole-machine `vm_stat` indicators, not an OS pressure-level gauge or worker attribution. The OS pressure-level gauge was not obtained. The vmmap process high-water is separate from the missing continuous Metal-driver peak.

Each failed model group stopped at its failed fact and restored the original worker: Ready state, registration, two real chats, original package/configuration hashes unchanged, wired limit zero. The complete target diagnostic group also restored the original worker successfully. Application phases use separate exclusive windows and restoration records; all four application phases also restored the original worker successfully, with Ready state, registration, two real chats, unchanged package/configuration hashes and wired limit zero.

Raw artifacts remain restricted local ignored `artifacts/dual-host/issue80-dflash2/`, including pinned original references, authenticated weights, source audits, CLI/model provenance, build logs, per-fact TRX/binary hashes/resources, and independent captured-checkpoint comparisons. Public sanitized summaries and hashes accompany this record. Local paths are not public download links. Credentials, private configurations, prompts, generated solutions, and raw model feature arrays are excluded from publication.
