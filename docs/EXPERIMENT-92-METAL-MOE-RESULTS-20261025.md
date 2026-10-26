# Experiment 92: Metal routed expert small-row specialization

**Result: reject this candidate for adoption.** All three lane variants regress in rotating-expert microbenchmarks. The four-lane representative also slows full-model decode by 7.49–8.54% across the fixed workloads, exceeding the registered 5% regression boundary. Diagnostic HumanEval 5 confirms the regression while both arms pass the official assertions. The original kernel remains selected; no code merge, deployment-default change, or 128K acceptance is claimed.

[Issue #92](https://github.com/LittleLittleCloud/Infer.NET/issues/92) · [approval](https://github.com/LittleLittleCloud/Infer.NET/issues/92#issuecomment-6406551531) · [code PR #93](https://github.com/LittleLittleCloud/Infer.NET/pull/93) · [independent results PR #94](https://github.com/LittleLittleCloud/Infer.NET/pull/94).

## Baseline, scope, and implementation

Plan revision 1 was approved on 2026-10-25. The results branch starts independently from main `4f51f241e87999c8cc3f79d733cd95ade9c711cd`; it does not require the candidate implementation. The measured runtime candidate is `7f114e94661c3a39ca46d888322f438ecd3b0830`. A later code-PR commit changes only the check driver and plan disposition; [code-identity.json](evaluations/issue92/code-identity.json) verifies identical product kernel sources and records the exact final SHA. A checkout SHA is not substituted for the measured executable's build.

Hardware: Apple M5 Pro, 20-core GPU, 48 GiB unified memory, AC power with the recorded High Power configuration. The full model runs locally on Metal. Windows does not participate in the measurements and no Windows service action was performed. Native AOT workers were built independently for main and candidate; actual builds, executable/native-library hashes, selectors, and harness identities are in [runtime-fixed.json](evaluations/issue92/runtime-fixed.json) and [runtime-pilot.json](evaluations/issue92/runtime-pilot.json).

Model: `mlx-community/Qwen3.6-35B-A3B-8bit`, revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, checkpoint affine8/group64 configuration. Both arms retain FP32 KV/recurrence, FP32 dequantized products/accumulation, fused-FP32 decode attention, managed prefill 32 and scratch 128, no prefix cache, and no speculation. The diagnostic prefill configuration is deliberately the one used in experiment #70; its absolute prefill throughput must not be compared with the fast native-prefill profile or the formal 128K goal.

The original decode kernel assigns a complete SIMD32 group to each output. The opt-in candidate assigns 4, 8, or 16 cooperating lanes per output and keeps multiple outputs in one SIMD32 group. Only routed expert gate/up and down select the new kernels. Packed checkpoint weights, gate/up fusion and SwiGLU, shared experts, routing, dense projections, LM head, prefill kernels, attention, KV, recurrence, selected expert count, and command-buffer scheduling keep their existing representations and algorithms. Summation grouping changes; bitwise intermediate identity is not assumed.

`DEEPINFER_METAL_EXPERT_LANES=0|4|8|16` is read natively. Invalid selections and conflicts with generic kernels are rejected. A native log/getter exposes the selection; unsupported formats explicitly use the original path. Zero remains the default. This is not an INT8 activation or tensor-matrix implementation and introduces no reduced-precision quality budget.

## What was borrowed from Splash

The reference is pinned to `incoai/splash` revision `1482f6bbc2abf12375332725302019594150b6cb`, not an assertion about all current upstream code. Its [column tile](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/common/gguf_column_tile.h), [expert dispatch](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/shared/moe_gguf.metal), and [tile policy](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/ops/MoE.hpp) specialize expert tiles with only one or two live rows instead of always padding them to MPP matrix tiles.

The transfer is the sparse-row specialization principle. Splash's per-lane output approach uses a companion tiled weight image and BF16/half operands. This experiment preserves our row-major packed image and FP32 operands; it is not a faithful port of that kernel. Eight selected experts are eight distinct matrices, not eight rows sharing one weight tile. Our negative result does not establish that Splash's implementation is slow.

Historical larger-row decode tiles and prefill double buffering already had negative evidence in `QWEN36-OWN-BENCHMARK.md` and `EXPERT-FLOW.md`. Those records remain intact; prefill tuning was outside this experiment.

## Kernel screening

Two cohorts used forward and reverse variant order. Each cell has two excluded warmup batches and seven formal batches, each containing 64 dispatches with buffer barriers. Gate/up uses K=2048,N=512; down uses K=512,N=2048, with eight selected experts. Fixed-ID batches reuse one set of experts; rotating batches visit all 256 experts twice. These are synthetic weights at real dimensions, not checkpoint-weight performance.

GPU command time divided by 64 is batch-amortized time including dispatch/barrier overhead, not an individual hardware-counter interval. The reverse confirmation uses stronger coefficient/input fixtures than the initial cohort; do not pool their trials. [kernel.json](evaluations/issue92/kernel.json) retains every warmup and formal timing, both oracle cohorts, source SHAs and statuses. [source-hashes.json](evaluations/issue92/source-hashes.json) verifies that product kernel sources are identical between those revisions.

Reverse-order confirmation:

| Projection/cache case | Original, microseconds | 4 lanes, latency change | 8 lanes, latency change | 16 lanes, latency change |
|---|---:|---:|---:|---:|
| gate-up / fixed | 39.273 | +6.11% | +7.69% | +5.33% |
| gate-up / rotating | 63.559 | +30.94% | +32.14% | +30.82% |
| down / fixed | 21.077 | -1.46% | +0.68% | +9.33% |
| down / rotating | 33.561 | +18.25% | +17.10% | +17.28% |

Every rotating cell also regresses in the initial cohort. The small fixed-ID down improvement for four lanes in the confirmation is absent in the initial cohort and is not a reproducible gain. Four lanes was fixed as a low-overhead representative in the full-model controller before that evaluation, not treated as a proven winner.

Fewer cooperating lanes reduce the number of independent SIMD output chains and change access over row-major weights. Memory access, longer serial chains, and scheduling are plausible explanations; no DRAM transaction, occupancy, or register-spill counters separate them. A claim of physical bandwidth saturation or a measured hardware-utilization percentage would be unsupported.

## Fixed full-model requests

Each arm/shape has one excluded full warmup and three formal requests. Order is main short/8K, candidate short/8K/32K, then main 32K. This counterbalances the large-input arm without eliminating all thermal/order drift. No competing GPU benchmark or native/AOT build ran during formal requests. Greedy fixed 128 outputs ignore EOS for these stress cases only. The request profile flag collects worker timings; hardware timestamp profiling is off.

| Input/output | Main decode, tok/s | Candidate decode, tok/s | Decode change | Median prefill latency change |
|---|---:|---:|---:|---:|
| 21/128 | 68.547 | 62.690 | -8.54% | -1.64% |
| 8192/128 | 59.470 | 55.017 | -7.49% | +0.02% |
| 32768/128 | 46.065 | 42.341 | -8.08% | +2.02% |

Decode is the sum of 127 intervals per formal request divided by summed worker decode seconds. Prefill deltas use the three-request median. Absolute aggregate prefill, individual values, resource checks, and all output IDs remain in [fixed.json](evaluations/issue92/fixed.json). All repeated 128-token sequences match within and across arms for each shape. The candidate exceeds the registered 5% decode-regression boundary in every tier. Unchanged prefill algorithms do not make the observed small prefill differences attributable to the new decode kernels.

## Diagnostic HumanEval 5

Both arms use the fixed seed-0 subset (seed selects tasks, not model sampling): HumanEval/156,94,72,80,149. Pi1.4.0, fresh workspace/session per task, context 32768, output limit 4096, timeout 300 seconds. A separate 21-input/32-output warmup is excluded. The official dataset revision is `b9453e8f1865df0a0057a1383767caa400acfbcf`, SHA256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`. Official assertions run in the pinned restricted Docker image, with a failing negative control on both arms.

| Metric | Fresh main | Four-lane candidate |
|---|---:|---:|
| Official assertions | 5/5 | 5/5 |
| Complete worker clocks | 15/15 | 17/17 |
| Input tokens | 31943 | 36851 |
| Output tokens | 3941 | 4209 |
| Prefill tok/s | 272.030 | 272.121 |
| Decode tok/s | 67.890 | 62.534 |
| Median observer TTFT, ms | 11264.337 | 11685.285 |
| Total Pi task wall, seconds | 176.873 | 204.121 |

Candidate aggregate worker decode changes by -7.89%. Both cohorts have zero task errors/timeouts and complete valid-output TTFT coverage. Their application trajectories differ in request/token counts, so this pilot ratio is diagnostic rather than a controlled operator-only speedup. The fixed-input regressions are the clearer attribution evidence. Formal20 is not run under the registered no-credible-benefit early stop; no HumanEval 20 adoption claim is made.

The observer adapter comes from the independently frozen #84 harness source, identified in runtime metadata. Worker clocks are raw `ServiceEvent.result.timings` normalized explicitly to the existing summarizer, not native emitted `INFERENCE_METRIC` logs. Throughput is sum(input tokens)/sum(prefill seconds) and sum(N-1 decode intervals)/sum(decode seconds). Pi task wall is separate. TTFT runs from observer HTTP send to first valid text/tool-call output, over two loopback proxy hops; Pi-to-observer dispatch is excluded, and tool-call buffering can include decode. Roles, usage and keepalives do not count as output. Raw observer metadata calling its executable “main” is historical harness wording; each arm's actual main/candidate build is recorded explicitly.

[Main pilot evidence](evaluations/issue92/main-pilot5.json) · [candidate pilot evidence](evaluations/issue92/candidate-pilot5.json). Charts use a separate fixed5 diagnostic configuration and preserve the adopted baseline selection.

## Correctness and resource limits

Independent scalar affine8 checks cover coefficient signs, zero/cancellation and non-power-of-two inputs, tails, selected expert boundaries/reordering, distinct per-slot down inputs, finite outputs and sentinels. Pre-fixed FP32 limits are relative L2<1e-4 and maximum absolute<0.01. The strengthened confirmation's worst values are 1.65173082294e-6 and 0.000125885009766. Existing graph hardware checks pass 60 cases plus handoff/scratch guards in each kernel cohort.

Earlier graph-helper comparisons reused the direct-kernel output buffer. A final stronger retest clears it before graph dispatch and checks sentinels, with Metal shader validation/debug layer enabled: all 320 reported direct/graph comparisons pass. The final project driver is equivalent to the validated temporary driver after normalizing its include path. These comparisons are not 320 independently generated fixtures. [Final oracle](evaluations/issue92/final-oracle.json), [code provenance](evaluations/issue92/code-identity.json).

Independent whole-model state/logit acceptance is **Not Obtained**; matching fixed token sequences and five official tasks do not replace it. Continuous Metal-driver physical-residency and per-submission physical-residency peaks are **Not Obtained**. No formal quality/resource acceptance is claimed.

All 24 fixed requests, including six excluded warmups, pass the recorded allocation/steady-swap predicates. The 18 formal fixed requests have zero new whole-machine swap-out. Two 32K warmups record109.5625 and212.09375MiB swap-out; other fixed warmups record zero. Maximum completed Metal allocation is 35.880386GiB for fixed requests and 34.647629GiB for each application cohort. Both post-warmup application intervals have zero new swap-out, bounding all 15/17 application requests at zero.

This was not a no-swap experiment: the approximately1Hz fixed-run monitor records 19.610077GiB cumulative whole-machine swap-out over its sampled interval; the application monitor records 9.306061GiB. These include loading/switching activity, are not current swap occupancy or per-process attribution, and omit intervals before the first sample and final restoration. Application load snapshots record about 2.414GiB for main and 6.892GiB for candidate. All slow valid observations remain. Sampled RSS does not measure full Metal residency or footprint. See [resources.json](evaluations/issue92/resources.json).

[128K capacity projections](evaluations/issue92/capacity-preflight.json) are not measured submission peaks or capacity acceptance. No 128K request ran, no wired limit changed, and KV precision was not reduced to fit a request.

## Restoration, evidence, and disposition

Every controller restored the original deployment. The final authenticated ready/registry audit and deterministic real `cuda-metal-split` request pass, with the same Mac PID before/after that request and identical pre-experiment token IDs. Original executable, native library, configuration and LaunchAgent hashes match; the LaunchAgent is running, exactly one model worker remains, all experimental ports are closed, and the wired limit remains 0. No original coordinator/TUI/Windows process was switched. [Restoration](evaluations/issue92/restoration.json), [final audit](evaluations/issue92/final-audit.json).

Raw evidence is restricted local material in the experiment worktree under `artifacts/dual-host/issue92-*`; the public [inventory](evaluations/issue92/evidence.json) provides hashes, not downloadable logs. Private configurations, credential stores, plists and locks are excluded. The first microbenchmark executable was overwritten during recompilation and its startup hash is **Not Obtained**; the retained confirmation executable and all full-model runtime binaries have recorded hashes. Numerical failures, missing gates, test-harness corrections and the pre-benchmark orchestration error are retained in [failures.json](evaluations/issue92/failures.json).

Reproduction: from the candidate code branch, `bash scripts/check-metal-expert-decode.sh`; `bash scripts/check-metal-expert-decode.sh --timing`; `bash scripts/check-metal-expert-decode.sh --timing --reverse`. Full-model evaluation requires the approved exclusive-GPU/service-restoration procedure and pinned local assets, not those operator commands alone. The results-branch `scripts/summarize-metal-moe-small-row.py --raw <restricted-fixed-run> --output <public-evidence-directory>` reconciles 24 fixed requests, their workload/token hashes, runtime identity and metrics. The [validation record](evaluations/issue92/validation.json) includes 131 passing managed checks, 26 homepage Python checks, 27 offline DOM checks, the Pi harness checks (14 tests, two skipped), site build, evidence reconciliation and whitespace checks. Offline DOM checks do not establish browser layout. Site sources are ready for review and are not published; the results PR remains unmerged.

Retain this negative record and keep the original implementation. A future experiment could assess whether a compatible expert weight layout makes a small-row kernel worthwhile; it needs a separate implementation/cost/quality plan. This round provides no basis to adopt the smaller lane groups or claim that every single-token MoE operation should use a matrix unit. Issue #92 and both PRs await manual review; no merge, automatic code-PR closure, or baseline update was performed.
