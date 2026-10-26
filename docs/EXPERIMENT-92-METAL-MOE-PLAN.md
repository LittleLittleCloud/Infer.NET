# Experiment 92: Metal single-token routed expert specialization

Plan revision 1 was approved by the human user on 2026-10-25. [Issue #92](https://github.com/LittleLittleCloud/Infer.NET/issues/92), [approval record](https://github.com/LittleLittleCloud/Infer.NET/issues/92#issuecomment-6406551531). Implementation is an unadopted candidate. Measurements are pending.

## Problem, baseline, and evidence

Baseline: latest `main`, `4f51f241e87999c8cc3f79d733cd95ade9c711cd`. Start a fresh branch from this main commit after approval; do not stack on unmerged experiments #33, #70, or policy PR #89. If main changes, audit and record the changes before implementation.

Experiment #70 / results PR #91 measured the entire MoE region at 40.010%, 33.635%, and 24.945% of sampled GPU region time for 21-, 8192-, and 32768-token inputs. These are encoder-instrumented shares, not hardware utilization or measurements of routed experts alone. The region includes routing, shared/routed expert projections, activation, merge, and next normalization. PR #90 remains an unadopted diagnostic implementation; its code will not be silently imported.

Current decode already fuses gate/up and SwiGLU, keeps routing on GPU, and uses one command buffer per token. In `native/DeepInfer.Metal/quantized.mm`, each SIMD32 group owns one output row; the graph dispatches four SIMD groups per threadgroup. Recommending those existing fusions as new work would be misleading.

## Splash reference and transfer limits

Pin source review to `incoai/splash` commit `1482f6bbc2abf12375332725302019594150b6cb`, the revision used by the existing project review, rather than claiming an audit of current upstream main:

- [Small-live-row column tile](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/common/gguf_column_tile.h): each lane owns an output column and computes register dot products without staging/barriers. The source explains that one- or two-row expert tiles can be bounded by padded MPP matrix operations.
- [Expert kernel dispatch and fused gate/up](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/shared/moe_gguf.metal): chooses the column path for sparse live rows and matrix tiles for larger groups.
- [Plan and tile policy](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/ops/MoE.hpp).

Transfer the small-live-row specialization principle, not the entire kernel: Splash uses a different tiled weight layout and BF16/half operands in the cited column path. This experiment preserves our row-major packed affine8 weights and FP32 operands. Eight selected experts are eight distinct matrices, not eight rows sharing one matrix. It therefore does not assume that padding a single token to an M8 matrix tile is beneficial, or claim native INT8 tensor computation.

The historical larger-row decode tile did not improve the default (`docs/QWEN36-OWN-BENCHMARK.md`). Prefill expert double buffering regressed and direct intermediate flow had no stable 32K gain (`docs/EXPERT-FLOW.md`). Those results remain intact; prefill tuning is outside this experiment.

## Hypothesis and bounded implementation

A smaller lane group per output may expose more independent output chains within each SIMD32 group and reduce reduction/scheduling overhead for single-token routed experts, without weight repacking. The cost is less contiguous memory access across lanes; a regression is plausible and will be retained.

1. Add an explicit opt-in decode expert selector, defaulting to the original kernel. Verify the actual selected native path in runtime metadata; do not rely on managed environment changes implicitly reaching native getenv.
2. Implement affine8/group64 routed expert gate/up and routed expert down variants with 4, 8, and 16 cooperating lanes per output. Each SIMD32 group handles 8, 4, or 2 outputs respectively. Keep current gate/up fusion, SwiGLU, selected-expert indexing, and per-slot down inputs. Retain bounds checks and reference fallback for unsupported shapes/formats.
3. Preserve checkpoint bytes, BF16 scale/bias storage, FP32 dequantized multiplication/accumulation, routing, KV, recurrence, shared expert, dense projections, LM head, prefill, and command-buffer scheduling. No coefficient factoring, activation quantization, weight expansion/repacking, speculative decoding, CPU offload, or CUDA changes in this round.
4. Implement only the minimal selector, kernels, independent validation, isolated timing driver, and runtime evidence required. Reimplement the principle for our layout; any copied source would require appropriate Apache-2.0 attribution.

Device: M5 Pro 20-core GPU / 48 GiB, Mac-only full-local Metal. Windows service remains unchanged. Expected benefit is **to be measured**. For orientation only, reducing the entire short-input MoE region by 15% would reduce sampled total time by approximately 6% before other effects; this is arithmetic sensitivity, not a prediction for routed-expert kernels or an observed result.

## Evaluation protocol and predeclared decisions

- Pin model `mlx-community/Qwen3.6-35B-A3B-8bit`, revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, affine8/group64. Use identical FP32 KV/recurrence and current fused-FP32 attention in both arms. This isolates decode kernel scheduling and is not Q8-KV or 128K acceptance.
- First verify actual GPU/device, source SHAs, binary/native hashes, native selector, unsupported-input rejection, and no silent fallback. Use a separate artifact directory/run ID and experimental ports. Before exclusive GPU tests, record original deployment identities/hashes; pause only the identified Mac model worker, then restore its unchanged deployment and verify a real request in a finally path. Do not modify Windows or merge anything.
- Independent operator oracle: decode packed checkpoint-format bytes independently on CPU and compute FP32 reference gate/up/SwiGLU/down. Check relative L2 < 1e-4 and maximum absolute error < 0.01; test cancellation, scale/bias signs, zero/large inputs, row tails, reordered/boundary expert IDs, and distinct per-slot down inputs. Check sentinels, finite outputs, repeated execution, and unsupported formats. Keep existing FP32 whole-model/state regression checks and eight-token reference checks. Diagnose first routing/logit divergence; no post-result tolerance relaxation.
- No new reduced-precision quality budget is needed because this proposal does not introduce reduced precision. Any later INT8/BF16 arithmetic proposal must separately register its quality budget and receive approval; PR #89 is not implicitly adopted here.
- Tune only on isolated real expert dimensions: gate/up K=2048,N=512 and down K=512,N=2048, eight selected experts. Include warm-cache and rotating-expert cases. Retain every variant and timing distribution; record warmups, timed iterations, synchronization boundaries, bytes, and device conditions. Synthetic timing is selection evidence, not full-model performance. Freeze the selected variant before held-out application evaluation; use distinct fixtures for correctness/selection and held-out validation.
- Run fixed 21/8192/32768-input, 128-output A/B requests, no prefix cache, greedy and EOS ignored only for these stress cases; one excluded full warmup and at least three formal trials per arm/shape, serial GPU use. Use identical managed prefill chunk 32 and scratch 128 to match #70's diagnostic configuration, recording that this is not the fast native prefill profile. Counterbalance A/B order. Keep complete token IDs, prefill/decode timings, failures, and slow rounds. Profiler off for the primary speed measurements. Use isolated kernel GPU timing for attribution; do not infer expert substage shares from #70's coarse region.
- HumanEval SOP: fixed seed-0 pilot 5 for iteration, then fixed seed-0 cohort 20 for fresh main/candidate adoption comparison. Pin dataset revision/hash, Pi version, tokenizer/template, context 32768, output limit 4096, timeout 300 seconds, fresh sessions/workspaces and identical observer topology. Include warmup separately. Official per-problem assertions plus failing negative control; Pi completion alone is insufficient. Candidate official passes must be no fewer than fresh main, with no new errors/timeouts; disclose all per-task differences. Do not tune on cohort 20.
- Obtain complete per-request worker timings and valid-output TTFT coverage. If using the #70 observer adapter, freeze its independent harness SHA and explicitly label normalization of ServiceEvent timings rather than claiming native INFERENCE_METRIC output. Aggregate throughput as sum(input tokens)/sum(prefill seconds) and sum(N-1 decode intervals)/sum(decode seconds); report task wall time separately. Missing or mismatched metrics make the comparison incomplete.
- Proposed adoption threshold: at least **5% higher aggregate worker decode throughput on HumanEval20** versus fresh main, no more than 5% regression in prefill latency or decode throughput in any fixed-input tier, all correctness/resource checks passing, and sufficient reproducibility to distinguish gain from drift. The 5% gain is a proposed engineering decision threshold for approval, not an existing measured benefit. If the pilot shows no credible gain, stop before cohort 20 and retain a negative result; no adoption claim.
- Resource checks follow `docs/QWEN36-8BIT-TARGETS.md`: Metal allocations <=38.5 GiB, per-steady-state-request whole-machine new swap-out <=512 MiB, disclose load/warmup swap separately, process footprint/memory pressure, residency evidence or explicitly Not Obtained, no OOM/crash/NaN/Inf/truncation/CPU weight offload. Perform capacity preflight before long inputs; do not raise wired limits or change KV precision to fit a request. No 128K run or formal target-completion claim is in scope.

## Delivery and tracking

After approval: branch `codex/experiment-<issue>-metal-moe-small-row` from audited main, code PR before evaluation, then an independent results PR from main regardless of outcome. Record raw evidence hashes/access restrictions, runtime manifests, protocol coverage, validation, failures, and restoration. Preserve all old experiment worktrees/evidence.

Homepage entry: existing [decode trend dashboard](https://littlelittlecloud.github.io/Infer.NET/#decode-trend), planned stable run prefix `metal-moe-small-row-20261025`; record actual per-run links in this Issue and results PR once data exists. No expected number is plotted as measured. Add the planned experiment entry through the normal project/site PR process after approval; no publishing or merge is authorized.

Approved scope: plan revision 1, including Mac-only scope, bounded 4/8/16-lane variants, unchanged arithmetic, HumanEval5/20 protocol, and the proposed 5% adoption threshold. Code PR and independent results PR will be linked from Issue #92. Results: Not Obtained.


## Evaluation disposition

The measured runtime candidate is `7f114e94661c3a39ca46d888322f438ecd3b0830`. Same-round fixed short/8K/32K decode regresses8.54/7.49/8.08%, with identical128-token outputs. Diagnostic pilot5 official assertions pass5/5 on both arms; main/candidate worker decode is67.8897/62.5338tok/s with15/15 and17/17 complete request clocks. No credible benefit was obtained; formal20 is not run and the candidate is not recommended for adoption. The original kernel remains default. Independent evidence and limitations are in [results PR #94](https://github.com/LittleLittleCloud/Infer.NET/pull/94).

A final test-harness correction resets output before graph-helper dispatch and checks graph output sentinels. The equivalent driver passed320 comparisons with Metal shader validation. This post-measurement change affects tests/documentation only; it does not relabel the measured runtime SHA or change the product kernels.
