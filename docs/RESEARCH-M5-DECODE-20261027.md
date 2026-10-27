# M5 Pro decode research: priorities after profiling and negative experiments

2026-10-27, America/Los_Angeles. Research only; no implementation or new inference measurement. Codex performed this investigation. The user requires subsequent implementation through **GitHub Copilot CLI with GPT-6.1 Sol**. Copilot was not invoked for this report, and no provider/model execution is claimed.

Tracking: [research Issue #101](https://github.com/LittleLittleCloud/Infer.NET/issues/101). Notebook entry: `m5-decode-research-20261027` (available on Pages only after review, merge and deployment).

Audited main: `4f51f241e87999c8cc3f79d733cd95ade9c711cd`, also the remote main at inspection. Local hardware reports Apple M5 Pro, 20 GPU cores and 48 GB memory; macOS 26.6.2, SDK 27.0. The report concerns the existing Qwen3.6-35B-A3B affine8/group64 model and native C#/Metal backend. It does not generalize measurements to other models.

## Decision

Prioritize ordinary target decoding in two workload classes: quantized linear operations for short contexts, and attention for long contexts. Investigate memory layout and useful work per load before attempting another lane-count-only expert optimization. Keep Q8 KV as a separate correctness/quality project. Defer another complete speculative-decoding integration until its measured cost per committed token beats scalar decoding.

This is a priority recommendation, not a measured improvement. There is no supported promise of 100/150 token/s or a twofold gain. The published homepage contains adopted historical records; recent unmerged result PRs materially change the research picture.

## Evidence snapshot

The following result PRs were open and unmerged when inspected. Their observations inform research but are not adopted product capabilities. Links pin result commits, so subsequent branch changes do not silently change this report's evidence.

| Evidence | Observation | Implication |
| --- | --- | --- |
| [Profile #70 / results #91](https://github.com/LittleLittleCloud/Infer.NET/blob/09699e89a225a42aa7f52dfb9e010ce0953a0b95/docs/EXPERIMENT-70-DECODE-PROFILE-RESULTS-20261025.md) | At short / 8K / 32K context, MoE is 40.0 / 33.6 / 24.9%; dense projections 37.9 / 31.8 / 23.9%; LM head 13.6 / 11.3 / 8.4%; attention 3.5 / 19.1 / 39.6% of sampled GPU regions. | Optimize different regions by context. These shares do not identify ALU versus bandwidth bottlenecks. |
| [Expert kernels #92 / results #94](https://github.com/LittleLittleCloud/Infer.NET/pull/94) at `70af5fd27d587bdcbaa3636b8f8a67e7cd913a58` | Smaller cooperating lane groups, with unchanged weight layout, regressed fixed tiers by 7.49–8.54%; pilot decode 67.890 to 62.534 token/s. | Changing lane grouping alone is already a negative experiment. |
| [Target block #95 / results #97](https://github.com/LittleLittleCloud/Infer.NET/pull/97) at `3f1990f9c62d70afe2b1ed567c7e73cadb64c02a` | Eight-row verification/commit transaction wall time improved, but one-row verification regressed; default-off formal application gates failed. | A faster transaction is not an emitted-token speedup or application acceptance. |
| [DFlash8 #98 / results #100](https://github.com/LittleLittleCloud/Infer.NET/blob/757c8126078feb13ee536e661ff7bc34dbbd157d/docs/EXPERIMENT-98-DFLASH8-E2E-20261026.md) | Verified-power 8K/4096: main 60.363, native DFlash8 28.702 token/s. 32K/4096: main 45.577, native 17.263. Draft numerical qualification fails. | Native verification improved the scalar DFlash prototype but still loses to ordinary decoding. |
| [Q8 #33 / results #87](https://github.com/LittleLittleCloud/Infer.NET/pull/87) at `1eb8564274b944126392109007b34b1954a4734e` | Codec/selected checks passed; full-model gate failed, 78/121 comparisons passed, final logits maxAbs 0.02856922. | There is no qualified Q8 speed result. Resolve same-semantics correctness before throughput/adoption. |

#70 used supported encoder-boundary timestamps because this M5 did not support dispatch-boundary sampling. The instrumentation changes scheduling. Region sums cover 91.49–94.31% of sampled command durations, and samples do not cover every generated interval. MoE includes shared/routed experts, routing, merge and normalization. Delta recurrence alone is only 1.3–2.1%, but DeltaNet projections are in the dense category. None of this proves physical hardware utilization.

#98's qualified power rerun identifies a 140W supply and UI-verified High Power. Earlier 35W observations remain separate. Main/native formal HumanEval20 throughput is withheld because Pi/worker token reconciliation failed, even though worker clocks exist. Do not substitute candidate-off's 67.558 token/s, a successful-only subset, or Pi wall time. Numerical, transport, timing and restoration failures remain in the source report.

## Existing code rules out several generic suggestions

At the audited main, [gpu_graph.mm](https://github.com/LittleLittleCloud/Infer.NET/blob/4f51f241e87999c8cc3f79d733cd95ade9c711cd/native/DeepInfer.Metal/gpu_graph.mm) already submits one command buffer per token, performs GPU routing and argmax, keeps recurrent/KV state on GPU, and fuses expert gate/up/SwiGLU and residual/normalization operations. It reads the chosen token after completion. Recommending these as new work would duplicate existing capabilities.

[quantized.mm](https://github.com/LittleLittleCloud/Infer.NET/blob/4f51f241e87999c8cc3f79d733cd95ade9c711cd/native/DeepInfer.Metal/quantized.mm) already has compile-time bit-width specialization and a vector path extracting eight adjacent codes. It is not simply a generic scalar unpack loop. Current GEMV dispatch mostly distinguishes tiny output counts and a common row tile; the very large LM head and dense projections share this kernel family.

[gpu_prefill.mm](https://github.com/LittleLittleCloud/Infer.NET/blob/4f51f241e87999c8cc3f79d733cd95ade9c711cd/native/DeepInfer.Metal/gpu_prefill.mm) implements the current FP32 decode-attention path with 128-key partitions, two KV groups and a separate stable combine. KV reuse across eight query heads already exists. The next experiment must improve this existing implementation, not claim to introduce GQA reuse or fused attention for the first time.

## Ranked ideas and implementation boundaries

### 1. Shape-aware affine8 linear operations

This is the highest-priority direction for short-context coding/chat. Dense plus head accounts for approximately 51.5% of the short-context sampled regions, and the larger MoE region is also dominated by linear work, although its substage split remains unmeasured.

Proposed first bounded experiment: use the dense projections and LM head as a controlled study of packed-weight layout and input-vector reuse. Start with the LM head (`K=2048`, `N=248320`) because it avoids expert routing and provides a large fixed geometry. Compare the existing packed row-major path with one explicitly specified tile-interleaved affine8 layout. Preserve every quantized code, BF16 scale/bias value and FP32 operator semantics; validate a complete pack/unpack round trip. Coordinate the lane mapping with the new layout, rather than repeating #92's smaller-lane scheme on unchanged rows. Measure input reuse, register pressure, cache behavior and traffic where hardware tools support them.

Do not simply expand the model to FP16/FP32, keep a permanent second model copy, or transplant Splash's BF16 arithmetic. Account for prefill consumers, tied/shared weight storage if present, loading transients and disabled-mode memory before proposing the layout. If dual layout storage is required, include it in the budget and stop on a failed capacity preflight. A cold repack is outside steady decode time but must be reported separately. Freeze at most two candidate layouts before microbenchmark selection, then freeze one before full-model validation.

Only extend a successful layout to dense projections or experts in separately scoped follow-ups. An LM-head-only improvement has limited reach: even halving its sampled 13.6% cost implies about 7.3% total speedup in an idealized Amdahl calculation. This is sensitivity arithmetic, not a forecast. A generic layout rewrite without real traffic/occupancy evidence could regress, as prior larger-row and expanded-weight experiments did.

### 2. Context-dependent FP32 attention geometry

For 32K and longer workloads this can be a better first implementation than the linear study. Preserve existing FP32 KV, exact causal bounds, grouped-head reuse and stable softmax. Compare bounded partition sizes such as 64/128/256 keys and corresponding threadgroup geometry, with the current 128-key path as the explicit control. Select based on real shape costs, not just dispatch count. Larger partitions reduce intermediate/combine work but can increase registers, shared memory or reduce occupancy; smaller partitions reverse that trade-off.

Use context-boundary fixtures, non-multiple tails, short contexts and 4096-token continuations. Changing the partition changes floating-point reduction order, so this is not automatically bit-exact. Preserve the approved operator tolerances and full-model divergence analysis. Freeze a deterministic context selector on tuning fixtures before held-out evaluation. Do not bundle Q8 storage or speculative verification into this experiment.

### 3. Repair Q8 correctness, then measure long-context bandwidth and capacity

Continue existing Issue #33 rather than opening another Q8 implementation. The documented format uses 1088 bytes per full-attention layer/token for K+V, versus 4096 bytes with FP32: 73.4375% less logical KV storage. This is not the same percentage reduction in total token time or physical DRAM traffic.

First isolate the reported layer-15 V byte difference with identical source activations and rounding/scale semantics; distinguish quantization-boundary propagation from an implementation mismatch. Keep a same-Q8 reference separate from FP32-versus-Q8 quality comparison. Do not waive the recorded full-model failure. Pending policy PR #89 is not merged and does not implicitly approve a new numerical/quality budget. Once qualified, run the fixed long-context quality/capacity cases and paired performance controls. Weight, Delta and convolution precision remain unchanged.

### 4. Resume speculation only after a cost-model gate

Measure the same clock for each complete cycle: draft + target verify + commit/rollback + ingest + synchronization + orchestration. Divide the full generation duration by committed output intervals; separately report mean committed tokens/cycle and acceptance by proposal depth. High acceptance alone is insufficient, and costs may overlap, so do not blindly sum overlapping host timers.

An earlier, explicitly power-unqualified #98 trace had 1395 cycles, 9765 proposals and 2694 accepted proposals: about 27.6% proposal acceptance and about 2.94 emitted tokens/cycle for that 4096-output request. Its verification, draft, ingest and commit timers were 117.970/22.108/1.359/1.055 seconds, with zero target replay. This identifies questions for fresh profiling, not a qualified-power attribution. Direct commit has already been implemented in that candidate; proposing it again does not address the remaining cost.

Before another complete integration, obtain a fresh matched trace showing whether draft cost, excess verification rows or acceptance depth dominates. Then separately test bounded adaptive verification width or genuinely faster multi-row target arithmetic. Require positive ordinary-main E2E benefit, including draft costs and tails, and resolve numerical qualification. Do not optimize only relative to the slower scalar-DFlash prototype.

### 5. Treat TensorOps as a capability and arithmetic study

Apple distinguishes compute-heavy matrix work from single-row bandwidth-sensitive decode. GPU Neural Accelerators are not the separate Neural Engine, and their presence does not guarantee an M=1 GEMV speedup. They are more promising for genuinely reusable multi-row work; eight selected experts are different weight matrices, not eight rows sharing one matrix.

The [WWDC26 TensorOps session](https://developer.apple.com/videos/play/wwdc2026/330/) documents integer tensor support added in macOS 26 updates, and additional scale-plane/cooperative-input capabilities in macOS 27. This machine's 27.0 SDK is not runtime availability on macOS 26.6.2. Check exact API/type support and deployment targets before implementation. Do not assume affine8 group64 with BF16 scales and additive biases maps directly to MX block scaling or integer dot products. Any custom dequantization, BF16 multiplication or scale/bias factoring must disclose its arithmetic changes and separately qualify them.

Apple sources: [M5 compute/bandwidth explanation](https://developer.apple.com/videos/play/tech-talks/111432/), [TensorOps quantized formats and custom operations](https://developer.apple.com/videos/play/wwdc2026/330/), [M5 Pro hardware specifications](https://support.apple.com/en-us/126319). The advertised 307 GB/s is peak hardware bandwidth, not measured Infer.NET bandwidth or an attained roofline. No hardware counter capture was performed in this research.

## Copilot CLI handoff and evaluation contract

Implementation remains pending a separately approved experiment. Choose the linear-layout study for short-context speed, or the FP32 attention study for long-context speed; do not combine them into one causal A/B.

The implementation brief must name the approved Issue and plan revision, fresh baseline SHA, exact native entry points above, one optimization hypothesis, correctness fixtures and predeclared adoption criteria. A proposed engineering gate is at least 5% aggregate worker decode improvement on the selected primary workload, no fixed-tier prefill/decode regression beyond 5%, passing correctness/resources, and reproducibility across counterbalanced rounds. The 5% improvement is a proposal for review, not an inherited requirement or observed gain. The separate 128K targets remain authoritative.

Copilot execution requirements:

1. Verify the installed GitHub Copilot CLI, authentication, model catalog and provider-reported model identity. Use the user's GPT-6.1 Sol model only; do not silently select a similarly named model or fall back to another provider. Record the exact accepted model identifier, CLI version, prompt hash, session ID and actual model usage. A valid command/model spelling is not invented here. The current shell did not resolve `copilot`; that is not proof it is absent from all other installations.
2. Codex supplies research, constraints and review. Copilot CLI authors the candidate implementation and fixes. Preserve authorship/provenance; this report does not attribute any existing code to a new Copilot run. No nested agents are needed.
3. Start a fresh `codex/experiment-<issue>-<name>` branch from audited main, not an unmerged candidate. Default-off control, actual native selector metadata, bounded allocation and no silent fallback are required. Keep code and independent results PRs separate; no merge or service/default change is implied.
4. Follow [Experiment SOP](EXPERIMENT-SOP.md) and [fixed targets](QWEN36-8BIT-TARGETS.md). Use fixed HumanEval5 for iteration, freeze the candidate, then fixed20 with official assertions and negative control. Also use identical token-ID inputs at short/8K/32K with 128 outputs and 8K/32K with 4096 outputs, warmup plus at least three formal serial trials, counterbalanced order. Long-capacity tests require preflight; HumanEval does not establish 128K acceptance.
5. Compare the same model revision, arithmetic, KV, cache, power, prefill profile and request limits. Preserve slow/failed rounds, per-request worker timing/token coverage, first-valid-output TTFT, exact package/model/input hashes and service restoration evidence. Profile-on runs remain separate from throughput. Compute decode as sum(N-1)/sum(decode seconds), not mean request rates or Pi wall time.
6. Record supplied power and actual mode, memory pressure, load/warmup versus steady-request swap, allocation high-water and physical-residency evidence or its absence. Keep the 38.5 GiB allocation and 512 MiB steady-request swap limits; do not raise system limits or stop unrelated jobs. A numerical/resource failure is a result, not permission to relax the criteria.

No new benchmark, implementation, measured chart point or adoption is part of this report. The notebook entry links to these findings without changing existing trend values or the adopted baseline.
