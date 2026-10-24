# Decode optimization research — Copilot / Claude Sonnet 5.5

2026-10-24. Requested researcher: GitHub Copilot CLI 1.0.103, actual usage-confirmed model `claude-sonnet-5.5`, session `49d9cd16-5288-477b-b7ac-c869fba154cf`. This is Codex's reviewed synthesis of the completed Chinese source-research report. [Evidence manifest](evaluations/decode-research-20261024/manifest.json). [First proposed experiment #70](https://github.com/LittleLittleCloud/Infer.NET/issues/70).

Research baseline: main `4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3`. Splash source was independently fetched from its verified Git remote and checked out at measured revision `1482f6bbc2abf12375332725302019594150b6cb`. No implementation, GPU benchmark, service change or system-setting change was performed. No new throughput point is reported. The original Copilot report and usage/session evidence remain in the ignored local research directory with hashes in the manifest; they are not public downloads.

## What the existing results establish

| Historical run | Decode tok/s | Clock and configuration |
|---|---:|---|
| Infer.NET main cd6f7da, 92 requests | 65.085 | Worker N−1 intervals, FP32 KV, split prefill / local decode |
| Infer.NET #58, 107 requests | 66.434 | Worker N−1 intervals, FP32 KV, entirely local Metal |
| Splash #61, 107 requests | 148.914 | Native executor-command counters, Q8 KV, DFlash2, full resident target/draft, 40GiB budget |

Sources: [main record](MAIN-RUN-CD6F7DA-20261023.md), [local record](EXPERIMENT-58-M5PRO-HUMAN20-20261024.md), [Splash record](EXPERIMENT-61-SPLASH-HUMAN20-20261024.md), [raw public counter summary](evaluations/issue61/performance-resident40g.json). Both local application cohorts passed 20/20 official assertions. The local Infer.NET run also recorded 2.013GiB whole-cohort swap-out: residency and memory pressure remain relevant confounders, even though that count is not per-request attribution.

The user's approximately threefold gap is the motivation. These particular historical headline values differ by about 2.24 arithmetically, but **do not establish a comparable speedup ratio**: clocks, KV precision, speculation, actual code, memory budget and generated request histories differ. Current main has not been remeasured in this research.

The Splash JSON contains 24,648 drafted and 17,868 accepted draft tokens, approximately 72.5% acceptance. Summing its request speculation counters gives 4,839 cycles and about 4.71 output tokens/cycle. These are observations of Splash's run, not predictions of Infer.NET acceptance or speed. No non-speculative Splash control exists in these records. DFlash2 is a strong candidate explanation to investigate, not an experimentally isolated cause.

## Source findings

All links below pin the audited source revisions.

- Infer.NET already uses one command buffer per decode token, GPU argmax, then a completion wait and host token read. Recommending those features as new would duplicate existing work. [Forward loop](https://github.com/LittleLittleCloud/Infer.NET/blob/4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3/native/DeepInfer.Metal/gpu_graph.mm#L394), [completion](https://github.com/LittleLittleCloud/Infer.NET/blob/4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3/native/DeepInfer.Metal/quantized.mm#L215).
- The graph contains many small norm/router/merge kernels and buffer barriers. Dense quantized GEMV, fused expert gate/up and selected-expert down run in the same layer chain. Dispatch count depends on attention policy and context; the documented legacy 675/685 counts must not be assigned unchanged to fused attention. Dividing whole-GPU duration by dispatch count does not measure launch overhead. [Graph helpers and router](https://github.com/LittleLittleCloud/Infer.NET/blob/4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3/native/DeepInfer.Metal/gpu_graph.mm#L268).
- Existing GPU timestamp profiling samples only position 32 and inserts counter barriers. HumanEval prompts normally start decode beyond that position, so representative per-stage evidence needs additional instrumentation and an explicit perturbation check. [Sampling gate](https://github.com/LittleLittleCloud/Infer.NET/blob/4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3/native/DeepInfer.Metal/gpu_graph.mm#L257).
- The quantized dot path performs FP32 dequantization and accumulation; expert gate/up is already fused. Vector quantization already exists; larger row tiles have prior negative evidence. Algebraic regrouping of scale/bias terms or MMA changes reduction/rounding and cannot be called bit-preserving. [Quantized kernels](https://github.com/LittleLittleCloud/Infer.NET/blob/4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3/native/DeepInfer.Metal/quantized.mm#L84), [prior row-tile study](QWEN36-OWN-BENCHMARK.md), [rejected expert pipeline](EXPERT-FLOW.md).
- Explicitly record `decode-attention=fused-fp32` in both arms. CLI defaults and deployed profiles can differ. The prior attention experiment demonstrated context sensitivity, so short HumanEval gains cannot substitute for 32K/128K measurements. [Attention evidence](DECODE-ATTENTION.md), [policy selection](https://github.com/LittleLittleCloud/Infer.NET/blob/4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3/src/DeepInfer.Host/SplitInference.cs#L180).
- Splash encodes draft and target verification graphs, selects verification rows, groups MoE rows by expert, and adapts verification work to acceptance/cost. These are algorithm and scheduling differences as well as kernel differences. [Draft/verify graph](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/Runtime.cpp#L1869), [verification budget](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/VerifyBudget.hpp), [MoE](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/ops/MoE.hpp).
- Splash's specialized GDN and quantized decode kernels merit separate comparison after profiling. Upstream kernel comments and microbenchmarks are evidence about upstream workloads, not Infer.NET speedup measurements. [GDN](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/decode/gdn.metal), [decode matrix kernel](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/decode/linear_gguf_sgmatrix.metal).

## Ranked experiments

Every row is a proposed experiment, not an implemented improvement. No percentage is promised without measurement.

| Priority | Narrow question / planned change | Expected benefit and risks |
|---|---|---|
| A0 — first, #70 | Default-off multi-position decode timestamps and separate encode/commit/wait/GPU/emission timing | Diagnostic value; no intended speedup. Quantify counter/barrier perturbation in separate runs. |
| A1 — after A0 and separate approval | One proven small-kernel fusion or redundant barrier removal | Benefit to be measured. Preserve arithmetic order and dependencies; GPU races and reduction changes fail validation. |
| B | One affine8 GEMV or DeltaNet implementation change chosen by measured dominant cost | Benefit to be measured. Separate reformulated GEMV from GDN so each has one variable; independent FP32 oracle and first-divergence analysis required. |
| C — existing #33 | Native Metal/CUDA Q8 KV storage, attention reads and byte-preserving handoff | Particularly relevant to long context and memory; short HumanEval benefit unknown. This already has a tracking Issue and fixed format; do not duplicate it. |
| D — separate proposal | DFlash2-style draft + batched target verification, with exact state rollback and explicit verification policy | Potentially material but unmeasured. Requires target/draft residency budget, draft revision, KV/Delta/conv checkpoints, rejection rollback, acceptance distribution and deterministic/quality policy. |

Host/GPU gaps should be measured before proposing asynchronous submission. Removing a per-token host wait may interact with encoding, scheduling and cancellation; the historical wall−GPU difference is neither a precise CPU-overhead measurement nor a strict speedup bound.

For Q8, [Issue #33](https://github.com/LittleLittleCloud/Infer.NET/issues/33) was independently read after Copilot's run: both native backends and transfer/import must use the fixed block32 layout; FP32 state remains unchanged; protocol tests alone are insufficient. [Targets](QWEN36-8BIT-TARGETS.md) remain authoritative. Copying Splash's Q8 layout without checking compatibility is inappropriate.

DFlash2 must remain a separately reported workload. It cannot redefine the fixed 128K acceptance configuration or silently change its no-MTP/no-speculative-prefill conditions. Speculation does not inherently require changing target weight precision; numerical differences, adaptive verify geometry and quality/repeatability still need explicit controls. Splash's `verify-rows=auto` can produce varying greedy MoE output; upstream describes repeatability under `all` only with the stated isolation/cache conditions. [Upstream sampling conditions](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/DEVELOPMENT.md#L1111).

## First experiment and SOP boundary

[Issue #70](https://github.com/LittleLittleCloud/Infer.NET/issues/70) contains the concrete **diagnostic-only** plan, fixed baseline, pinned dataset/image/model, explicit configuration, metrics, correctness/resource gates, service restoration, evidence paths and proposed adoption criteria. Approval has not been obtained. No optimization stage is bundled into that approval request.

After human approve, follow [EXPERIMENT-SOP.md](EXPERIMENT-SOP.md): fresh main-based experiment branch, code PR, evaluation, independent results PR, human review, then adoption decision. Use fixed HumanEval5 for iteration and fixed20 for formal comparison; aggregate Σinput/Σprefill-seconds and Σ(N−1)/Σdecode-seconds, preserve official assertions, first-valid-output TTFT, coverage, resources and failures. Profile-on measurements remain separate from clean throughput. Failures also receive a results record. The fixed 128K workload still requires its own full acceptance.

The research PR only records research and the pending experiment-list entry. It is neither #70's implementation PR nor a measured-results PR. No expected value is added to a chart. [Prefill chart](https://littlelittlecloud.github.io/Infer.NET/#prefill-trend), [decode chart](https://littlelittlecloud.github.io/Infer.NET/#decode-trend).

## Review corrections and remaining uncertainty

The raw Copilot report contained useful source investigation but also unsupported extrapolations. This synthesis deliberately corrects them:

- The source-derived byte/roofline estimate omitted important cache, scale-storage, reuse and achieved-bandwidth effects. It cannot establish a 94 tok/s ceiling, a 10–20% optimization limit, or that matching Splash requires speculation. No hardware bandwidth benchmark was run.
- The historical wall/GPU averages were exploratory local-log aggregates, not a controlled same-main cohort. They cannot establish an asynchronous speedup ceiling or isolate CPU encoding cost.
- No ablation proves speculation is the principal cause, that Q8 helps HumanEval by at most 1.5%, or that FP32 KV necessarily fails the 37 tok/s gate. Historical FP32 long-context results and the full target specification must remain visible.
- Whole-cohort swap in the #58 record prevents dismissing residency. Recheck steady per-request swap, pressure, GPU residency and power on the actual candidate.
- The draft's suggested TTFT +5% rule was not an existing target-document rule. #70 separately states its proposed diagnostic acceptance criteria. No new target threshold or approval is inferred from the model's prose.
- Current main runtime performance, real per-stage GPU costs, profiling perturbation, effective bandwidth, same-configuration Splash controls, and Infer.NET draft acceptance remain unmeasured.

The next decision should follow #70's controlled measurements, not the report's speculative percentages.
