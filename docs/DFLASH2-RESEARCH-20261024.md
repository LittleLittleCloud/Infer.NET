# DFlash2 integration: speculation-first research

> Session closeout, 2026-10-25: the user authorized preserving research/results and closing experimental code PRs #74/#78/#81 without adoption. Earlier approval/pending/publication statements below describe the historical stage, not a current request to continue. The later #80 diagnostic20 includes the frozen MTP reference. See the [final disposition and combined summary](SPECULATIVE-DECODE-SESSION-CLOSEOUT-20261025.md). No acceptance threshold or original measurement was changed.

2026-10-24. The user approved diagnostic #70, then directed the work toward speculative decoding. #70 retains that approval but is deferred; its original plan explicitly excludes speculation. [Speculation-specific experiment #72](https://github.com/LittleLittleCloud/Infer.NET/issues/72) is a concrete new proposal awaiting approval. No implementation or performance run has started.

GitHub Copilot CLI1.0.103, actual usage-confirmed `claude-sonnet-5.5`, completed a focused follow-up source study in session `49d9cd16-5288-477b-b7ac-c869fba154cf`. This is the reviewed synthesis, supplemented by independently fetched draft metadata. [Evidence manifest](evaluations/dflash-research-20261024/manifest.json). The original report/session remain local with hashes. Resumed-session usage is cumulative and is not the cost of this follow-up alone.

Audited main: `4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3`. Splash: `1482f6bbc2abf12375332725302019594150b6cb`.

## Recommendation

Make speculative decoding the optimization direction. Establish exact proposal/commit/rejection state semantics first, then integrate a real DFlash2 draft and batched target verification. Generic graph profiling is no longer the immediate implementation priority.

A serial verifier with synthetic proposals is a correctness reference, not DFlash2 inference or an acceleration. The eventual speed hypothesis is that a draft plus batched verification amortizes target work over several accepted tokens. It remains unmeasured on Infer.NET. Splash's 72.5% acceptance was measured with its affine8 target, Q8 KV and native draft implementation; it is not an Infer.NET acceptance prediction or an isolated explanation for the observed speed difference.

## Draft asset now pinned

[Checkpoint configuration](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2/blob/51ef7b6923ad6c14cb1bb41c37a9041446496ab6/config.json) and immutable file metadata were fetched, without downloading weights.

| Field | Verified metadata |
|---|---|
| Repository / revision | `incoai/Qwen3.6-35B-A3B-DFlash2` / `51ef7b6923ad6c14cb1bb41c37a9041446496ab6` |
| Architecture / source dtype | `DFlash2DraftModel` / BF16 |
| Parameters | 526,251,520, per repository metadata |
| Layers / hidden / vocabulary | 6 / 2048 / 248320 |
| Attention heads / KV heads / head dimension | 32 / 8 / 128 |
| Context window / proposal block | 2048 / 8 |
| Target layer IDs | `[1,6,11,16,22,27,32,37]` |
| Selector rank / top-k | 256 / 16 |
| Mask / EOS token IDs | 248077 / 248044 |
| Weight file bytes | 1,052,513,456 |
| Remote LFS SHA256 | `d1f16c0ff85afbe6166fabfae998af3ec16fa11da87da4391b5e0e0a7c796a4d` |

Public copies: [config](evaluations/dflash-research-20261024/draft-config.json), [file provenance](evaluations/dflash-research-20261024/draft-file-provenance.json). Remote hash metadata is not local weight verification. Tensor names/shapes, tap meaning, tokenizer compatibility and a numerical draft oracle remain gates before a real loader/kernel implementation.

Splash repacks draft projection tensors to AF4G64 while retaining selected BF16 tensors. That is an upstream implementation choice, not an inherent DFlash2 requirement. A future Infer.NET draft precision must be declared separately from the unchanged affine8 target precision. [Draft checkpoint preparation](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/DraftCheckpoint.cpp#L28).

## Integration dependencies

1. **Hidden taps and draft context.** The draft consumes target intermediate layer outputs, not only final logits. Infer.NET has diagnostic per-layer traces, but they copy all-layer data to the CPU. Add only the required device taps after independently matching layer numbering, post-residual versus normalized semantics, dtype and target embedding/head conventions. Prompt context must populate the draft window too. Split prefill does not currently transfer the required front-half hidden taps, so the first real-draft scope should be fully local Metal. [Draft layout](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/DFlashDraft.hpp#L71), [existing target trace](https://github.com/LittleLittleCloud/Infer.NET/blob/4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3/native/DeepInfer.Metal/gpu_graph.mm#L50).
2. **Proposal and verification.** DFlash2 drafts a block using its noncausal masked-query architecture and selector; it is not an ordinary autoregressive small-model loop. Greedy target verification must match the anchor/proposal row offset, accept a contiguous matching prefix, and choose the correct correction/bonus token. Bound all work by remaining output/context capacity. Start with a fixed block/verification policy; adaptive cost budgets would add another experimental variable. [Draft graph](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/model/Runtime.cpp#L1869).
3. **All-row target predictions.** Existing prefill processes multiple tokens but applies the classifier only to the last row. A real batched verifier needs each relevant row's target logits/argmax, correct causal masks and positions. Existing prefill tiling and MoE grouping can change floating-point reduction order relative to sequential decode. Exact sequential token agreement must be tested, not assumed. [Prefill path](https://github.com/LittleLittleCloud/Infer.NET/blob/4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3/native/DeepInfer.Metal/gpu_prefill.mm#L634).
4. **State transactions.** Rejection must recover full-attention KV visibility, every DeltaNet recurrent matrix, convolution history, graph position, draft rings/context and pending token identity. Existing prefix-state import only accepts layers0–19 during split handoff; it cannot rewind the full decode graph. `Reset` clears the request rather than restoring a committed prefix. [Current state API](https://github.com/LittleLittleCloud/Infer.NET/blob/4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3/native/DeepInfer.Metal/gpu_prefill.mm#L795), [graph state allocation](https://github.com/LittleLittleCloud/Infer.NET/blob/4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3/native/DeepInfer.Metal/gpu_graph.mm#L376).
5. **Efficient recovery.** A correctness reference can snapshot recurrent/conv buffers, rewind append-only KV visibility and replay only committed inputs. Those recurrent/conv buffers total65,863,680bytes (about62.81MiB) from current source shapes, excluding KV, draft and transaction metadata. This is an allocation estimate, not measured copy cost. Splash uses a GDN tape/retained-row design; porting that idea needs its own numerical validation. [Upstream GDN](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/decode/gdn.metal).
6. **Serving semantics.** Commit at the existing loop's consumed-input boundary, which can differ from the last emitted output position. EOS, output caps, cancellation and reset must never retain rejected state. Emit only verified tokens. Keep cycle-completion/emission-lifecycle clocks distinct from generation clocks: block delivery does not justify inventing per-token N−1 timestamps. Standard decode remains unavailable if equivalent timing is absent. [Serving loop](https://github.com/LittleLittleCloud/Infer.NET/blob/4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3/src/DeepInfer.Host/SplitService.cs#L259).
7. **Residency budget.** The BF16 checkpoint file is about1.05GB; runtime draft activations, shared target heads, ring KV, taps, verifier scratch and recovery storage must be measured too. Neither file size nor historical target allocation proves a fit. The fixed38.5GiB Metal and512MiB per-request swap-out limits remain; no sysctl increase, offload or precision change is authorized. Historical128K allocations suggest tight headroom but do not prove a universal impossibility. No128K claim follows from short-context speculation.

## First experiment: #72

The complete plan is in [Issue #72](https://github.com/LittleLittleCloud/Infer.NET/issues/72). Scope is deliberately part of speculative decoding: default-off proposal acceptance, all-layer state snapshot/restore, serial verification/replay, rejection patterns and correct committed-state continuation. No generic kernel optimization is bundled in.

Test every rejection offset and all/none acceptance for bounded1–8-token proposal blocks, repeated rollback, EOS/caps, cancellation/reset and context boundaries. Compare saved-token fixed128/4096-output fixtures, logits and state at committed boundaries against ordinary greedy decode. Run fixed HumanEval5 iteration and20 formal application/assertion comparisons. Disabled mode must preserve timing coverage and allocations, prefill≤5% regression and decode≤5% regression; the separately proposed TTFT regression limit is5%. Enabled serial-reference speed is diagnostic only. Preserve resource failures and mismatches.

After approval: fresh main experiment branch, codePR, evaluation and independent resultsPR, with human review and no automatic merge. The next real-draft/batched-verifier proposal must fix tap semantics, draft precision, correctness policy and adoption threshold before running. Approval of the correctness harness does not silently authorize those later changes.

The existing [128K goals](QWEN36-8BIT-TARGETS.md) and [experiment SOP](EXPERIMENT-SOP.md) remain authoritative. No target or quality threshold is relaxed. No speculative percentages are plotted as measured speed.
