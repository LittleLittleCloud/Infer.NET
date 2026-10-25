# Research: integrating Qwen3.6 official MTP weights into Infer.NET (decode speed)

> Session closeout, 2026-10-25: the user authorized preserving research/results and closing experimental code PRs #74/#78/#81 without adoption. Earlier approval/pending/publication statements below describe the historical stage, not a current request to continue. The later #80 diagnostic20 includes the frozen MTP reference. See the [final disposition and combined summary](SPECULATIVE-DECODE-SESSION-CLOSEOUT-20261025.md). No acceptance threshold or original measurement was changed.

- Status: **research only, pending approval (independently reviewed revision 3).** No MTP implementation or benchmark is approved. Nothing here is a measurement of Infer.NET MTP behaviour.
- Proposed experiment ID: `official-mtp-20261024` (Section 10 summarizes the dedicated experiment proposal).
- Research baseline: main `4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3`. The experiment itself pins the latest main at Issue creation.
- Author: GitHub Copilot CLI (Claude Sonnet 5.5), external researcher. External fetches on 2026-10-24. Revision 2 corrects revision 1 after independent review; the original report is kept by the parent as `initial-report.md`.
- Out of scope: DFlash, Windows RTX 4090, 128K acceptance, sampling, product implementation, inference runs and deployment.
- Constraints honoured: no inference, no weights loaded, no complete shard downloaded, no GPU/service/SSH activity. Remote safetensors headers were read with bounded `Range` requests (all HTTP 206; an HTTP 200 would have been aborted).

**Evidence labels used throughout.** *[Metadata]* = read from a pinned primary source (HF API/config/index/header, pinned upstream code). *[Derived]* = arithmetic or reasoning from metadata, not a measurement. *[Unknown]* = not verified; never estimated as measured. *[Prior S0]* = unmerged prior evidence (Issue #72 / PR74, commits `afdb36fd…` code, `d13476ae…` results), cited only as design context; it did not pass all service acceptance and is not assumed adopted.

## 0. Summary of the revised proposal

1. One trained MTP block (full attention + MoE), shared embedding/LM head, 19 BF16 tensors, 1,689,281,536 B *[Metadata]*. Our affine8 target has none.
2. **Draft cache reconditioning is required every cycle**, not only at prompt bootstrap: speculative steps after the first use draft-generated hidden states, so their K/V differs from K/V computed from verified target hidden states even when the token is accepted (Section 4, traced in pinned vLLM).
3. Convention (Section 5): initial verified anchor emitted once; thereafter the cycle anchor is already emitted but unconsumed; a cycle commits `[anchor,d1..dj]` and emits `[d1..dj, g_j]`; `k ≤ remaining−1`; invariant consumed position = `P + N_emitted − 1` (length) and `P + N` (natural EOS).
4. Timing (Section 7): a real generation-equivalent clock with simultaneous-token timestamps; standard aggregate `SUM(N−1)/SUM(T_last−T_first)`. If it cannot be audited: speed null, no adoption.
5. Single reviewable first trial (Section 8): BF16-preserving matrices, FP32 norm scales (raw BF16 + 1, no second BF16 rounding), shared affine8 target embedding/head, FP32 activations/accumulation/state and KV, native BF16 operators, k ∈ {1,2} only, single-local Mac, context ≤ 32768, output ≤ 4096.
6. Weight authenticity (Section 1.6): download the two official pinned BF16 shards (6,064,305,104 B), verify each full LFS SHA256, extract only the 19 MTP tensors. Range-only payload extraction is an explicitly *unauthenticated alternative*, not the default.
7. Concrete resource/performance gates (Section 10.6). Speedup is **unmeasured**.

## 1. Official artifact manifest

### 1.1 Pinned revisions *[Metadata]*

| Repo | Immutable revision | Notes |
| --- | --- | --- |
| [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/995ad96eacd98c81ed38be0c5b274b04031597b0) (official BF16) | `995ad96eacd98c81ed38be0c5b274b04031597b0` | 26 shards; index `total_size` 71,903,645,408 |
| [Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/tree/95a723d08a9490559dae23d0cff1d9466213d989) (official FP8) | `95a723d08a9490559dae23d0cff1d9466213d989` | block-FP8; separate `mtp.safetensors` |
| [mlx-community/Qwen3.6-35B-A3B-8bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit/tree/e06a74e6236a60c8367e1a3214e83d8b61b637b0) (our target) | `e06a74e6236a60c8367e1a3214e83d8b61b637b0` | 8 shards; `total_size` 37,721,128,672; zero `mtp*` index keys |

Key files (full URLs at the pinned revision):
- BF16 [config.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json), [model.safetensors.index.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/model.safetensors.index.json), [README.md](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/README.md), shards [25](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/model-00025-of-00026.safetensors) and [26](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/model-00026-of-00026.safetensors).
- FP8 [mtp.safetensors](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8/blob/95a723d08a9490559dae23d0cff1d9466213d989/mtp.safetensors).
- Target [config.json](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit/blob/e06a74e6236a60c8367e1a3214e83d8b61b637b0/config.json), [model.safetensors.index.json](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit/blob/e06a74e6236a60c8367e1a3214e83d8b61b637b0/model.safetensors.index.json).

Full LFS SHA256 values from the HF tree API *[Metadata]*:

| File | Size (B) | LFS SHA256 |
| --- | ---: | --- |
| BF16 `model-00025-of-00026.safetensors` | 3,832,888,256 | `778e7f76602f05042b69ba7f3ec91f1fdffef390540b16074041c258fb81d154` |
| BF16 `model-00026-of-00026.safetensors` | 2,231,416,848 | `1a97404220077ed3d4182e10385b152004cab608377f50cec9f54a6b8d28b613` |
| FP8 `mtp.safetensors` | 853,860,608 | `3e9c9fef90a5c02314cf27eeccc72f9f61ce711c12c869244c7127c787b5bd7d` |

Retained local evidence hashes (sha256, prefix; the proposed public manifest will carry full values and is prepared by the parent): config `93a4693f…`, BF16 index `41b93561…`, shard-26 header `6f2afede…`, shard-25 header `a1213b03…`, FP8 config `570ef7ea…`, FP8 mtp header `4100a8e3…`, target config `5da49c3a…`, target index `3db12ede…`, README `c4ddaa06…`. The parent's independent `parent-verification/` metadata corroborates the revisions and both header byte counts.

### 1.2 Starting facts verification

| Claim | Status | Basis |
| --- | --- | --- |
| Target index has zero `mtp.*` tensors | Verified *[Metadata]* | target index |
| Official BF16 index lists 19 `mtp.*` tensors in shards 25/26 | Verified *[Metadata]* | index plus both shard headers (HTTP 206, `Range: bytes=0-7` then the JSON header range) |
| FP8 sibling has separate `mtp.safetensors` ≈854 MB | Verified *[Metadata]* | tree API size 853,860,608; header_len 192,120; 1,560 tensors |
| `mtp_num_hidden_layers=1`, `mtp_use_dedicated_embeddings=false` | Verified *[Metadata]* in all three configs | configs (the MLX config keeps the keys although weights are absent) |
| Tokenizer compatibility | Verified by the parent and re-checked here *[Metadata]*: vocab IDs equal, normalized merges equal, all 26 common added-token IDs (248044–248069) equal; the target adds 7 audio/TTS tokens (248070–248076); files are **not byte-identical** (sha256 official `5f9e4d49…`, target `87a7830d…`) | tokenizer.json at both revisions |

Decision: the experiment keeps the **target's existing tokenizer**; it does not silently switch to the latest official tokenizer. *[Unknown]*: chat-template equality was not hashed.

### 1.3 Official BF16 MTP tensors (19 tensors, all BF16, 1,689,281,536 B) *[Metadata]*

Byte counts are `numel × 2`, cross-checked with header `data_offsets`. Names carry the `mtp.` prefix.

| Tensor | Shape | Bytes | Shard |
| --- | --- | ---: | --- |
| `mtp.fc.weight` | [2048, 4096] | 16,777,216 | 26 |
| `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`, `mtp.norm.weight` | [2048] each | 4,096 each | 26 |
| `mtp.layers.0.input_layernorm.weight`, `…post_attention_layernorm.weight` | [2048] each | 4,096 each | 26 |
| `mtp.layers.0.self_attn.q_proj.weight` | [8192, 2048] (query + output gate) | 33,554,432 | 26 |
| `…self_attn.k_proj.weight`, `…v_proj.weight` | [512, 2048] each | 2,097,152 each | 26 |
| `…self_attn.o_proj.weight` | [2048, 4096] | 16,777,216 | 26 |
| `…self_attn.q_norm.weight`, `…k_norm.weight` | [256] each | 512 each | 26 |
| `…mlp.gate.weight` (router) | [256, 2048] | 1,048,576 | 26 |
| `…mlp.shared_expert.{gate,up}_proj.weight` | [512, 2048] each | 2,097,152 each | 26 |
| `…mlp.shared_expert.down_proj.weight` | [2048, 512] | 2,097,152 | 26 |
| `…mlp.shared_expert_gate.weight` | [1, 2048] | 4,096 | 26 |
| `…mlp.experts.gate_up_proj` (fused) | [256, 1024, 2048] | 1,073,741,824 | **25** |
| `…mlp.experts.down_proj` | [256, 2048, 512] | 536,870,912 | 26 |

Experts ≈95 % of bytes. No `mtp.embed_tokens`/`mtp.lm_head`: the draft uses the target's (`tie_word_embeddings=false`, so embedding and head are separate tensors in the target). The MTP layer is full attention + sparse MoE (16 Q heads, 2 KV heads, head_dim 256, partial rotary 0.25, rope_theta 1e7, mRoPE interleaved sections [11,11,10], top-8 of 256 experts plus a gated shared expert).

Norm elements: 5 × 2,048 (`pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `mtp.norm`, `input_layernorm`, `post_attention_layernorm`) + 2 × 256 (`q_norm`, `k_norm`) = **10,752** *[Derived]*. Total elements 844,640,768; non-norm elements 844,630,016 *[Derived]*.

### 1.4 Official FP8 sibling *[Metadata]*

The FP8 sibling **is an official artifact**. Experts are per-expert `F8_E4M3` with BF16 `weight_scale_inv` (128×128 blocks; e.g. [512,2048] → scale [4,16]); `mtp.fc`, norms, `mlp.gate`, `shared_expert_gate` stay BF16; no embedding/head. It is a lossy block-quantized derivative of BF16, so it **cannot serve as a BF16-preserving source**; it is useful as a structure/naming cross-check and as a smaller official file. Using it needs an F8 decoder that Infer.NET does not have (`SafeTensors.cs` has no F8 path).

### 1.5 Format comparison and reuse

| Property | Official BF16 | Official FP8 | Our target affine8/g64 | Reuse |
| --- | --- | --- | --- | --- |
| Weights | BF16 | F8_E4M3 + BF16 128×128 scales | U32-packed 8-bit + BF16 scales + BF16 biases, group 64 per matrix (`mlp.gate`, `shared_expert_gate` also 8-bit) | none directly |
| Norm convention | zero-centred, output `x̂·(1+w)` | same | baked `w+1` (mlx-lm `sanitize` adds 1.0) | **convert**: derive scale = raw BF16 + 1 **in FP32, no second BF16 rounding**, for all 7 norm kinds including `q_norm`/`k_norm` (`Qwen3NextRMSNorm` aliases `GemmaRMSNorm` in pinned vLLM; this correct finding is preserved) |
| Experts | fused `gate_up_proj [E,2I,H]` (gate = first I rows, up = last I rows) + `down_proj [E,H,I]` | per expert | per-matrix stacked tensors (`Qwen36Weights.Matrix`) | split gate/up (layout conversion only, values unchanged) |
| Embedding/head | shared with target | shared | separate affine8 `QuantMatrix`, ≈540.3 MB each (weight 508,559,360 B + scales 15,892,480 B + biases 15,892,480 B) *[Derived]* | **reuse directly**, zero extra copy |

### 1.6 Precision options *[Derived]* sizes, none measured at runtime

Affine8/g64 matrix storage is 1 B/element + 4 B per 64 elements (BF16 scale + BF16 bias) = **1.0625 B/element = 0.53125 × BF16** (revision 1 wrongly used 0.5625).

| Option | Bytes before padding/container *[Derived]* | Status |
| --- | ---: | --- |
| **A. BF16 matrices + 10,752 FP32 norm elements** | 844,630,016 × 2 + 10,752 × 4 = **1,689,303,040** | **Proposed first trial** |
| B. Explicit draft-only affine8/g64 for all non-norm elements, norms FP32 | 844,630,016 × 1.0625 + 43,008 = **897,462,400** | Not approved; changes the draft; separate experiment |
| C. FP8 sibling file | 853,860,608 (file) | Not a BF16-preserving source; not proposed |

The target is never quantized or re-quantized. Option A needs native BF16 operators (BF16 GEMV/grouped-MoE kernels on Metal); there is **no FP16/FP32 matrix expansion** (that would double resident bytes and is a different experiment) and **no draft quantization**. Resident bytes can differ from file bytes (padding, alignment); fit is a measured gate, never inferred from file size.

Weight authenticity: HTTP Range reads verify **layout** (header, offsets) only, not payload against an official digest. The proposed extraction (approved step, not performed now) downloads the **two official pinned BF16 shards** (3,832,888,256 + 2,231,416,848 = **6,064,305,104 B**), verifies each full file against the LFS SHA256 above, then extracts only the 19 MTP tensors, recording sha256 of each tensor, the sidecar and the converter, and may delete only the experiment-owned staging copies after success. No full base-model download. A Range-only payload extraction can't claim full-shard authenticity (self-generated tensor hashes prove nothing about the official file) and is only an *unauthenticated alternative*.

## 2. Official MTP semantics (pinned sources)

Sources (immutable):
- vLLM [`4e591657157a021cab997f911d67f50c0d18a71e`](https://github.com/vllm-project/vllm/tree/4e591657157a021cab997f911d67f50c0d18a71e): [`qwen3_5_mtp.py`](https://github.com/vllm-project/vllm/blob/4e591657157a021cab997f911d67f50c0d18a71e/vllm/model_executor/models/qwen3_5_mtp.py), [`qwen3_5.py`](https://github.com/vllm-project/vllm/blob/4e591657157a021cab997f911d67f50c0d18a71e/vllm/model_executor/models/qwen3_5.py), [`qwen3_next.py`](https://github.com/vllm-project/vllm/blob/4e591657157a021cab997f911d67f50c0d18a71e/vllm/model_executor/models/qwen3_next.py), [`llm_base_proposer.py`](https://github.com/vllm-project/vllm/blob/4e591657157a021cab997f911d67f50c0d18a71e/vllm/v1/spec_decode/llm_base_proposer.py).
- SGLang [`c5e734e437facd61e1b454919851600541d9e672`](https://github.com/sgl-project/sglang/blob/c5e734e437facd61e1b454919851600541d9e672/python/sglang/srt/models/qwen3_5_mtp.py).
- Transformers [`50e35cc0ca18ec19b80653690de553e6decf74f5`](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py).
- mlx-lm [`4cfc6478473b0668ede3eb0cc93f2a9181a15052`](https://github.com/ml-explore/mlx-lm/blob/4cfc6478473b0668ede3eb0cc93f2a9181a15052/mlx_lm/models/qwen3_5.py) and `qwen3_5_moe.py`.
- Authored model card (the pinned README above): "MTP: trained with multi-steps"; SGLang `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`; vLLM `num_speculative_tokens: 2`. These are GPU serving defaults, not Apple-silicon evidence. Local copies and sha256: `artifacts/dual-host/mtp-research-20261024/upstream/`.

One draft step *[Metadata]* (vLLM `Qwen3_5MultiTokenPredictor.forward`; SGLang agrees):

```
e = pre_fc_norm_embedding(embed(token_in))          # shared target embedding; embedding FIRST in the concat
h = pre_fc_norm_hidden(hidden_in)
x = fc(concat[e, h])                                # [2048,4096]
x = FullAttnMoEDecoderLayer(x, position, draft KV)  # input_layernorm → gated attn (q_norm/k_norm, partial RoPE) → residual → post_attention_layernorm → top-8 MoE + gated shared expert → residual
y = mtp.norm(x)                                     # same y feeds the shared head AND the next draft step's hidden_in
draft_token = argmax(target_lm_head(y))
```

| Question | Answer | Source |
| --- | --- | --- |
| Target hidden tap | post-final-norm hidden (`Qwen3NextModel.forward` ends with `self.norm(...)`; no aux taps for this method) | `qwen3_next.py` |
| Pairing | row at position *p* = `(H_p, x_{p+1})`: target hidden at *p* with the token that FOLLOWS (shift by one; last slot takes the newly sampled token) | `set_inputs_first_pass` in `llm_base_proposer.py` |
| Position | draft row for hidden `H_p` sits at position *p*; chained draft steps advance by 1 | proposer |
| Layer reuse | `spec_step_idx % num_mtp_layers`; with 1 layer the same weights serve every step | `qwen3_5_mtp.py` |
| Draft KV | the MTP layer has its **own** KV, independent of the target's 10 attention layers | proposer |
| Rejected rows | `prepare_inputs` computes `num_rejected = num_draft + 1 − len(sampled)` and shrinks query length/seq_len so rejected rows are not fed or visible | `prepare_inputs` |
| Shared weights | embedding and LM head shared; `is_unused_checkpoint_weight` keeps only `mtp.*`, `embed_tokens`, `lm_head` | proposer, `qwen3_5_mtp.py` |
| Fused experts | `gate_up_proj[..., :I, :]` gate, `[..., I:, :]` up | Transformers `chunk(2)`; mlx-lm `sanitize` |
| HF / mlx-lm as oracle | neither implements an MTP forward (`_keys_to_ignore_on_load_unexpected=["^mtp.*"]`; mlx-lm filters `mtp.`); not oracles | sources |

*[Unknown]*: numeric equality of target MLX norms with official+1 (tensor bytes not read); acceptance behaviour on Metal; SGLang first-pass details beyond what matches vLLM.

## 3. Infer.NET integration map (main `4ca29eed…`) *[Metadata: source read]*

| Area | File / symbol | Fact |
| --- | --- | --- |
| Weights | `src/DeepInfer.Backends/Models/Qwen36Weights.cs` | geometry/quant hard-checks; `Matrix(prefix, rows, cols, experts, bits)`; separate Embedding/Classifier `QuantMatrix`; **no MTP loading** |
| Safetensors | `Models/SafeTensors.cs` | `Get/ReadBFloat16/ReadInto/ReadFloat`; no F8 |
| Managed graph | `Compute/Qwen36Gpu.cs` | `Prefill` loops `prefillChunk`; `Forward` single token; `Suffix`/`PrefixState`/`PrefixKvRange` = split prefix import/export (**not to be used or copied**); `ConfigureFusedFp32Decode`, `Reserve/Grow/Stats`; no rollback, no hidden accessor |
| Session | `Split/SplitSessions.cs` `MetalSplitSession` | `Reset`, `EnsureCapacity`, `Forward`, `Prefill`, `Stats`; ctor takes `kvBytes`, `prefillScratchTokens`, `fusedFp32Decode` |
| Request loop | `Host/SplitService.cs` `RunMac` | per token: cancellation check, `gpu.IsEnd`, append, `first`/`last` clocks, SSE `token`, `EnsureCapacity`, `Forward`; `InferenceTimings.Create(prefillMs, first, last-first, count)`; per-token `DECODE` stderr log |
| Native decode | `native/DeepInfer.Metal/gpu_graph.mm` | `GModel`: `position`, 40 layers (30 DeltaNet conv 3×8192 FP32 + state 32×128×128 FP32; 10 attention K/V), `x` (pre-norm residual), `norm` (final-normed hidden), argmax buffers; `di_graph_forward` returns only token/logits/trace; **no ABI exposes `x`/`norm`** |
| Native prefill | `gpu_prefill.mm` `di_graph_prefill_range` | count ≤ `maxTokens`; advances position and DeltaNet/conv in place for all rows; classifier on **last row only**; optional `output` = pre-norm `x` for all rows; no row-wise argmax. Kernel selection (`p_gemv`, `p_delta`): matrix kernel at rows ≥128, bf16-nax at ≥64 tokens, grouped experts and chunked delta at ≥64 tokens, else sequential `PDelta`; so a 2–3 row verify uses a different kernel mix than both decode and 32-row prefill |
| Reference | `Models/Qwen36Model.cs` | C# single-token engine (not the Mac service path); norm/attention/MoE/RoPE math reusable as a diagnostic oracle (CPU is never the product backend) |

Current baseline *[Issue #58 record, `docs/EXPERIMENT-58-M5PRO-HUMAN20-20261024.md`]*: fused-fp32 decode attention, FP32 KV, managed prefill chunk 32 (`DEEPINFER_METAL_PREFILL_CHUNK=32`), scratch 128, `--short-prefill-tokens 262144`, affine8 target, Pi 1.4.0 HumanEval-20 seed 0: prefill 268.205 tok/s, decode 66.434 tok/s, client TTFT median 10,914.875 ms, response-sampled Metal allocation 34.730 GiB (continuous peak unmeasured), whole-machine swap-out 2.013 GiB. These are a prior record, **not** the comparison baseline: the experiment measures a new paired main (Section 10). Main *source* and the running worker's build are distinct objects, each hashed.

New work (all default-off): `Qwen36MtpWeights` loader + converter; native hidden export (post-final-norm rows after decode/verify); native draft graph (`gpu_mtp.mm`) with own FP32 draft KV; multi-row verify entry with row-wise argmax and per-row post-norm hidden; snapshot/restore; default-off generation-loop branch; options/telemetry. S0's `graph_snapshot.mm` ownership/epoch/request-id checks (`afdb36fd…`, *[Prior S0]*) may be **minimally and explicitly re-used inside a new main-based code PR** as reviewed scope; the code PR must not be stacked on PR74, and S0 is not validated acceleration. The split prefix-import API must not be used to bypass all-layer state ownership.

## 4. Draft cache reconditioning (corrected finding)

Traced in pinned `llm_base_proposer.py`: **every** `propose()` call (every generation cycle) first runs `set_inputs_first_pass` over the **whole verified target block**: `input_ids[:n-1] = target_token_ids[1:]`, last slot = newly sampled token, `hidden_states[:n] = target_hidden_states`, positions unchanged; rejected rows are removed (`prepare_inputs`, `num_rejected_tokens`). The draft layer therefore rewrites its K/V for the accepted target rows from **target hidden states** each cycle, then samples from the last row and runs the remaining speculative steps with its own (draft-generated) hidden states. Steps after the first write draft-conditioned K/V at later positions. That K/V is **not** the K/V obtained from verified target hidden states even when the draft token equals the accepted token, so it must not be kept on accept-all; initial prompt bootstrap alone is insufficient.

Define two independent cursors and a provenance rule:
- `C` = target cursor: target positions `0..C-1` consumed. Anchor `a` (emitted, unconsumed) is the token at position `C`.
- Draft cache row `p` is the pair `(H_p, x_{p+1})`, **target-conditioned** only if `H_p` is a verified target hidden. `D` = draft cursor: rows `0..D-1` are valid and target-conditioned. Draft-generated rows (written from draft hidden `y`) are **provisional** and are never counted in `D`.
- Seed hidden provenance: `H_{C-1}` is the post-final-norm hidden of the target row at position `C-1` (the row that predicted `a`). It is stored with its target position and the target epoch; a seed from a rejected/rolled-back row is invalid.
- Cycle-start invariant: `D = C` (row `C-1 = (H_{C-1}, a)` written, so draft proposal `d1` is available), no provisional rows live.

Cycle (width k): `d1` is available from the initial row or preceding catch-up; for `k=2`, `d2` from a provisional row at position `C` with input `(y_{C-1}, d1)`. Verification (Section 5) accepts `j`, emits correction/bonus `g_j`. **Catch-up pass (only if another speculative cycle follows):** overwrite draft rows `C … C+j` with target-conditioned pairs `(H_{C+i}, x_{C+i+1})` where `x = [d1,…,d_j, g_j]`; this is one `(j+1)`-row draft pass, and its last row gives the next `d1`. Provisional rows (positions beyond what the catch-up rewrote) are invalidated by cursor, not read. New cursors: `C' = C+j+1`, `D' = C'`.

Worked check (user example, k=1, accept-all, `P` prompt tokens, `C=P`): first pair `(H_{P-1}, t0)` at `P-1` predicts `d1`. Verify consumes `t0` at `P`, `d1` at `P+1`, predicts `g1`. Catch-up writes `(H_P, d1)` at `P` and `(H_{P+1}, g1)` at `P+1` (2 rows); row `P` did not exist before (k=1 has no provisional row), so just seeding the last anchor would leave a hole. For k=2 accept-all the provisional row at `P` (input `(y_{P-1}, d1)`) is **overwritten** by `(H_P, d1)`.

Cache tests (required, oracle mode): after every cycle compare draft rows `0..D-1` with a fresh full target-conditioned recompute over the committed sequence (accept-all k=1 and k=2, partial accept, reject at offset 0, EOS inside a block, length cap); poison provisional/rejected draft rows with NaN and show they are never read; negative control showing a "keep draft-generated K/V" variant differs from the recompute (diagnostic only, never a product path); check `D`, `C` and seed provenance after every cycle. Generated-only draft (no prompt rows) and other cache policies are separate unapproved alternatives.

Bootstrap: chunkwise during prefill, export post-norm hidden of each prefill chunk (≤32 rows) and run the draft over rows `p ≤ P-2` with shifted prompt tokens (`x_{p+1}` is known from the prompt, the chunk-boundary row uses the next chunk's first token). Row `P-1` needs the anchor `a0` and is the first draft step of cycle 1. All bootstrap cost is **in prefill/TTFT**, not hidden in warmup.

## 5. Consumed / emitted / pending semantics (single convention)

Notation: `P` prompt tokens; `N` = tokens emitted so far; `N_max` = `maxTokens`; `remaining = N_max − N`; `C` = target consumed positions.

- After prefill+bootstrap the target has consumed `P` tokens; `g` = greedy argmax of the last prompt row = the **initial verified anchor**. If it is EOS (and not `BenchmarkIgnoreEos`), finish with no tokens (as baseline). Otherwise **emit it once** (`N=1`, `C=P`). Invariant: `C = P + N − 1`.
- Cycle (anchor `a` already emitted, unconsumed, `remaining ≥ 1`): choose `k = min(K, remaining − 1)`. If `k = 0` (`remaining = 1`): ordinary autoregressive step (no drafting): consume `a`, emit the predicted token. Else: verify inputs `[a, d1..dk]` (`k+1` rows); `g_i` = target argmax after row `i`; `j` = number of leading accepted proposals (`d_i == g_{i-1}`).
- Commit consumed `[a, d1..dj]` (`C += j+1`), emit `[d1..dj, g_j]` (`j+1` tokens, `N += j+1`); `g_j` (correction if `j<k`, bonus if `j=k`) is **unconsumed** and becomes the next anchor. Invariant `C = P + N − 1` holds after every cycle. Since `j+1 ≤ k+1 ≤ remaining`, nothing is emitted past `N_max`.
- **Length stop:** when `N = N_max` the last emitted token is the unconsumed anchor: `C = P + N − 1`.
- **Natural EOS:** at the first `m` with `g_m = EOS` (scan `m = 0..j`; a proposal `d_{m+1} = EOS` equal to `g_m` is the EOS itself, never emitted): commit rows `0..m` (`C += m+1`), emit `[d1..dm]`, stop; EOS is pending, unemitted (unless `BenchmarkIgnoreEos`). Then `C = P + N` (every emitted token consumed, EOS pending), matching the baseline's natural-EOS boundary.
- Reject / EOS / length truncation requires **exact** state truncation: attention KV positions reset to `C`; DeltaNet conv/state restored from the pre-verify snapshot and replayed over the `m+1` (or `j+1`) committed rows when a rollback happened; draft side per Section 4; seed hidden = verified post-norm hidden of the last committed row. Checks compare target state digest, last-row logits and seed hidden with the baseline at the same position — not just the output array.

Examples (`P=10`):
1. **k=1 accept-all, length.** `N_max=5`. `N=1, C=10`. Cycle 1 (`remaining=4`, `k=1`): verify `[a0,d1]`, `d1=g0`, `j=1`; commit 2 rows (`C=12`), emit `d1,g1` (`N=3`): `12 = 10+3−1` ✓. Cycle 2 (`remaining=2`, `k=1`): `j=1`, `C=14`, `N=5`; stop: `C = 14 = P+N−1` ✓ (`g` unconsumed).
2. **Cap/reject, k=2.** `N=1,C=10`; `d1≠g0` → `j=0`: commit `[a0]` (`C=11`), emit `[g0]` (`N=2`) ✓; rows for `d1,d2` rejected: KV position reset to 11, DeltaNet restore + replay of 1 row. Later at `remaining=1`: `k=0`, plain step: consume anchor (`C=P+N−1+1`), emit one token → `N=N_max`, `C = P+N−1` ✓ (it consumed the previous anchor, the new one is unconsumed).
3. **Natural EOS, k=2.** `N=1,C=10`; `g0=d1`, `g1=EOS`: `m=1`; commit `[a0,d1]` (`C=12`), emit `[d1]` (`N=2`), EOS pending: `C = 12 = P+N` ✓; third row rolled back (restore + replay 2 rows).

Alternative (not chosen): keep the old convention with a final-pending emission tail and `k ≤ remaining−2`. It is more complex at the tail; the chosen convention is complete on its own.

Cancellation/timeout: checked per cycle before draft, before verify and after rollback; on cancel the target and draft are abandoned and the next `Reset` reinitializes both; snapshot storage is released or reused. S0's two stream-finish failures (HumanEval/149 off, HumanEval/94 on) are unexplained *[Prior S0]*, are generic stream failures, and need predeclared handling (Section 10.5).

## 6. Target verification: the speed-critical component

- A **serial verifier** (k+1 single-token `Forward` calls) is an oracle only; it does strictly more work than baseline and is never an acceleration implementation.
- **True multi-row verification** runs `k+1` rows per layer. Weight reuse across rows is the *design intent*, **not guaranteed**: actual kernels may re-read rows/experts (grouped-expert tiling, per-row attention, tile-size fallbacks). MoE touches up to `min(256, 8(k+1))` distinct experts *[Derived upper bound]*.
- Reuse of native prefill requires: row-wise argmax over all rows (new), per-row final norm/hidden export, behaviour at 2–3 rows (sequential `PDelta` below 64 tokens, no grouped experts below 64) *[Metadata: source]*; none timed *[Unknown]*.
- Numerical order: batched GEMM/grouped experts vs single-token GEMV, chunked vs sequential delta recurrence, prefill vs fused-fp32 decode attention. Whether any argmax flips is *[Unknown]* and is the decisive gate.
- **Greedy exactness:** candidate-on output must equal the candidate-off (single-token) continuation token for token on the same runtime. If numeric tolerances pass but any argmax differs, unchanged-output correctness **fails**. Fix decode-equivalent numerics within the approved scope or stop. A different output-equivalence claim requires a new proposal and approval; the criterion cannot be relaxed after results.
- Deterministic state checks after each cycle: 40-layer digests (attention over rows `[0,C)` only, DeltaNet/conv bitwise), last-row logits, seed hidden; draft rows per Section 4; finiteness of active/read hidden, logits, scratch and snapshot regions. Intentionally poisoned invisible suffixes are excluded and must remain unread. Draft math may use declared tolerances (the draft cannot change output) but non-finite is a hard failure.

## 7. Timing clock and metric

Real generation-equivalent clock, from worker clocks (monotonic), per request:
- `T_first` = time the **initial verified anchor** is ready for emission, *after* prefill and the chunkwise bootstrap (so bootstrap is inside prefill/TTFT).
- `T_last` = time the final committed output token is released, after **all** costs needed to release it: draft, verify, acceptance/readback, rollback/replay, catch-up and any state commit required before that release. Cleanup after the last release (snapshot release, etc.) is recorded separately and in whole-request wall but not in `T_last`.
- Tokens validated together legitimately share one ready timestamp; **no invented intra-block spacing** and no network/stream block timestamps.
- Standard decode = `SUM(N_i − 1) / SUM(T_last_i − T_first_i)` over all requests (aggregate, **not** a median of per-request throughputs, not Pi wall).
- Cost placement: the first cycle's row-`P-1` draft pair and `d1` belong to the decode interval (conservative; bootstrap through `P-2` belongs to TTFT). Prefill and TTFT gates catch any cost shifted out of the interval.
- If this clock cannot be implemented and audited, speed is null and nothing is adopted. S0's `timings=null` was an instrumentation gap of that experiment; it does not mean a block decoder can never have a valid standard clock.

## 8. Cost model and bottlenecks (M5 Pro 20-core GPU / 48 GiB, target affine8)

All items *[Derived]* or *[Unknown]*; no Splash/DFlash/CUDA speed or acceptance is carried over.

Startup pays the full-prompt bootstrap in prefill/TTFT and the first anchor-conditioned draft row once in decode. In steady state, the preceding reconditioning pass already produced `d1`; do not charge another first-row draft pass. Per cycle: `T_cycle = (k−1)·C_draft_step + C_verify(k+1 rows) + C_snapshot + [C_restore + C_replay(j+1 rows) if j<k] + C_catchup(j+1 draft rows, head on last row) + C_overhead`, with emitted `E = j+1`. Catch-up is needed only when generation continues into another speculative cycle. Skip future-proposal/cache work at EOS, the output cap or a final AR-only tail. Aggregate speed-up is `SUM(E)·C_decode / SUM(T_cycle)`, including startup decode costs, rather than an average of cycle speed ratios. Acceptance and all costs are unmeasured.

The first cycle reaches `D=C` only after its initial anchor-conditioned row; bootstrap alone ends at `D=P−1`. Continuing cycles re-establish `D=C` and produce the next `d1` together. Terminal checks validate the existing committed draft prefix and invalidation boundary; they do not require computing unused future-anchor rows.

Hypothesized bottlenecks (rank untested):
1. Verifier at 2–3 rows on prefill-path kernels (MoE union, attention kernel, sequential delta).
2. Shared head reads: the ≈540.3 MB affine8 head is applied once per draft head evaluation (startup once, then `k−1` provisional heads plus one continuing-cycle catch-up head) plus once for `k+1` verify rows (read-once not guaranteed).
3. Draft layer bytes touched per row ≈129 MB BF16 (fc 16.8 + q 33.6 + k/v 4.2 + o 16.8 + router 1.0 + shared expert 6.3 + top-8 experts 8×6.29 = 50.3 MB) *[Derived]*; BF16 native kernels are new code.
4. Rollback: S0 *[Prior S0, unmerged, not re-verified]* reported CPU-host snapshot capture/restore diagnostics of about 1 ms each. The 62.8 MiB state (30 × (2,097,152 + 98,304) = 65,863,680 B) *[Derived]* implies a roofline of well under 1 ms, but that roofline is **not** an expected cost; use the S0 diagnostic caveat until measured, and the replay of `j+1` rows is the larger term.
5. Command-buffer/CPU round trips between draft steps and verify (k per cycle unless the chain stays on GPU) *[Unknown]*.
6. Memory: components below; fit is a measured gate.

| Component | Estimate *[Derived]* | Shared/duplicated |
| --- | --- | --- |
| Draft weights (A) | 1,689,303,040 B ≈ 1.57 GiB before padding | new |
| Embedding/head | 0 extra | shared |
| Draft KV (FP32) | 1 layer × K+V × 2 heads × 256 × 4 B = 4,096 B/token (16 MiB at 4,096 ctx; 128 MiB at 32,768) | new, independent |
| Target KV (FP32) | 40,960 B/token, unchanged | unchanged |
| Recurrence snapshots | 62.8 MiB × live snapshots (1–2) | new, only when enabled |
| Verify/draft workspace | [Unknown] | new or shared scratch |
| Candidate-off | exactly no added GPU buffers | gate |

## 9. Oracle and approval requirements

- **Draft oracle:** CPU/C# (reusing `Qwen36Model` math) or Python/PyTorch implementation of the draft step and the cache recompute on small fixed inputs with the same converted weights; independent oracle only, never a product backend. Compare `fc` output, layer output, `mtp.norm` output, argmax and top-8 router indices on target hidden vectors exported by the baseline runtime.
- **Conversion oracle:** full-shard SHA256 verification, per-tensor sha256, shape checks (§1.3), FP32 `+1` norm check, gate/up split roundtrip.
- **Approval needed before:** the two-shard download (6,064,305,104 B), any model run (hidden export needs a baseline run), any benchmark, deployment of any candidate worker, and any change to the experiment's fixed choices.

## 10. Proposed experiment

**Status: PROPOSED, pending approval. Do not start.** Canonical fixed proposal: [Issue #76](https://github.com/LittleLittleCloud/Infer.NET/issues/76).

- **ID:** `official-mtp-20261024`. **Title:** Qwen3.6 official MTP (BF16-preserving draft, shared affine8 head) greedy speculative decode on single-local M5 Pro.
- **Homepage entry points:** `#prefill-trend`, `#decode-trend`; a point is plotted only from approved evidence and only if the Section 7 clock is available; no placeholder, zero or interpolated points.
- **Local artifacts:** `artifacts/dual-host/issue<N>-official-mtp/<run-id>/`; **public evidence:** `docs/evaluations/issue<N>/` (private data redacted; no credentials, private paths, benchmark prompts or solutions). **Independent** code PR (new, main-based) then results PR; merge only with explicit user authorization.

### 10.1 Hypothesis
With unchanged affine8 target and unchanged greedy output, an official-MTP draft with k ∈ {1,2}, per-cycle draft reconditioning and true multi-row verification raises aggregate standard decode over a **new paired main** run. *[Unknown]*; the trial may falsify exactness, acceptance, net cost or memory fit.

### 10.2 Fixed first-trial design (not left as questions)
- Baseline: latest main at Issue creation (full 40-hex), built once; paired arms **main**, **candidate-off**, **candidate-on(k)** from the same session, same hardware, serial GPU scheduling.
- Target `mlx-community/Qwen3.6-35B-A3B-8bit@e06a74e6…`, unchanged, target tokenizer kept. Draft: official BF16 `Qwen/Qwen3.6-35B-A3B@995ad96e…` shards 25/26 `mtp.*`; BF16 matrices; norm scales FP32 = raw BF16 + 1 (no second BF16 rounding); shared affine8 target embedding/head; FP32 activations, accumulation, state, target KV and draft KV; native BF16 operators; no FP16/FP32 matrix expansion; no draft quantization.
- Weights: two-shard download with full LFS SHA256 verification, extraction of 19 tensors, hashed sidecar/converter; only owned staging copies may be deleted after success; no full base-model download.
- Width: **k ∈ {1,2} only**. Single-local Mac (`--short-prefill-tokens 262144`), fused-fp32, FP32 KV, chunk 32, scratch 128, context ≤ 32,768, output ≤ 4,096. Greedy only.
- Draft cache: full-prompt chunkwise bootstrap plus per-cycle reconditioning (Section 4). Generated-only drafts are separate unapproved alternatives.
- Snapshot ownership: minimal, reviewed reuse of S0 graph-snapshot ownership/epoch checks inside the **new** code PR (not stacked on PR74); no split prefix-import API.
- Out of scope: Windows (no measurement, no modification), DFlash, 128K (formal 128K targets currently disable MTP; this is a separate experimental configuration, no 128K compliance claim, `docs/QWEN36-8BIT-TARGETS.md` untouched), sampling, restricted-vocab head, draft quantization, deployment adoption.
- Files (code PR): new weights loader/converter, native draft/verify/snapshot units, narrow hooks in `Qwen36Gpu.cs`, `SplitSessions.cs`, `SplitService.cs`, `Options.cs`, tests, short doc.

### 10.3 Implementation gates (stop at first failing gate; report)
- **G0** converter/oracle (no GPU): shard SHA256, tensor hashes, shapes, FP32 norm derivation, gate/up split; CPU oracle on synthetic inputs.
- **G1** default-off proof: exactly zero added GPU buffers (allocator stats), identical kernels/responses; CPU tests.
- **G2** draft numerics vs oracle (declared tolerances, finite) on baseline-exported hidden vectors; draft-cache recompute tests (Section 4).
- **G3** verifier exactness (decisive): multi-row vs serial decode for hidden, logits, **argmax**, state digests on natural-EOS and fixed-128/4096-token fixtures (warm 1 / formal 3), k=1 and 2. Any argmax mismatch fails unchanged-output correctness → stop.
- **G4** end-to-end token equality vs candidate-off: fixtures, natural EOS, length stops (`remaining ≤ k+1`, `remaining=1`), cancellation/timeout recovery, repeated rejections, `BenchmarkIgnoreEos`.
- **G5** Pi 1.4.0 pilot (5 requests, seed 0), k=1 and k=2, with official assertions and the negative control.
- **G6** formal 20 (seed 0) at the pre-frozen width; original service restored afterwards.

HumanEval dataset revision `b9453e8f1865df0a0057a1383767caa400acfbcf`, compressed SHA256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`; assertion image `python@sha256:5fb4bc53dd0180955e31e5245b648bbb24a4ac4ab03a1de81513642325220bcd`; per-task timeout 300 s. Fixed task selection uses the existing benchmark selector with seed 0 and sizes 5/20; persist the ordered IDs before runs.

### 10.4 Pilot selection (fixed before formal)
Among widths passing G3–G5 hard gates (exactness, finiteness, resources, no unresolved failures), choose the one with the higher pilot aggregate standard decode; ties → k=1. The chosen width is frozen before formal 20; both pilots and all fixtures are preserved. If no width passes, publish negative results. Once formal 20 starts, **all planned attempts are retained** unless unsafe; never drop failures or selectively retry.

### 10.5 Measurements, replay control and failures
- Per request: complete worker clocks (Section 7), client TTFT (coverage for every request), prefill/decode split, token reconciliation (requested/emitted/consumed/accepted/rejected, failures included), finish reason; acceptance per position, mean accepted length, cycles, rollback count/cost, and separate draft/verify/catch-up/snapshot/restore/replay times.
- "Same selected requests" means the same **task set**; Pi tool-call histories need not match between arms. Paired captured HTTP request bodies/token IDs are a **separate deterministic replay control** (private/redacted), not a claim of identical live Pi trajectories.
- Hashes: source, native library, package, running process, HF pins, converter, tensors. Residency/per-submission safety recorded.
- Failures: stream-finish-class failures are counted and retained, investigated with paired captured requests, never retried selectively.
- Negative control: official assertions must fail on deliberately corrupted output.

### 10.6 Concrete gates (engineering criteria, not measured data)
Resources (every formal request): Metal allocator high-water ≤ **38.5 GiB** across all tracked target + MTP + verify + snapshot buffers (continuous allocator high-water, not only post-request samples); whole-system swap-out ≤ **512 MiB** per formal request; AC power, High Power mode, default wired limit (0 override); no OOM/crash, no active NaN/Inf, no CPU weight offload; process footprint and memory pressure recorded.

Candidate-off vs paired main: prefill latency ≤ +5 %, decode ≥ −5 %, TTFT ≤ +5 %, no extra GPU buffers; all timing and TTFT coverage complete.

Candidate-on adoption (proposed margin): aggregate standard decode ≥ **1.20×** the new paired main (also reported vs candidate-off), enabled prefill and TTFT regression ≤ 5 %, no unresolved failures, and all three formal arms main/off/on each complete all 20 Pi tasks without errors and pass official 20/20 assertions. The margin is an engineering payoff threshold, not a statement about measured noise. If the clock is not auditable → no speed claim, no adoption.

### 10.7 Staged stop conditions
Stop and record a negative result if: G3 shows an unremovable argmax mismatch; acceptance is below break-even (Section 8); any resource gate fails or non-finite value appears; cache recompute tests fail; or an unexplained stream termination appears in candidate-off. No k=3, draft quantization, Windows, restricted head, 128K or generated-only variants without a new approval. A negative or null result is valid and published unchanged.

## 11. Unresolved questions
- Numeric equality of target MLX norms and official+1 *[Unknown]*.
- Multi-row vs decode argmax agreement on real prompts *[Unknown]* (G3).
- Acceptance on Pi/HumanEval with an affine8 target and BF16 draft (draft was trained against the BF16 target) *[Unknown]*.
- Cost of native BF16 kernels, 2–3-row prefill-path kernels, catch-up pass and round trips *[Unknown]*.
- Whether per-row DeltaNet state capture in the multi-row kernel can replace snapshot+replay *[Unknown]*.
- Chat-template parity between official and target packages *[Unknown]*.
- S0's stream-finish failures and its TTFT/allocator-peak gaps *[Prior S0]* are unresolved and must be closed for this experiment's own runs.

## Appendix A. Evidence index
- Local pinned evidence: `artifacts/dual-host/mtp-research-20261024/raw/` (API/tree/config/index/headers/README) and `…/upstream/` (pinned source); parent corroboration under `parent-verification/`.
- Prior S0 (unmerged): code `afdb36fd1106cf29b19b66ac94745dda481fc429`, results `d13476ae9238493122db25614dcd2677f19373fc` (not relied upon).

Public metadata: [artifact manifest](evaluations/official-mtp-research-20261024/artifact-manifest.json) and [research provenance](evaluations/official-mtp-research-20261024/research-provenance.json). Parent review corrected terminal cache work, steady-state cost accounting and acceptance wording; preserved both original Copilot reports locally.
