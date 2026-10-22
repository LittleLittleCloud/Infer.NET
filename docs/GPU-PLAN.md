# GPU status and next milestones

## Completed in v0.2

- Original Metal C ABI bridge: device/queue/pipeline, persistent FP32 weight buffers, safe .NET handles and deterministic disposal.
- Actual Metal GEMV with two SIMD groups per output row, including irregular shapes.
- CPU numerical reference and complete-vocabulary logit/generation comparisons on M5 Pro.
- FP32 TinyStories model loading/tokenization, sequential prefill/decode, attention and request-local KV cache in C#.
- Actual text through coordinator and independent Metal workers; Native AOT deployment with bridge sidecar.

## Current Metal status — Qwen3.6 MoE

The original Qwen3.6-35B-A3B executor now runs the complete token forward on GPU:
5/8-bit affine projections, normalization, convolution, RoPE, Gated DeltaNet,
full attention/KV updates, Top-8 routing, fused experts and argmax. FP32 state
and activations are retained. One command buffer per token replaces the previous
191 submissions. Native AOT short-context decode measured 96.322 tokens/s on
M5 Pro, compared with 92.186 for the external MLX baseline at the same 24-input /
128-output lengths. Original batched prefill now processes 32-token chunks using FP32 matrix instructions, batched causal attention and register-resident recurrent updates. Streaming and longer-context performance validation remain next work. Dense Qwen2.5 still
uses GPU GEMV with CPU attention. See [benchmark evidence](QWEN36-OWN-BENCHMARK.md).

## CUDA work next

Discovery only remains implemented. Add context/stream/module/allocation ownership, an original SM89 GEMV kernel, actual RTX 4090 numerical tests, then integrate the same transformer contract. No CUDA inference or CUDA performance is claimed without that hardware validation.

## Distributed models later

Current workers hold complete model replicas. Layer/expert placement, bounded binary activation transport, collectives, remote failure semantics and sharded KV ownership remain future work. A real DeepSeek model requires a compatible architecture, tokenizer, quantization implementation and explicit memory budget; the 15M demo does not establish that support.

## Completed in v0.3

Official Qwen2.5-3B-Instruct BF16 safetensors, byte-BPE/chat template, QKV biases and split-half RoPE are supported. BF16 weights are consumed directly by a second Metal GEMV kernel with FP32 accumulation. The original single-device Qwen inference path is implemented in .NET plus the Metal bridge; optimized GPU attention/batched prefill remain next steps.
