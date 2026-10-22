# v0.3 Qwen 3B validation — 2026-10-20

Environment: Apple M5 Pro / 48GB unified memory, macOS 26.6 arm64, .NET 10.0.100. Model: official Qwen/Qwen2.5-3B-Instruct at aa8e72537993ba99e69dfaafa59ed015b17504d1. Both official BF16 shards were downloaded and their SHA-256 hashes verified; weights remain BF16 in Metal buffers and accumulate in FP32. The checkpoint and tokenizer are separately downloaded under the retained Qwen Research License.

## Results

| Check | Exact outcome |
|---|---|
| Release solution build | Succeeded; 0 warnings, 0 errors |
| Unit/contract/binary-format tests | 40 passed, 0 failed, 0 skipped |
| Managed Qwen real-process RPC suite | 12 checks passed (independent reference verification also run separately) |
| Native AOT Qwen real-process suite | 13 checks passed, including independent reference verification |
| Qwen tokenizer/chat template | Five prompts matched independent Transformers token IDs exactly |
| BF16 Metal GEMV versus CPU | Max absolute error 0.0000009536743 on irregular 53×77 shape |
| Complete 151936-vocabulary logits | Max absolute error 0.00032925606 versus Transformers FP32; top token identical |
| Actual 3B greedy generation | All 22 generated token IDs identical to independent reference; EOS respected |
| Qwen Metal TUI | Managed and native PTY tests passed; Tab/Enter generated and displayed Chinese model text |
| Default one-click launcher | Exercised real Qwen Chinese generation and Ctrl+C cleanup |
| Original synthetic/failure regression | 24 checks passed |
| Original TinyStories real-process regression | Passed after shared executor changes |
| Native host publish | Succeeded without reported trim warnings; includes Metal bridge sidecar |
| Native TUI publish | Succeeded with one upstream IL2104 warning from Microsoft.AspNetCore.Components 8.0.0 |

The independent reference used PyTorch 2.14.1 and Transformers 5.20.2, the same BF16 source checkpoint promoted to FP32, eager attention and greedy generation on MPS. Those Python tools are optional acceptance tooling only; the .NET worker does not invoke them. Script: scripts/reference-qwen.py; raw reference: artifacts/qwen-reference.json.

## Observed real response

User: 用一句话介绍你自己。

> 我是Qwen，由阿里云开发的超大规模语言模型，旨在提供各种领域的信息和帮助。

The request passed through the CLI/coordinator into a separately launched worker. It returned model ID qwen2.5-3b-instruct, backend metal, isSimulation=false, finishReason=eos, 34 prompt tokens, 22 output token IDs and 14168 completed Metal commands. The product uses the official single-turn chat template, including its default system message. Another Chinese prompt requesting 2+2 returned 4; requests do not share a KV history.

The end-to-end suite also verifies correct model/format advertisements, fresh per-request KV caches, default CLI model selection, context-overflow HTTP 400, unavailable CUDA HTTP 503 and real-token benchmarking. Model load completes before the worker registers its loaded-model capability. Default startup uses one replica to avoid loading unnecessary additional 3B copies.

## One small native RPC benchmark

One measured request, no warmup, 16 actual generated tokens:

- End-to-end measured time: 3227.692 ms.
- Generated tokens/sec including chat prefill and RPC: 4.957.
- Completed Metal commands: 12144.
- Summed GPU command timestamp time: 1135.892 ms.

This is a correctness/operability observation, not an optimized decode throughput claim. C# still performs attention, RoPE, norms, activation, sampling and KV updates; each GPU GEMV is individually dispatched and synchronized. Sequential prefill and CPU/GPU synchronization contribute substantial overhead.

## Evidence and boundaries

- artifacts/qwen-e2e.json: managed RPC outcomes.
- artifacts/qwen-e2e-native.json: native reference/RPC outcomes, full generated text and token IDs.
- artifacts/qwen-reference.json: independent full logits/tokenizer cases/generated IDs.
- artifacts/tui-smoke.txt and artifacts/qwen-launcher-smoke.txt: terminal evidence.
- models/qwen2.5-3b-instruct/provenance.json: fixed revision and file digests.

The application context is capped at 2048 tokens including chat-template/prompt/output, despite the larger trained limit. Output is capped at 256 tokens. No model sharding, persistent multi-turn chat, streaming, GPU attention, quantization or CUDA compute is implemented. Qwen3 is a different architecture and is not claimed to be supported. Linux/Windows, NVIDIA hardware and real multi-host networking were not locally validated. Historical TinyStories v0.2 results remain in docs/VALIDATION-v0.2.md.

## 7B and decode-timing follow-up

Qwen2.5-7B-Instruct was downloaded and SHA-256 verified, then actually executed via the native coordinator and independent BF16 Metal worker. Both 3B and 7B were measured with one excluded warmup and three 128-token responses under the same prompt. Pure decode measured 14.485 tokens/s (3B) and 10.072 tokens/s (7B); end-to-end throughput including prompt prefill was 11.140 and 7.756 tokens/s. Decode timing counts N−1 output intervals and excludes prefill/EOS detection. Full reproducibility, limitations and per-model phase times are in docs/DECODE-BENCHMARK.md.

The 7B native TUI and separate one-click launcher were exercised with actual generated Chinese text and shutdown cleanup. The original 3B launcher remains the default. New timing/edge tests bring the current unit suite to 40 passes. The optional timing fields are additive protocol-v1 changes; no inference kernel was optimized for this measurement. Independent Transformers full-logit validation remains specific to 3B, not 7B.
