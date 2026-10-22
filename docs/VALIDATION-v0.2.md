# v0.2 local validation — 2026-10-20

Environment: Apple M5 Pro, 48GB unified memory, macOS 26.6 / arm64; .NET SDK 10.0.100 selected by global.json. The workspace was originally empty; Git is initialized on main, without a commit, remote or pull request.

## Acceptance results

| Check | Outcome |
|---|---|
| Portable locked restore after native publishing | Passed |
| Release solution build | 0 warnings, 0 errors |
| Contract/unit/model tests | 24 passed, 0 failed, 0 skipped |
| Managed real-model end-to-end suite | 18 checks passed |
| Native AOT real-model end-to-end suite | 18 checks passed |
| Synthetic/failure-path real-process regression | Managed and native: 24 checks each passed |
| Managed RazorConsole with real Metal model | PTY test passed: Tab/Enter → real text |
| Native RazorConsole with native host / real Metal model | PTY test passed: Tab/Enter → real text |
| Double-click launcher | Real model and 2 live workers rendered; Ctrl+C exited and reaped the cluster |
| Host native publish | Passed; executable plus libdeepinfer_metal.dylib; no reported trim warnings |
| TUI native publish | Passed with one upstream IL2104 warning from Microsoft.AspNetCore.Components 8.0.0 |
| Independent upstream C reference | Complete 48-token continuation matched exactly |
| CUDA / Linux / Windows / multi-host network | Not hardware/runtime verified locally |
| CI | Updated for macOS 15 arm64 and Linux x64; not run remotely |

## Numerical correctness

Actual FP32 Metal GEMV versus SIMD CPU reference:

| Shape | Maximum absolute error | Acceptance threshold |
|---|---|---|
| 1 × 1 | 0 | 0.0001 |
| 53 × 77 | 0.000000953674 | 0.0001 |
| 288 × 288 | 0.00000572205 | 0.0001 |
| 32000 × 288 | 0.00000572205 | 0.0001 |

Complete 32,000-vocabulary logits after sequential prefill were compared for `Once upon a time`, `The little girl` and `Hello 世界`. The largest observed absolute difference was 0.0000190735, below the 0.002 gate; top-token IDs matched. CPU and Metal also produced exactly the same 48 greedy generated token IDs.

For an independent implementation check, llama2.c revision 350e04fe35433e6d2941dce5a1f53308f87058eb was downloaded into ignored artifacts/reference, compiled locally, and run with the same checkpoint/tokenizer, temperature 0 and 52 total forward steps. Its complete text matched the C#/Metal prompt plus 48-token continuation exactly. The product does not invoke that executable.

## Actual end-to-end behavior

The real-model suite launches a coordinator, two independent Metal model workers and one independent CPU reference worker. It verifies real loaded-model advertisements, round robin between GPU workers, real output/token IDs, completed GPU dispatches/timing, per-request KV reset, changed-prompt behavior, CPU token equivalence, tokenized-context HTTP 400 propagation, unsupported CUDA HTTP 503, the actual CLI path, and real-token benchmarking.

For the 48-token acceptance request, five prompt tokens plus 47 decode forwards execute 2,236 Metal GEMV commands (43 per forward). GPU time is measured from completed Metal command buffer GPU timestamps; it excludes CPU operations and RPC/dispatch overhead. It is not total request latency.

The unit/model tests additionally cover odd-width GEMV, shape errors, score-ranked BPE, UTF-8 fallback round trips, literal special-token input, truncated/invalid checkpoints/tokenizers, GQA tensor layout, cancellation and context handling. The synthetic regression retains authentication, limits, worker death/lease expiry and coordinator restart coverage.

## Observed output

Prompt: `Once upon a time`

Continuation:

> , there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.

This is generated from the pinned TinyStories 15M checkpoint, not canned text. The two real workers repeat it because greedy decoding is deterministic. Model/tokenizer SHA-256 values and pinned source revisions are in models/provenance.json.

Raw managed/native model evidence is in artifacts/real-e2e.json and artifacts/real-e2e-native.json. These generated files include actual text, token IDs, request/GPU timings, verification logs and real benchmark results. Terminal evidence is in artifacts/tui-smoke.txt; the external reference comparison is in artifacts/reference/comparison.json. Artifacts are ignored by Git.

## One observed native RPC benchmark

Two Metal worker replicas, 1 warmup excluded, 3 measured requests × 32 generated tokens:

- Total measured time: 762.241 ms.
- Round-trip p50 / p95: 254.751 / 254.983 ms.
- Generated model tokens/sec: 125.94 (96 actual tokens; includes prefill and RPC).
- Completed GPU dispatches: 4644; summed GPU command time: 46.293 ms.
- Simulation: false; synthetic units/sec: zero.

This is a small local sample, not an optimized or statistically broad performance claim.

## Limits

This is genuine end-to-end inference with a small supported model and actual Metal GEMV, not full-GPU execution or DeepSeek support. Normalization, attention, RoPE, SwiGLU, sampling and KV cache remain in C#. The context is 256 tokens. There is no quantization, streaming, GPU attention, optimized batched prefill, persistent chat session or model sharding. CUDA compute is still unsupported. The 15-second RPC timeout exists but was not forced in this acceptance run. Concurrency regression accepts either successful output or explicit HTTP 429; it does not guarantee slot exhaustion for the very fast synthetic backend.
