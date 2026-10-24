# Qwen3.6-35B-A3B: external Mac baseline

This is an external MLX-LM reference measurement, **not execution by DeepInfer.NET**.
No MLX/Python runtime is added to the .NET product. The independent .NET/Metal
backend now implements this MoE model too. Its implementation and
its own measurements are documented in [QWEN36-OWN-BENCHMARK.md](QWEN36-OWN-BENCHMARK.md).

The latest 2026-10-21 repeat uses 24 input / 128 output tokens, one warmup and
three trials per engine. Default BF16 MLX measured **90.590 tokens/s**, mean
first token **87.309 ms**; FP32 MLX measured **85.272 tokens/s**, mean first token
**93.925 ms**. The own FP32 Native AOT path measured **96.604 tokens/s**, mean
first token **92.312 ms** after batching prefill. The own 128 generated IDs match
FP32 MLX exactly; normal BF16 outputs differ. Same-precision short-input latency
is comparable; BF16 still has lower first-token latency. No long-context parity
is established. Raw files: `artifacts/baselines/qwen36-mlx-prefill-{fp32,final}.json`
and `qwen36-prefill-final-concurrent.json`; current implementation and validation
are described in the own-engine benchmark document linked above.

## Measured results — 2026-10-20 (America/Los_Angeles)

| Input tokens | Decode (tokens/s) | Local TTFT, median | MLX prompt phase (tokens/s) | Peak MLX memory |
|---|---:|---:|---:|---:|
| 24, short chat | 90.449 | 86.568 ms | 278.923 | 23.980 GB |
| 4,096 | 86.913 | 1,452.592 ms | 2,821.400 | 25.844 GB |
| 16,384 | 79.973 | 6,309.585 ms | 2,598.931 | 26.096 GB |

Three measured requests per row, 256 output tokens each: **2,304 measured output
tokens and 2,295 measured decode intervals** in total. The three per-row decode
standard deviations were 0.056, 0.188 and 0.188 tokens/s. All three measured runs
within each case produced identical output token IDs. One excluded warmup per
case is retained in the raw evidence.

Model loading took 3.42 seconds, excluded from request timings. Loaded text-model
MLX memory was approximately 23.84 GB. The natural-EOS functional answer was:

English rendering of the recorded output:

> 17 multiplied by 23 equals 391.
>
> Machine learning is an artificial intelligence technology that enables computer systems to automatically learn patterns and improve performance by analyzing data, without requiring explicit programming instructions.

Original output (verbatim):

```text
17乘以23等于391。

机器学习是一种人工智能技术，它使计算机系统能够通过分析数据自动学习规律并改进性能，而无需进行显式的编程指令。
```

The raw report passed 62 consistency checks covering the model revision,
hardware, benchmark source hash, request/output lengths, monotonically increasing
timestamps, N−1 decode intervals and aggregate timing arithmetic. The benchmark
process exited successfully. No thermal/performance warning was reported by
`pmset`; this does not prove a fixed device clock. System swap already existed
before loading and increased from 1,662.06 MiB to 1,796.25 MiB during the overall
run; no attribution of those swapped pages to the model is made. The .NET 7B
service was restored and its worker verified healthy after the MLX process exited.

## Model and environment

- Machine: Apple M5 Pro, 20 GPU cores, 48 GiB unified memory, AC power.
- Quantized checkpoint: [mlx-community/Qwen3.6-35B-A3B-5bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-5bit).
- Immutable revision: `c90c1a8b7e7a6a9e39e549f0361a96eb05bbb499`.
- Download: approximately 24.759 GB including metadata/tokenizer and vision weights.
- Quantization: affine, default 5 bits/group size 64; 80 tensor-specific 8-bit overrides.
- Text-only execution via MLX-LM's built-in `qwen3_5_moe` model; vision weights are
  discarded by its sanitizer. No custom remote Python model code is enabled.
- Python 3.13.14; MLX/MLX-Metal 0.32.4; MLX-LM 0.32.2; Transformers 5.20.2.
- Full Python environment pins: `scripts/requirements-mlx-baseline.txt`.
- The downloader verifies the five weight shards and tokenizer against the pinned
  repository's published SHA-256 hashes, validates small Git blobs where available,
  retains the upstream Apache-2.0 license, and writes `provenance.json`.

## Method

One model process, batch size 1, greedy sampling, thinking disabled, no draft
model/speculative decoding, no prefix reuse, no quantized KV cache, prefill chunks
of 2,048 tokens. Each request creates fresh attention/recurrent state.

The harness first performs a natural-EOS Chinese functional check: compute
`17 × 23` and explain machine learning. This only checks practical generation;
it is not a full-logit comparison or a model-quality evaluation.

Each benchmark case has one excluded warmup and three measured requests, each
forced to produce exactly 256 tokens by temporarily disabling EOS termination,
as the upstream `mlx_lm.benchmark` does. This fixed-length policy is only for the
benchmark. It does not make the continued text after EOS a quality sample.

Cases are a short natural chat prompt, exactly 4,096 input tokens, and exactly
16,384 input tokens. Long cases tokenize repeated Chinese/English source text,
preserve the final 128 tokens containing the question/assistant template, and
truncate the middle to obtain the exact input count. Token IDs are saved for
reproduction. The source text is in the harness. Tokenization and file writes
are outside the timed region.

- **Decode:** wall time from the first emitted token through the last emitted
  token, counting N−1 intervals. Aggregate = total intervals / total time.
  This matches the interval definition used by the .NET benchmark, although the
  model, precision and runtime differ.
- **Local TTFT:** request start to first generator yield. Includes prompt
  processing and first-token generation; excludes model loading, tokenization,
  network and server overhead.
- **Prompt throughput:** MLX's reported `prompt_tps`. Its prompt phase includes
  producing the first token; it is not isolated matrix-multiplication throughput.
- **Memory:** peak MLX allocator memory per measured request in decimal GB.
  This is not whole-system RSS or the Mac's total used unified memory.
- Load time and all warmups are excluded from measured request summaries.
- We stop the previously launched .NET 7B model service during the measurement,
  then restore it. The desktop and other ordinary applications remain running.

## Reproduce

From the repository root on an Apple Silicon Mac:

```sh
uv venv --python /opt/homebrew/bin/python3.13 artifacts/baselines/mlx-env
uv pip install --python artifacts/baselines/mlx-env/bin/python -r scripts/requirements-mlx-baseline.txt
artifacts/baselines/mlx-env/bin/python scripts/download-qwen36-mlx.py
# Stop other inference workloads before measuring.
caffeinate -i artifacts/baselines/mlx-env/bin/python scripts/benchmark-qwen36-mlx.py
```

Raw evidence is written incrementally to
`artifacts/baselines/qwen36-mlx-5bit.json`: hardware/software versions, model
revision, quantization settings, power/thermal/swap observations, functional
sample, prompt/output token IDs, per-token timestamps, warmups and measured runs.
Downloaded model binaries and raw artifacts are ignored by Git. Model provenance
and license are retained under `models/qwen3.6-35b-a3b-5bit-mlx/`.

The public [oMLX community result](https://omlx.ai/benchmarks/performance/k2tn9171)
uses the same chip/core count/RAM and a 5-bit model label, but a different runtime
and measurement protocol. Its model hash is not provided on that page. It is a
reference data point, not a guaranteed reproduction target.
