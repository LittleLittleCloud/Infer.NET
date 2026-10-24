# Qwen 3B / 7B decode benchmark — 2026-10-20

> Historical record: this model path was retired on 2026-10-23. Only Qwen3.6-35B-A3B remains supported. Legacy loaders, model metadata and dedicated scripts referenced below were removed; the commands and paths describe the original experiment and are no longer current instructions. See [current setup](../README.md) and [Qwen3.6 validation](QWEN36-8BIT.md).


Actual DeepInfer.NET Native AOT execution on Apple M5 Pro / 48GB unified memory, macOS 26.6. One BF16 Metal worker was loaded at a time, behind the coordinator. Both use original BF16 weights with FP32 accumulation and C# CPU attention; there is no integer quantization. The shader/kernel implementation is the same for both model sizes.

Prompt: `请用中文详细解释什么是机器学习，至少写五百字。`

One warmup request excluded, then three sequential measured requests of 128 generated model tokens each. Greedy sampling, single-request worker capacity, application context cap 2048. Model loading and warmup are excluded.

| Model | Pure decode (tokens/s) | Mean prefill (s) | End-to-end (tokens/s) | Measured output / decode intervals |
|---|---:|---:|---:|---:|
| qwen2.5-3b-instruct | 14.485 | 2.721 | 11.140 | 384 / 381 |
| qwen2.5-7b-instruct | 10.072 | 3.892 | 7.756 | 384 / 381 |

Pure decode is the wall time from the first emitted output token through the last emitted output token. It counts N−1 intervals per response and includes actual CPU operations, Metal dispatch/wait and GPU compute. It excludes prompt prefill and any extra final forward needed to discover EOS. The aggregate divides total decode intervals by summed decode time; it does not average per-request rates. End-to-end throughput includes full generation/prefill/EOS handling, RPC and response processing.

Worker `firstTokenMs` is an internal compute measurement; the HTTP endpoint is non-streaming, so it is not a user-visible/network TTFT. These three samples are a local baseline of the current implementation, not an optimized hardware throughput claim. GPU attention, batched prefill and command-buffer batching remain unimplemented.

## Actual 7B response

English rendering of the recorded input:

User: Introduce yourself in one sentence.

Original input (verbatim):

```text
用一句话介绍你自己。
```

English rendering of the recorded output:

> I am Qwen, a large language model developed by Alibaba Cloud, capable of answering questions, creating text, expressing opinions, and engaging in anthropomorphic dialogue.

Original output (verbatim):

```text
我是Qwen，由阿里云开发的大型语言模型，可以回答问题、创作文字，还能表达观点和拟人对话。
```

The 7B model is the official Qwen2.5-7B-Instruct checkpoint at a09a35458c702b33eeacc393d103063234e8bc28. All four safetensors shards were SHA-256 verified. It uses an untied output classifier, 28 layers and 3584 hidden dimensions. The tokenizer is byte-identical to the pinned 3B tokenizer. The 7B functional check covers real Chinese text through coordinator/worker/Metal, but no independent full-logit Transformers comparison for 7B was run in this benchmark; that comparison was previously performed for 3B.

## Reproduce

```sh
python3 scripts/download-qwen.py --size 3b
python3 scripts/download-qwen.py --size 7b
dotnet publish src/DeepInfer.Host -c Release -r osx-arm64 -o artifacts/native/osx-arm64
python3 scripts/benchmark-qwen.py
```

Use `DeepInfer7B.command` for the 7B terminal UI; `DeepInfer.command` retains the 3B default. Results and generated token IDs are saved in artifacts/qwen-decode-comparison.json (ignored by Git). Timing arithmetic/edge tests are included in tests/DeepInfer.Tests/TimingTests.cs.
