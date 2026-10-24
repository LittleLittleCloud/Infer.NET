# Qwen3.6-35B-A3B — original .NET / Metal baseline

> Long-context capacity and new workload measurements are documented in [LONG-CONTEXT-BENCHMARK.md](LONG-CONTEXT-BENCHMARK.md). The 24-input/128-output performance tables below remain short-context historical measurements; they are not 128-input/256-output or 256K results.
Measured on 2026-10-21 UTC, Apple M5 Pro, 20 GPU cores,
48 GiB physical unified memory, macOS 26.6 arm64, .NET 10.0.100 Native AOT.
The coordinator and model worker are independent native processes. The Python
benchmark driver only sends HTTP requests and records results; it performs no
model computation. There is no MLX, llama.cpp or Ollama inference dependency.

## Earlier short-input batched-prefill comparison

Final Native AOT comparison on 2026-10-21, the same M5 Pro 20-core GPU / 48 GiB.
Identical pinned 5-bit checkpoint, 24 input tokens, 128 output tokens, greedy
non-thinking mode, batch size one, no prefix reuse or speculative decoding.
Each engine used one excluded warmup and three trials. Only one model process
used the GPU at a time. First-token values below are **means for all engines**;
the own interval is internal to the worker, while MLX is a local generator.
Load and shader preparation are excluded from these warm request measurements.

| Metric | Own sequential prefill control | Own batched prefill | MLX FP32 | MLX default BF16 |
|---|---:|---:|---:|---:|
| Mean first token (ms) | 217.091 | **92.312** | 93.925 | **87.309** |
| Mean own prefill (ms) | 216.571 | **91.692** | different phase boundary | different phase boundary |
| Aggregate decode (tokens/s) | 96.598 | **96.604** | 85.272 | 90.590 |
| Mean completed own HTTP request (ms) | 1532.750 | **1408.021** | no RPC | no RPC |
| Kernels / complete own request | 101,856 | **86,400** | not counted | not counted |
| Completed submissions / request | 151 | **128** | not counted | not counted |

Mean first token fell **57.5%** with unchanged decode speed in the controlled
own-engine comparison. FP32 first-token latency is comparable to FP32 MLX;
the 1.6 ms mean difference is too small to establish a universal advantage.
Normal BF16 MLX still has a **5.0 ms / 5.7%** first-token advantage. This is a
short-input result, not a long-context performance claim. The three batched
own first-token values were 91.377 / 92.053 / 93.507 ms, with decode
96.559 / 96.635 / 96.617 tokens/s. **All 128 output IDs match FP32 MLX exactly**
and also match the sequential own control. BF16 MLX outputs differ.

This short-input comparison used prompt chunks of up to 32 tokens layer by layer.
The current default is 128 tokens with fused long attention; see the long-context
report above for the subsequent measurements. The earlier change consisted of:

- Original FP32 Metal 8×8 SIMD matrix instructions reuse packed weights across
  adaptive 8/16/24/32-token tiles and 64 output rows (32-row diagnostic retained).
  The inner tile has 32 columns. No external matrix/inference library is used.
- Convolution prepares all tokens from prior history/current projections.
  DeltaNet loads recurrent state once, loops through the chunk in registers,
  and stores the final state plus the correct last-three-token history.
- Full-attention Q/K normalization, RoPE and KV updates, causal score computation
  and gated softmax/value output are batched across tokens. Every query uses
  the exact chunk-base-plus-token-offset boundary; future tokens are excluded.
- Norms, residuals, Top-8 routing and expert operations are batched. Each token
  keeps its individual expert choices. Expert gate/up shares input loads.
- Eight adjacent quantized codes are aligned once and extracted at constant bit
  positions, removing repeated cross-word branches. Activations/state and
  accumulation remain FP32; weights/scales stay in their compact original format.
- Each prompt chunk completes 672 kernels plus classifier/argmax only on the
  final chunk. A 24-token prompt therefore uses **675 kernels / one submission**,
  versus 16,131 / 24 before. Shader preparation and bounded scratch allocation
  happen during model loading. Decode remains one token forward per submission.

Expert grouping, byte-expanded expert codes, an eight-vector projection and
several matrix layouts did not improve the complete workload and were discarded.
In particular the byte cache increased memory and reduced decode to about 88
 tokens/s. No expanded-weight cache is retained in the default or shipped path.

Validation on the final source:

- Release build: zero warnings/errors; **66 unit tests** pass.
- **50 native hardware checks** pass for both 64- and 32-row configurations:
  10 graph primitive cases, 36 independent scalar projection cases covering
  5/8-bit weights, row tails and every adaptive token tile, and four independent
  double-precision recurrent-state/convolution-history cases (1/2/3/32 tokens).
- Native batch-prefill verification passes all **120 independent FP32 MLX layer
  states** plus all 248,320 final logits and eight independent greedy IDs.
  Worst layer relative L2 **2.6669225e-6**, max absolute **3.2424927e-5**;
  logits relative L2 **2.2811550e-6**, max absolute **2.7894974e-5**, argmax 117593.
- An unforced **65-token sequential-GPU differential fixture** crosses chunk
  boundaries 31→32 and 63→64, including a final one-token chunk. All 200 captured
  layer states pass: worst relative L2 **5.7867481e-6**; logits relative L2
  **1.6860643e-6**, max absolute **2.3961067e-5**, same argmax 103724. This is a
  regression against the preceding GPU executor, not an independent long-text
  MLX oracle. The long near-tied-routing caveat below still applies.

Native host and TUI publishing succeeded. The native batch-prefill trace/greedy
verification also passed. All **24 process/RPC integration checks** passed;
the actual one-click MoE launcher generated Chinese text through the independent
worker, and Ctrl+C removed its owned processes. Final live acceptance additionally
checks natural EOS, repeated fresh-state arithmetic, and rejection of a tokenized
context overflow below the character limit.

Reproduce, with other model workers stopped:

```sh
dotnet publish src/DeepInfer.Host -c Release -r osx-arm64 -o artifacts/native/osx-arm64
python3 scripts/benchmark-qwen36-gpu.py --variants token-prefill concurrent --output-prefix qwen36-prefill-final
./artifacts/native/osx-arm64/deepinfer verify-qwen36 --prefill
python3 scripts/compare-qwen36.py
bash scripts/check-metal-graph.sh
# Independent reference environment only:
artifacts/baselines/mlx-env/bin/python scripts/benchmark-qwen36-mlx.py --prompt-tokens 0 --generation-tokens 128 --precision fp32 --output artifacts/baselines/qwen36-mlx-prefill-fp32.json
```

`DEEPINFER_METAL_PREFILL=0` retains sequential token prefill;
`DEEPINFER_METAL_PREFILL_MATRIX=0` retains batching with the vector projection.
`DEEPINFER_METAL_PREFILL_ROWS=32` and `DEEPINFER_METAL_PREFILL_TOKEN_TILE=8|32`
are diagnostic tile controls. Ordinary startup uses the measured defaults.
First-token timing is not time-to-display in the non-streaming TUI/API.
Cancellation is observed between prefill chunks and between decode tokens.

Final raw evidence (machine-local, ignored by Git):

- `artifacts/baselines/qwen36-prefill-final-{token-prefill,concurrent}.json`
- `artifacts/baselines/qwen36-prefill-final-ab-summary.json` and its log
- `artifacts/baselines/qwen36-prefill-parity-summary.json` (cross-engine ID/metric proof and raw-file hashes)
- `artifacts/baselines/qwen36-prefill-final-acceptance.json` (live native service checks)
- `artifacts/baselines/qwen36-mlx-prefill-{fp32,final}.json` and their logs
- `artifacts/reference/qwen36-prefill-final-{verify,comparison}.log`
- `artifacts/reference/qwen36-prefill-final-boundary{,-comparison}.log`
- `artifacts/prefill-{build,tests,hardware-tests,hardware-tests-r32,publish,tui-publish}.log`
- `artifacts/prefill-{protocol-integration,launcher}.log`

## Complete GPU executor: preceding sequential-prefill release

Measured 2026-10-21, M5 Pro 20-core GPU / 48 GiB. Each engine used the same
checkpoint, natural 24-token Chinese chat prompt, greedy non-thinking mode,
128 output tokens, one warmup and three trials. Only one model process used
the GPU at a time. MLX was measured again immediately before the final own-engine
run. Decode is N−1 output intervals divided by their total wall time, excluding
load, prefill and the first token. The own measurements go through the native
coordinator/worker; the decode interval is measured inside the worker.

| Metric | Previous own release | Current own GPU executor | External MLX-LM |
|---|---:|---:|---:|
| Aggregate decode tokens/s | 18.905 | **96.322** | 92.186 |
| Individual decode trials | 18.897 / 18.914 / 18.904 | **96.309 / 96.271 / 96.385** | 92.169 / 92.313 / 92.076 |
| Worker mean first token / MLX median first token (ms) | 1193.528 | 218.073 | **85.906** |
| Mean own prefill (ms) | 1188.284 | 217.361 | batched, different boundary |
| Mean completed own HTTP request (ms) | 7913.099 | 1537.534 | single-process, no RPC |
| Kernels / decode forward | 501 | 675 | not counted here |
| Submissions/waits / forward | 191 | **1** | not counted here |
| Activation / state / accumulation | FP32 | FP32 | normal BF16 model path |

The measured short-context decode improved **5.10×** over the previous release
and was **4.49%** above the external reference. This establishes parity for this
particular decode workload, not all-context or end-to-end parity. Prefill remains
sequential; first-token latency is still about 2.54× MLX's. Outputs differ from
normal BF16 MLX after rounding/routing differences; all 128 output IDs do match
the preceding own FP32 engine exactly. The separate FP32 MLX oracle passes the
numerical and greedy checks described below. The product links neither MLX nor
another inference runtime.

Two changes account for the improvement:

1. The complete token forward now stays on GPU: embedding, normalization,
   convolution, both attention types, routing, experts, residuals and argmax.
   Intermediate activations/state remain in native Metal buffers. Normal decode
   returns only one token ID. A concurrent encoder overlaps independent
   projections, with explicit dependency barriers. One command buffer is
   committed and waited for per token, rather than 191.
2. Each quantized SIMD lane now unpacks eight consecutive values from two words,
   reusing affine parameters. A 32-lane SIMD group covers four 64-value groups
   per iteration. Shapes not divisible by 256 columns use the earlier loop.
   There is no precision reduction, weight expansion, skipped layer, expert
   reduction or forced routing in the performance path.

Intermediate measurements: the initial complete GPU executor reached 46.073
 tokens/s; four-way loop unrolling plus concurrent projections reached 54.323;
 the new contiguous unpacking reached 97.033 in its first run and 96.322 in the
 final repeated comparison. An eight-row tile did not improve the four-row
 default; an alternative subgroup layout and eight-way loop unrolling regressed
 and were discarded.

Current verification:

- Release build: zero warnings/errors; **66 unit tests** pass on this Mac,
  including 5/8-bit scalar oracles, 64/192-column fallback, 256/512/2048/4096
  columns, tail rows, eight reordered experts, changed inputs, batched jobs
  and fused gate/up/SwiGLU/down. The optional eight-row tile also passes 66.
- **10 native hardware checks** pass: Top-8 ties/extremes, RMSNorm and GPU argmax
  ties/non-finite rejection.
- Native AOT: **120 independent FP32 MLX layer comparisons** pass. Worst relative
  L2 **1.1575084e-6**, max absolute **1.5258789e-5**. Complete 248,320-logit
  relative L2 **1.0226854e-6**, max absolute **1.4543533e-5**, argmax 117593.
  Eight independent greedy token IDs also match.
- A 512-token synthetic host-reference fixture with identical expert choices
  held fixed passes: worst layer relative L2 **2.4101258e-6**, logits relative
  L2 **1.1011861e-6**, argmax 248046. This is an operator/state diagnostic only.
  Without holding choices fixed, the earlier scalar-unpack GPU graph selected
  a different eighth expert at position 348/layer 27: the CPU eighth/ninth scores
  differed by only **3.8147e-6**. The unforced long fixture did **not** pass tight
  layer tolerances. Do not interpret the fixed-route pass as arbitrary long-text
  token identity. Ordinary workers reject the forced-route environment variable.
- Native host and TUI publish successfully. **24 process/RPC integration checks**
  pass. The actual MoE one-click launcher renders RazorConsole, produces Chinese
  text through a separate worker and reaps its owned processes on Ctrl+C.

Reproduce the complete-GPU run after stopping other model workers:

```sh
dotnet publish src/DeepInfer.Host -c Release -r osx-arm64 -o artifacts/native/osx-arm64
python3 scripts/benchmark-qwen36-gpu.py --variants concurrent
bash scripts/check-metal-graph.sh
```

The default controller compares host, serial-GPU, concurrent-GPU and eight-row
GPU variants, checks real dispatch/submission counts and exact own-engine IDs,
and cleans up every cluster. `DEEPINFER_METAL_VECTOR_QUANT=0` selects the preceding
unpack loop for controlled comparisons. Legacy numerical measurements below
predate the current GPU executor and unpacking; use their original recorded
artifacts for historical results.

Current machine-local evidence (ignored by Git):

- `artifacts/baselines/qwen36-gpu-final-128.json` and `qwen36-gpu-final.log`
- `artifacts/baselines/qwen36-mlx-parity-128.json` and its log
- `artifacts/baselines/qwen36-gpu-vector.log`, `qwen36-gpu-ab.log`
- `artifacts/reference/qwen36-gpu-final-{verify,comparison}.log`
- `artifacts/reference/qwen36-vector-long-fixed{,-comparison}.log`
- `artifacts/gpu-graph-{build,tests,tests-wide,hardware-tests,publish,tui-publish}.log`
- `artifacts/gpu-graph-{protocol-integration,launcher}.log`

## Quantized kernels and fused experts: preceding host executor

Measured 2026-10-21 UTC on the same M5 Pro 20-core GPU / 48 GiB Mac. A single
Native AOT executable was run sequentially in three worker configurations.
Each used the same 24-token prompt, 128 output tokens, greedy non-thinking
decode, one warmup and three trials. Only one model worker used the GPU at a
time. **All 12 responses returned exactly identical token IDs across modes.**

| Metric | Previous kernels, unfused | Tiled GEMV, unfused | Tiled GEMV + fused experts |
|---|---:|---:|---:|
| Aggregate decode tokens/s | 11.844 | 16.729 | **18.905** |
| Mean prefill (ms) | 1847.688 | 1349.082 | 1188.284 |
| Mean worker first token (ms) | 1852.591 | 1354.283 | **1193.528** |
| Mean completed HTTP request (ms) | 12576.866 | 8947.404 | 7913.099 |
| Mean GPU command interval total/request (ms) | 6525.019 | 3259.874 | 3138.047 |
| Kernels per decode forward | 541 | 541 | **501** |
| Submissions/waits per decode forward | 231 | 231 | **191** |
| Kernels per complete request | 81668 | 81668 | 75628 |
| Submissions per complete request | 34858 | 34858 | 28818 |

The new GEMV alone improved decode by **41.2%**. Expert fusion added **13.0%**;
the combined improvement over the preceding release was **59.6%**. Final trial
decode rates were 18.897, 18.914 and 18.904 tokens/s. Weight quantization, FP32
activation/state/accumulation precision, model layers, routing and prompt are
unchanged. There is no external inference runtime. These are local sequential
runs, not a statistical guarantee for other contexts or thermal conditions.

The optimized GEMV specializes 5/8-bit decode and 64-value affine groups. One
SIMD group completes each row without a threadgroup reduction barrier; large
matrices pack four row groups into 128 threads. Expert slots use a 2D grid and
down projections use per-slot input vectors. The expert gate/up kernel shares
input loads and computes SwiGLU on GPU; expert down consumes that native buffer
after an explicit Metal memory barrier in the same submission, alongside the
independent shared down projection. CPU routing, shared-expert activation,
normalization, full attention and sequential prefill remain as before.

The actual-checkpoint microbenchmark in `scripts/benchmark-metal-quant.py`
compares old and new kernels on the same matrices and selected expert IDs.
It repeats warm resident weights and records command GPU intervals and host
wall time separately; it is not end-to-end performance or kernel-only hardware
profiling. Raw samples and library/checkpoint digests are saved in
`artifacts/kernels/quant-gemv.json`. Some tiny matrices show little improvement;
the large projections and selected expert matrices account for most gains.

All **58 unit tests** passed, including independent scalar/double-precision
oracles for 5/8-bit GEMV with 512/2048/4096 columns, tail rows, eight reordered
experts, batched/shared inputs, and fused SwiGLU/down with changed inputs and
mixed down precision. Invalid fused calls are rejected before GPU submission.
Native AOT verification passed the 120 layer states and eight independent
greedy tokens. Worst layer relative L2: **1.2781632e-6**; max absolute:
**1.9073486e-5**. Final logits relative L2: **1.2161233e-6**; max absolute:
**1.4305115e-5**, with the same argmax 117593. Reduction order changes are covered
by the stated numerical tolerances; bitwise float equality is not claimed.
The actual MoE one-click launcher and RazorConsole TUI also passed end to end
with the new kernels, and Ctrl+C reaped all launcher-owned processes. Evidence:
`artifacts/quant-kernel-launcher.log` and `artifacts/qwen-moe-launcher-smoke.txt`.

Reproduce the three configurations after stopping existing model workers to
avoid loading two copies into unified memory:

```sh
dotnet publish src/DeepInfer.Host -c Release -r osx-arm64 -o artifacts/native/osx-arm64
python3 scripts/benchmark-qwen36-kernels.py
```

The controller uses ports 5300–5301 by default, launches and cleans up each
owned cluster, and validates actual kernel/submission counts and cross-mode
token equality. It sets GENERIC+EXPERT_UNFUSED for the preceding release,
EXPERT_UNFUSED alone for new GEMV, and neither flag for the final optimized
path. Flags are prefixed `DEEPINFER_METAL_`; no SERIAL flag is used in this test.
For ordinary use start the demo/launcher without those diagnostic flags.

Raw evidence: `artifacts/baselines/qwen36-kernel-ab-summary.json`, the three
`qwen36-kernel-{generic-unfused,tiled-unfused,optimized}.json` files, and
`artifacts/reference/qwen36-kernel-native.log` / `qwen36-kernel-comparison.log`.
The earlier measurements below are preserved as historical baselines.

## Reduced synchronization: preceding release

The same Native AOT executable was measured consecutively in serial diagnostic
mode and normal batched-submission mode. Both used identical 5/8-bit weights,
FP32 arithmetic, the 24-token prompt, one warmup and three 128-token trials.
Only one model worker used the GPU during each measurement. All eight responses
(warmup plus three trials per mode) returned exactly the same token IDs.

| Metric | Serial control | Batched submissions |
|---|---:|---:|
| Aggregate decode tokens/s | 7.732 | **11.835** |
| Mean prefill (ms) | 2932.973 | 1848.463 |
| Mean worker first token (ms) | 2937.682 | 1852.985 |
| Mean completed HTTP request (ms) | 19365.053 | 12585.184 |
| Submissions/waits per decode forward | 541 | **231** |
| Completed kernel dispatches per request | 81668 | 81668 |
| Completed command buffers per request | 81668 | **34858** |

Decode improved **53.1%**, and per-forward submissions fell **57.3%**. Individual
batched trial decode rates were 11.824, 11.832 and 11.849 tokens/s. GPU command
interval totals stayed close: 6.594–6.598 seconds per serial request and
6.524–6.526 seconds per batched request. Most of the wall-time reduction came
from eliminating repeated submit/wait boundaries, with unchanged GEMV/delta
kernels. Command-buffer GPU intervals are not kernel-only profiler counters.
This measures submission batching, not batched prompt prefill.

The implementation groups independent QKV/Z/A/B or Q/K/V projections. MoE has
three submission stages: router plus shared input projections; expert gate/up
plus shared down; expert down after its CPU activation. CPU dependency boundaries
remain explicit. It still submits/waits 231 times per decode forward; GPU
normalization/routing/full attention and batched prefill are future work.

All 49 unit tests passed, including mixed-shape/5/8-bit independent batches,
selected experts with distinct input vectors, input changes across repeated
submissions, invalid/aliased/cross-device/disposed job rejection before any
submission, and source-generated JSON compatibility for submission counters.
Native AOT verification again passed all 120 layer comparisons and eight
independent greedy token IDs, with the same numerical errors recorded below.
The native host passed all 24 process/RPC integration checks. The rebuilt
Native AOT TUI and actual MoE one-click launcher generated Chinese text through
the batched worker, and Ctrl+C reaped all launcher-owned inference processes.
Logs: `artifacts/quant-batch-protocol-integration.log` and
`artifacts/quant-batch-launcher.log`.

For the controlled serial measurement, start the worker with the diagnostic
environment variable (the demo passes it to its child worker):

```sh
DEEPINFER_METAL_VECTOR_QUANT=0 DEEPINFER_METAL_GENERIC=1 DEEPINFER_METAL_SERIAL=1 ./artifacts/native/osx-arm64/deepinfer demo --model qwen3.6-35b-a3b --stay
# In another terminal:
python3 scripts/benchmark-qwen36-own.py --expected-submissions-per-forward 541 --output artifacts/baselines/qwen36-serial-ab.json
```

Stop that demo, then start it with `DEEPINFER_METAL_GENERIC=1` and
`DEEPINFER_METAL_EXPERT_UNFUSED=1` (without SERIAL). Run
`python3 scripts/benchmark-qwen36-own.py --expected-submissions-per-forward 231 --output artifacts/baselines/qwen36-batched-5bit.json`.
These flags retain this preceding release's kernels and execution path. The
script validates actual kernel and submission counts.
The serial control is `qwen36-serial-ab.json`; cross-mode token equality and speed
ratios are retained in `qwen36-batch-ab-summary.json`. Native correctness logs:
`artifacts/reference/qwen36-batched-native.log` and `qwen36-batched-comparison.log`.

## Original serial-submission baseline

One warmup, then three sequential requests with the same 24-token chat prompt,
greedy non-thinking decoding, 128 generated tokens each. All requests reached
the length limit and returned identical token IDs from fresh per-request state.

| Trial | Prefill (ms) | Worker first token (ms) | Decode intervals | Decode (ms) | Decode tokens/s |
|---|---:|---:|---:|---:|---:|
| 1 | 2922.542 | 2926.649 | 127 | 16445.651 | 7.722 |
| 2 | 2919.694 | 2923.712 | 127 | 16405.475 | 7.741 |
| 3 | 2936.497 | 2940.537 | 127 | 16534.739 | 7.681 |

Aggregate decode: **7.715 tokens/s**, computed as total N−1 token intervals
divided by their total elapsed time. Mean prefill: **2926.244 ms**.
Mean worker first token: **2930.300 ms**. Mean completed HTTP request:
**19393.818 ms**. First-token timing is internal to the non-streaming worker,
not a measurement of when text appears on the user's screen.

Each measured request completed 81,668 own GPU dispatches and approximately
6.79–6.84 seconds of GPU command time. Command time and wall time are distinct.
The original baseline committed and synchronized each projection/delta
command individually. CPU operations and this submission boundary are included
in the reported throughput. This is the initial correctness baseline, with
unoptimized kernels and sequential prefill.

The separately measured MLX baseline used BF16 activations, all-GPU execution,
batched prefill and 256 output tokens. Its 90.449 tokens/s short-input result
is an external reference, not this implementation's performance. These runs
are not a controlled kernel comparison; their execution precision and output
lengths differ. This historical release was capped at 2048 total tokens; new long-context measurements are separate.

## What is implemented

- Direct safetensors loading into native Metal buffers, affine 5-bit/group64
  packed weights, BF16 scales/biases, 8-bit router matrices.
- Original Metal GEMV for ordinary and selected-expert stacked matrices;
  original GPU Gated DeltaNet state recurrence. No weight expansion to FP32.
- C# tokenizer/chat template and generation control; original GPU RMSNorm,
  depthwise convolution/SiLU, partial RoPE, gated causal full attention,
  stable Top-8 routing, shared expert gating, expert merging and argmax.
  Activations, state and arithmetic accumulation remain FP32.
- All 40 text layers: 30 Gated DeltaNet + 10 full attention; 256 routed experts,
  eight active per token per layer, plus a shared expert. All weights reside on
  this Mac; no expert offload, rank transport or second GPU is involved.
- Fresh caches for each request, EOS/context/length handling, Native AOT worker,
  coordinator RPC and RazorConsole TUI integration.

Current scope: this fixed text architecture/checkpoint, one active request per
worker. This historical release allowed 2048 total tokens and at most 256 output tokens; see the separate long-context report for the new limits. No vision, MTP, YaRN,
continuous batching or streaming.

Checkpoint: `mlx-community/Qwen3.6-35B-A3B-5bit`, revision
`c90c1a8b7e7a6a9e39e549f0361a96eb05bbb499`. All five shards were verified against
pinned SHA-256 digests by the downloader. `mlx` identifies the affine checkpoint
layout and independent reference; it is not the execution backend.

## Correctness evidence

An independent MLX-LM 0.32.2 process generated FP32 oracle artifacts, then
exited before our process loaded the weights. The product never invokes it.
The reference uses the same quantized weights with FP32 floating tensors to
compare at our precision; this differs from its BF16 performance benchmark.

English rendering of the recorded output:

`verify-qwen36` verifies exact prompt token IDs and eight greedy generated token
IDs. It also evaluates the 24-token prompt-plus-generated sequence, exporting
all 40 layer outputs at token positions 0, 16 and 23. Independent generation:
“I am Tongyi Qianwen, from Alibaba Cloud.” Native AOT passed the greedy token comparison.

Original output (verbatim):

```text
我是通义千问，由阿里云
```

| Comparison | Result |
|---|---:|
| Layer states checked | 120 |
| Worst layer relative L2 error | 1.2297192e-6 |
| Worst layer maximum absolute error | 3.4332275e-5 |
| Final 248,320 logits relative L2 error | 1.1128401e-6 |
| Final logits maximum absolute error | 1.4781952e-5 |
| Final argmax, both engines | 117593 |

The comparison script rejects relative L2 error above 1e-4 or maximum absolute
error above 0.01, non-finite values, shape differences and final argmax mismatch.
This finite fixture validates both recurrent attention and ordinary attention
through successive tokens; it does not establish correctness for all prompts
or long contexts. GPU unit tests separately compare 5/8-bit selected expert GEMV
and a multi-step delta recurrence with independent scalar arithmetic.

An actual coordinator request generated 27 tokens, ended on EOS and returned:

English rendering of the recorded output:

> I am Tongyi Qianwen, a large language model independently developed by Tongyi Lab, dedicated to being your capable thinking partner and assistant.

Original output (verbatim):

```text
我是通义千问，由阿里云通义实验室独立研发的大语言模型，致力于成为您得力的思考伙伴与助手。
```

English rendering of the recorded input:

Release build passed with zero warnings/errors; all 45 unit tests passed on
this Mac. Native AOT publishing succeeded for host and TUI. The 24-check
process/RPC integration suite passed against the native host. The real
`DeepInferMoE.command` was exercised in a pseudo-terminal: it loaded one model
worker, rendered RazorConsole, generated Chinese text through the coordinator,
and Ctrl+C removed all launcher-owned processes.
The original Qwen 3B Native AOT end-to-end suite also passed all 12 checks,
including its independent greedy token fixture, request-local KV reset,
context-overflow error, default CLI and real-model benchmark.
Additional own-MoE RPC checks returned exactly `4` for “Calculate 2 plus 2; return only the number.”,
repeated identical token IDs with fresh state, and returned HTTP 400 for an
oversized tokenized context. The coordinator wraps the worker's
`invalid_model_input` payload in its existing `worker_error` envelope.

Original input (verbatim):

```text
计算2加2，只回答数字。
```

## Reproduce

```sh
dotnet build DeepInfer.slnx -c Release
dotnet test DeepInfer.slnx -c Release --no-build
dotnet publish src/DeepInfer.Host -c Release -r osx-arm64 -o artifacts/native/osx-arm64
dotnet publish src/DeepInfer.Tui -c Release -r osx-arm64 -o artifacts/tui-native/osx-arm64
./artifacts/native/osx-arm64/deepinfer demo --model qwen3.6-35b-a3b --stay
# Another terminal, while only this model worker is using the GPU:
python3 scripts/benchmark-qwen36-own.py
```

For independent numerical validation only (requires the separately installed
reference environment described in `MLX-BASELINE.md`), stop the model worker
first to avoid loading two copies into unified memory:

```sh
artifacts/baselines/mlx-env/bin/python scripts/reference-qwen36.py
./artifacts/native/osx-arm64/deepinfer verify-qwen36
python3 scripts/compare-qwen36.py
```

Machine-local raw evidence (ignored by Git):

- `artifacts/baselines/qwen36-own-5bit.json`: full warmup/trial responses, IDs,
  GPU dispatch/time and request timings, configuration and provenance digest.
- `artifacts/baselines/qwen36-own-intro.json`: Chinese EOS generation through RPC.
- `artifacts/baselines/qwen36-own-acceptance.json`: arithmetic/state/context checks.
- `artifacts/reference/qwen36/reference.json`, per-layer `.f32` files and logits.
- `artifacts/reference/qwen36/own/`: original engine traces and logits.
- `artifacts/reference/qwen36/comparison.json`: all numerical comparison results.
- `artifacts/reference/qwen36-own.log`: Native AOT trace/greedy verification.
- `artifacts/own-moe-protocol-integration.log`: 24 process/RPC checks.
- `artifacts/qwen-moe-launcher-smoke.txt`: actual launcher/TUI generation.

The raw throughput values above were measured before later documentation and
launcher checks. No GPU benchmark ran concurrently with another GPU workload.
