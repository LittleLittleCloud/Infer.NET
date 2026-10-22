# Long-context and long-output measurements

This benchmark uses the own .NET Native AOT / Metal Qwen3.6-35B-A3B engine,
not MLX inference. The Python `tokenizers` package prepares data only; the own
C# tokenizer must reproduce every prepared chat token ID before a request runs.

The pinned checkpoint's `text_config.max_position_embeddings` is **262,144**.
Input, chat-template overhead and output share that budget. For example,
245,760 input tokens leave 16,384 output tokens. A 270,000-token input plus any
output exceeds this unscaled RoPE configuration; no YaRN extension is implemented.
Raising a number alone would not establish correct extended-context inference.

The engine now permits that native capacity and up to **32,768 output tokens**.
Capacity support, component correctness, whole-model long-context correctness,
and measured whole-request performance are different claims. A fixed-output 245,760-input / 16,384-output end-to-end request has now completed
(see below); that establishes execution at the combined 262,144-token limit,
not comprehensive long-context answer quality.
The saved plan explicitly marks unexecuted cases as `planned`, never measured.

Current prefill now defaults to 2048-token outer chunks, GPU grouped expert
GEMM and chunkwise DeltaNet, with attention workspace bounded to 128 queries.
The implementation, final tests and subsequent measurements are documented in
[PREFILL-GEMM-DELTA.md](PREFILL-GEMM-DELTA.md). All tables below preserve their
original configurations; the completed 240K-input run predates those changes.

## Earlier fused long-prefill optimization (2026-10-21)

Defaults for this earlier comparison: 128-token chunks and original FP32 fused long attention.
The quantized weights, FP32 activations/state/KV/accumulation, model, device and
single-request execution remain the same. No external inference runtime or
prefix reuse was introduced.

| Exact input tokens | Previous prefill (s) | Current prefill (s) | Prefill throughput before → after | Speedup | Evidence |
|---:|---:|---:|---:|---:|---|
| 8,192 | 37.865 | 26.922 | 216.3 → 304.3 tokens/s | 1.41× | Three trials per variant, same final binary |
| 32,768 | 257.645 | 126.731 | 127.2 → 258.6 tokens/s | 2.03× | One historical baseline trial vs one final trial |

The 8K requests use the exact same saved prompt, 128 fixed output tokens and
16,384-slot cache. All three legacy/chunk32 requests run before the three
fused/chunk128 requests. No warmup request is excluded; loading and shader
preparation are outside timing. Previous prefill trials are 38.640 / 37.478 /
37.478 seconds; current trials are 27.761 / 26.497 / 26.509 seconds. This is a
local sequential A/B measurement, not a universal hardware throughput claim.

The 32K request reuses the exact earlier prompt; both old and new requests
reserve 65,536 cache slots. The old request generated 8,192 outputs and the new
request 128, so this row compares **prefill only**. The current first 128 output
IDs match the old output prefix. The one-trial comparison is weaker evidence
than the three-trial 8K comparison.

8K aggregate decode measured 69.86 → 67.49 tokens/s in these requests; no decode
improvement is claimed. Within each variant all three outputs match. Across
variants the 8K outputs first differ at generated token 60. FP32 reduction-order
changes and discrete MoE routing are known sensitivity points; this particular
long-output divergence has not been isolated to a specific route. These timings
do not establish arbitrary long-context token identity or quality. There is no
new matched long-prefill MLX measurement or parity claim.

The attention operator now keeps QK, online softmax and PV in bounded tiles,
sharing KV loads across four query tokens and eight GQA heads. It emits only
partition statistics and uses the stable combination stage. With 128-token
chunks instead of 32, the 8K prefill also uses 64 submissions instead of 256.
Full attention still has quadratic arithmetic in input length. Selected-expert
projections and the causally sequential DeltaNet recurrence remain further
optimization targets; no claimed phase percentages come from a GPU profiler.

The isolated 32-query FP32 attention microbenchmark, one excluded warmup and
three trials per path, improves by 5.14× at 32K keys and 5.56× at 262K keys.
Those are attention-only timings, not model token throughput. Raw results:
`artifacts/baselines/qwen36-fused-attention-unroll256-micro.json`.

GPU grouped-expert experiments were removed. During development, expanding the
chunk exposed an undersized short-score scratch allocation. That was fixed;
the 57 random fused/short checks explicitly cover its real 128-token allocation.
All final measurements above follow the fix. Affected exploratory reports are
excluded and identified in `qwen36-prefill-discarded-experiments.json`.

Final evidence:

- `artifacts/baselines/qwen36-fused-long-prefill-summary.json` records input,
  report and binary hashes, timing samples, ratios, output comparisons and limits.
- `qwen36-prefill-final-legacy8k.json`, `qwen36-prefill-final-fused8k.json`,
  `qwen36-prefill-final-fused32k.json` and their `*-data/` folders retain all raw
  requests, outputs, resource samples and checked GPU work counts.
- 68 unit tests, 150 native Metal checks and 25 native process/RPC checks pass.
  Short independent FP32 comparison checks 120 layer states and all final logits;
  worst layer relative L2 is 2.7836e-6, and eight greedy IDs match. A separate
  65-token sequential-GPU differential fixture checks 200 states with worst
  relative L2 5.8435e-6. It is not an independent long MLX oracle.
- Four final real-RPC acceptance checks pass: distant 8K needle retrieval,
  rejection before prefill above 262,144 tokens, independent short greedy IDs
  after long cache use/rejection, and repeated fresh-state arithmetic. Raw:
  `artifacts/baselines/qwen36-fused-long-acceptance.json`.
- Native Host publish succeeds; native TUI publish/simulation PTY RPC smoke passes.
  TUI still uses the documented upstream IL2104 warning exception described below.

To reproduce the controlled 8K pair with no other model using the GPU:

```sh
artifacts/baselines/mlx-env/bin/python scripts/run-qwen36-context-suite.py \
  --cases 8192:128 --trials 3 --fixed-output --attention legacy --chunk 32 \
  --prompt-file artifacts/baselines/qwen36-context-fixed-data/8192-4096-prompt.txt \
  --prefix local-prefill-legacy8k
artifacts/baselines/mlx-env/bin/python scripts/run-qwen36-context-suite.py \
  --cases 8192:128 --trials 3 --fixed-output \
  --prompt-file artifacts/baselines/qwen36-context-fixed-data/8192-4096-prompt.txt \
  --prefix local-prefill-fused8k
```

The optimized full-window job uses a separate prefix, `qwen36-context-fused-full`,
and the exact earlier 245,760-token prompt. It completed on 2026-10-21 at
20:54:43 UTC, with the then-current 128-token / fused-attention implementation:

| Input | Fixed output | Prefill | Prefill tokens/s | Decode tokens/s | Complete HTTP |
|---:|---:|---:|---:|---:|---:|
| 245,760 | 16,384 | 2,228.387 s | 110.286 | 9.339 | 3,982.840 s |

The output intentionally ignores EOS and does not establish natural long-answer
quality. The engine reserved 262,144 slots / 10 GiB FP32 KV and reported
35,369,533,440 bytes of Metal allocations. System swap was already 7.39 GiB
before the request, peaked at about 10.01 GiB in sampled records, and was about
7.91 GiB near the end of prefill; these are system totals, not model-page
attribution. The supervisor reaped the benchmark cluster and verified its
restored idle service healthy. This measurement predates grouped expert GEMM
and chunkwise DeltaNet; do not relabel it as a result of those changes.

Raw report: `artifacts/baselines/qwen36-context-fused-full.json`; all outputs,
resource samples, prompt/token hashes and final job state retain that prefix.

## Earlier 32-token / materialized-score measurements (2026-10-21)

The table below preserves the baseline before the fused-attention optimization.
One sequential request at a time, one measured trial per row, no excluded
warmup. All rows are our own engine on the M5 Pro / 48 GiB. Fixed-output rows
explicitly ignore EOS; the natural report honors EOS. These are not controlled
three-trial MLX comparisons. First-token timings are worker internal.

| Input tokens | Actual output | Mode | First token (s) | Prefill tokens/s | Decode tokens/s | First → last decode window |
|---:|---:|---|---:|---:|---:|---:|
| 128 | 256 | fixed | 1.706 | 75.1 | 92.8 | 92.8 → 92.8 |
| 2,048 | 4,096 | fixed | 7.631 | 268.5 | 74.6 | 77.0 → 72.4 |
| 8,192 | 4,096 | fixed | 37.188 | 220.3 | 67.7 | 69.9 → 65.5 |
| 128 | 16,384 | fixed | 0.701 | 184.1 | 67.1 | 93.6 → 53.9 |
| 32,768 | 8,192 | fixed | 257.661 | 127.2 | 38.6 | 40.7 → 35.8 |
| 8,192 | 3,522 | natural EOS | 37.385 | 219.2 | 68.2 | 70.0 → 66.2 |

The first 128:256 request was the first measured request of this live worker;
its first-token result should not be compared as a warm-prepared-kernel latency
to the later 128:16384 request or the earlier 24-token short benchmark. The
16,384-output decode phase lasted 244.112 seconds. The 32K/8K request finished
in about 470 seconds; it reserved 65,536 cache slots (2.5 GiB KV) and reported
about 25.02 GiB of Metal allocations. System swap was already about 7.5 GiB
before testing; it did not increase in these completed runs.

The natural request asked for a detailed review of source/docs and allowed up
to 16,384 outputs; it actually ended at 3,522 with EOS. Its complete RPC lasted
89.038 seconds. The fixed 16K row establishes sustained computation at that
length, not a naturally generated 16K report or a quality result.

Summary exports: `artifacts/baselines/qwen36-context-summary.json`,
`qwen36-context-summary.csv`, and `qwen36-context-decode-windows.csv`.
Run `python3 scripts/summarize-qwen36-context.py` to refresh them; the managed
full-window job refreshes them automatically when it ends. Pending or failed
cases are kept separate from completed trials.

## Reproduce

Start one own-engine MoE worker, with no competing GPU model:

```sh
cd /path/to/Infer.NET
artifacts/native/osx-arm64/deepinfer demo --model qwen3.6-35b-a3b --stay
```

The current live cluster may already occupy port 5100; reuse it rather than
starting a second model. In another terminal, prepare the complete size matrix:

```sh
artifacts/baselines/mlx-env/bin/python scripts/benchmark-qwen36-context.py \
  --output artifacts/baselines/qwen36-context-plan.json
```

Run a natural-EOS workload, including an optional UTF-8 document/code corpus:

```sh
artifacts/baselines/mlx-env/bin/python scripts/benchmark-qwen36-context.py \
  --run --cases 8192:16384 32768:16384 \
  --corpus /absolute/path/to/documents.txt \
  --output artifacts/baselines/qwen36-context-natural.json
```

Omit `--corpus` to use a hashed snapshot of this repository's source and docs.
Source filenames and hashes, the exact saved UTF-8 prompt, its hash, all chat
IDs' hash and actual input token count are recorded. Material is truncated to
the input budget; when the source is too short, repetition counts are recorded.
A repeated repository corpus is a reproducible scaling workload, not a claim
of performance on a distinct 256K real-world document.

For a fixed-length throughput diagnostic:

```sh
artifacts/baselines/mlx-env/bin/python scripts/benchmark-qwen36-context.py \
  --run --fixed-output --cases 128:16384 32768:8192 \
  --output artifacts/baselines/qwen36-context-large-fixed.json
```

`--fixed-output` explicitly sends `benchmarkIgnoreEos=true`; model forwards
continue after EOS to complete the requested count. These outputs do not
establish natural completion length or quality. Default natural-EOS mode
records the actual output count and finish reason, without padding or pretending
that a short answer completed the requested budget.

The full plan covers 128:256, 2048:4096, 8192:4096, 32768:8192,
131072:8192, 245760:16384 and an explicitly unsupported 270000:8192.
`--trials 3` repeats each request; default is one exploratory trial, with **no
excluded warmup**. Model loading is outside request timings. Tokenization,
reset and any first-request cache growth are included in worker first-token
time; the prefill phase starts after reset/cache allocation. Service responses
are still non-streaming: worker first-token timing is not client-visible TTFT.
The complete HTTP latency is reported separately.

`--worker-pid PID` also samples that worker's RSS. All runs sample system-wide
swap, VM counters, load average and thermal status every five seconds. Swap
already in use before a run must not be attributed to this engine; compare the
before/after records. Metal allocation counters describe allocated buffers,
not an independent measurement of physical resident pages.

## Metrics and implementation

Reports contain first-token/prefill milliseconds, prefill tokens/s, aggregate
decode tokens/s, complete RPC latency, actual output length, finish reason,
and non-overlapping decode windows of 256 intervals. Each window records its
context length and elapsed time, exposing changes as KV history grows.
Window counts/times and actual kernel/submission counts are checked against
the completed generation, including the extra EOS-detection forward in natural
mode. Full output text/IDs remain in separately hashed raw result files.

All 256 experts have persistent Metal buffers, with no application-level expert offload; only chosen experts execute. macOS can still page memory under pressure. The long-context
change retains FP32 activations, recurrent state, accumulation and KV cache.
The ten full-attention layers need **40,960 KV bytes per capacity token**:

| Capacity tokens | FP32 KV alone |
|---:|---:|
| 8,192 | 0.3125 GiB |
| 32,768 | 1.25 GiB |
| 65,536 | 2.5 GiB |
| 131,072 | 5 GiB |
| 262,144 | 10 GiB |

Capacity grows in powers of two and is retained until model disposal. Thus a
32K input + 8K output reserves 65,536 slots. Weights, recurrent state, attention
scratch, runtime and OS memory are additional. With the optimized default 128-token chunk, short score scratch is bounded to
16 MiB, and partition statistics reach 516 MiB at maximum capacity; decode uses
the same partition scratch plus 16 MiB of scores. Allocation failure is reported rather than silently truncating history.

Long prefill now uses original FP32 tiled fused attention: QK, online softmax
and PV share bounded threadgroup tiles, and only 1024-key partition statistics
are written before their stable combination. Long scores are not materialized
in the default prefill path. The chunk is now 128 tokens; explicit chunk32 /
legacy-attention switches retain the previous implementation for comparison.
Decode keeps its preceding score/partition path. Full attention remains quadratic
in input length, and decode attention grows with existing history. Short-input
throughput cannot be extrapolated directly to 256K.

Other models retain their 32,768-character / 256-output limits. Qwen3.6 permits
8,388,608 UTF-16 prompt characters; the HTTP envelope is 64 MiB. Long inference
RPC deadlines are 24 hours, with cancellation checked between prefill chunks
and decode steps. No continuous batching, prefix reuse, speculative decoding,
vision, MTP, YaRN, expert offload or cross-device model partitioning is enabled.

## Validation

- Release build: zero warnings/errors; 68 unit tests pass.
- 60 Metal graph/projection/recurrence checks pass, including 65/128-token tails.
- 33 long-context hardware checks pass: 28 analytic legacy/fused causal attention
  cases across partition boundaries, the 2K boundary, large strides, and the final
  valid context position **262,143**; five reservation/statistics guards.
- 57 additional random fused/short attention checks pass against a double oracle,
  with varying query/key/value features and 1/3/5/17/32/65/128-token chunks. They
  also use and check the real bounded 128-token short score allocation. Worst
  absolute error is 4.86e-8 and relative L2 7.91e-7. These independent expectations
  do not use the preceding GPU attention implementation as an oracle.
- Independent short FP32 MLX numerical regression remains within tolerance:
  120 layer states, all final logits and eight greedy output IDs. The forced-fused short path checks 120 states with worst layer
  relative L2 2.7836043e-6 and maximum absolute difference 3.2424927e-5.
  This independent oracle is short-context only.
- 25 actual native process/RPC integration checks pass, including larger-envelope
  bounds, legacy-model limits, authentication, leases, retries and backpressure.
- Host Native AOT publish succeeds. TUI Native AOT publish succeeds using the
  existing documented `TreatWarningsAsErrors=false` option; upstream
  `Microsoft.AspNetCore.Components` still reports IL2104. A strict TUI publish
  without that documented option fails on this upstream warning.

Independent whole-model long-context token identity/quality is not established
by the primitive oracle. Near-tied MoE routing can amplify small differences in
floating-point reduction order, as documented in the earlier benchmark report.

Raw logs: `artifacts/long-context-*.log`. Reports and exact workloads:
`artifacts/baselines/qwen36-context*.json` and corresponding `*-data/` folders.

## Managed full-window run

The supervised runner requires its two ports to be free and never kills a
pre-existing service. It starts one model, runs the benchmark, saves job/result
states, and reaps its owned cluster. With `--restore-live` it starts a fresh idle
cluster afterwards, freeing the grown long-context buffers even on failure or
cancellation:

```sh
artifacts/baselines/mlx-env/bin/python scripts/run-qwen36-context-suite.py \
  --fixed-output --restore-live --keep-awake --prefix qwen36-context-full
```

The optional `--keep-awake` holds an idle-sleep assertion for the managed job only; it is released after cleanup.

Default cases explicitly flag unsupported 270000:8192 and then measure
245760:16384 (262,144 combined). The full run can take a long time, including with fused attention. Its actual completion time/speed must come from the result,
not extrapolation of the shorter measurements above.

Live job state: `artifacts/baselines/qwen36-context-full-job.json`.
Request/results: `artifacts/baselines/qwen36-context-full.json`.
Progress: `artifacts/baselines/qwen36-context-full-cluster.log`.
Memory samples are flushed every five seconds to `qwen36-context-full-data/245760-16384-trial1-memory.jsonl`; the finished report also records its path.
The supervisor PID is recorded in the job state. Sending it SIGTERM cancels the
benchmark, records cancellation and restores the idle service; normal completion
and active-decode cancellation/restore have both been tested with real processes.

Additional whole-model checks passed after the 32K/8K run: retrieve a 12-digit
needle from the start of an 8,247-token prompt; reject an over-native-limit token
input with HTTP 400 before prefill; reproduce the independent short greedy IDs
after long cache use; repeat `2+2` with fresh state and identical output IDs.
These are useful acceptance samples, not a comprehensive long-context quality
evaluation. Raw results: `artifacts/baselines/qwen36-context-acceptance.json`.

The original unfused full-window job was cancelled to prioritize prefill
optimization after reaching 61,440 input tokens in 728.5 seconds. Its job/result
files retain the cancelled state. Cache allocation had succeeded, and system
swap increased from its pre-existing baseline; this is not a completed
performance or quality result. Never combine its partial progress with the
optimized measurements. Consult each job/result file for its actual state.
