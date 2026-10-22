# Attention dispatch batch verification: M5 Pro

The outer model prefill chunk is **2048 tokens** in both our measured profile
and the MLX comparison script. The previous 128-token value is a batch of
queries encoded into one attention dispatch, not the model prefill chunk.

| Scope | DeepInfer selected configuration |
|---|---:|
| Model prefill chunk | 2048 input tokens |
| Attention query dispatch batch | 128 queries, configurable up to 2048 |
| Attention compute query tile | 4 queries × 8 GQA heads |
| Inner key tile | 128 keys |
| Key reduction partition | 1024 keys |

All attention dispatch batches for one outer chunk share its command buffer.
Increasing 128 to 1024 reduces dispatch count, but does not remove a CPU wait
per 128 queries: those waits were already absent. Our 8K/128-output requests
still use 131 command buffers; 32K requests use 143.

Upstream [MLX v0.32.4 attention dispatch source](https://github.com/ml-explore/mlx/blob/v0.32.4/mlx/backend/metal/scaled_dot_product_attention.cpp#L200-L206)
selects a 64-query, 32-key tile on the expected M5 NAX BF16 path for head
dimension 256. This is a source-derived interpretation, not a Metal capture.
Neither value is a 1024-token outer prefill chunk.

## Implementation and verification

`DEEPINFER_METAL_PREFILL_ATTENTION_BATCH=128|256|512|1024|2048` controls query
packing, dispatch iteration, and partial-result allocation. Values above 128
require BF16 fused attention. The allocation is also enlarged on context
reservation. Shader arithmetic, compute geometry, precision, weights and
public interfaces are unchanged. The product default remains 128.

Native AOT/Metal publication succeeds. Verification for this change passes:

- 23 real GPU checks compare a single large batch with composition of original
  128-query calls, including offsets, guards, reservation and invalid options.
  All tested outputs are bit-identical.
- 69 independent double-oracle attention cases pass unchanged tolerances.
- 20 analytic BF16 cases at batch 2048, 12 at batch 128, and 33 original FP32
  long-context/reservation checks pass, through cache position 262143.
- 68 unit tests, 25 process/RPC checks and four real-model acceptance checks pass.

The maximum-window checks exercise operator bounds and analytical inputs;
they are not a maximum-window full-model quality benchmark. Large batches
need more partial-result storage; at 32K, 512 adds 390 MiB and 1024 adds 910 MiB
relative to 128. Buffer allocation does not measure physical residency.

## Measurement protocol

The same published executable/library, five-bit checkpoint, exact saved
8K/32K prompts and fixed 128 outputs are used throughout. Greedy generation
continues past EOS for this throughput diagnostic. Prefix reuse is disabled;
model loading is excluded. All GPU workloads run serially.

The initial pilot tested all five batches. The first confirmation round had
between-round drift and battery power was observed; its timing cannot establish
an AC comparison with earlier MLX results. A further three-warmup 32K battery
round found only 0.49% lower mean prefill time for 512, with 390 MiB extra buffers.
All historical reports and limitations remain in
`artifacts/baselines/qwen36-attention-batch-summary.json`.

The separate AC rerun uses **three excluded full warmups and three measured
requests for every batch and input length**. AC is required before/after each
request and sampled every five seconds during generation. This does not measure
GPU clocks, and sequential batches can still experience background-load or
temperature differences. The AC reports are not a new MLX comparison.

| Query batch | 8K mean prefill | 32K mean prefill | Additional 32K buffers |
|---|---:|---:|---:|
| 128 | 3.431 s / 2387.5 tokens/s | 17.856 s / 1835.1 tokens/s | 0 |
| 512 | 3.453 s | 17.643 s | 390 MiB |
| 1024 | 3.459 s | 17.717 s | 910 MiB |

There are 18 excluded warmups and 18 measured requests, with **132 AC power
observations**, all matching AC attachment. All 128 output IDs match across
every batch, warmup and trial for each input length. The three 128-batch 8K
prefill times are 3431.2834 / 3429.5656 / 3432.5484 ms; 32K times are
17711.9659 / 18041.3323 / 17815.4680 ms. The default's mean decode throughput
is 86.13 tokens/s at 8K and 74.90 at 32K, measured across 127 generation
intervals per request.

512/1024 are 0.64%/0.81% slower at 8K and 1.19%/0.78% faster at 32K in this
round. These small, length-dependent changes do not establish a broad speedup,
especially given sequential ordering and the 128-batch 32K variation.
**Keep the default at 128. Enlarging the dispatch batch alone does not resolve
the prefill gap.** AC absolute times are retained separately from prior battery
and MLX measurements; no new code-speedup or MLX parity claim is made.

## Reproduction and evidence

Stop the owned idle cluster first; the runner refuses to replace an existing
service. Leave other GPU inference idle.

```sh
artifacts/baselines/mlx-env/bin/python scripts/sweep-qwen36-attention-batch.py \
  --batches 128 512 1024 --warmup 3 --trials 3 --tag ac --require-ac
python3 scripts/summarize-qwen36-attention-ac.py
```

Reports are `artifacts/baselines/qwen36-ab{128,512,1024}-{8,32}k-ac.json`, with
raw RPC payloads and power/VM samples in their `-data/` directories. The summary
audits hashes, current binary identity, matching prompt/output IDs, actual GPU
work counters and AC observations. Operator logs use
`artifacts/attention-batch-{checks,oracle,max-context,default-context,fp32-context}.log`.
The default worker was restored on port 5100 and passed four fresh real-model
checks after the AC sweep, recorded in
`artifacts/baselines/qwen36-attention-batch-ac-acceptance.json`.

