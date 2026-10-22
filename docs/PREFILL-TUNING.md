# Expert and attention tuning: M5 Pro, 2026-10-21

DeepInfer.NET now uses smaller expert inner tiles and a wider attention
pipeline for prefill. Product inference remains our original .NET/Metal
executor. MLX is an independent benchmark process.

The subsequent [attention dispatch-batch verification](ATTENTION-BATCH.md)
distinguishes the 2048-token outer model chunk from the 128-query attention
dispatch batch. It includes a separately recorded AC-powered rerun. Historical
absolute timings below are not a new power-controlled comparison.

## Selected implementation

- Expert gate/up retains 32 output rows and 64 routed inputs, reduces the inner
  dimension from 128 to 64 and uses two SIMD groups (64 threads). Expert down
  uses the same tile with one group (32 threads). Dispatch sizes match the MPP
  execution scope. Gate/up threadgroup weight storage falls from 16 to 8 KiB;
  down falls from 8 to 4 KiB.
- Four-query, 128-key BF16 attention uses 16 SIMD groups (512 threads) for at
  least 32 queries. This reduces each thread's live FP32 matrix fragments.
  Decode and tiny batches retain four groups. Other query/key configurations
  retain their original pipeline. Query packing, causal masking, partition
  layout and cache bounds are unchanged.
- Weight storage, BF16 operand rounding and FP32 accumulation are unchanged
  from the previous hybrid profile. Residuals, normalization, DeltaNet state
  and arithmetic, routing probabilities and decode GEMV remain FP32. All
  libraries retain safe math. No additional persistent workspace is needed.

The existing M5 `DeepInferMoE.command` launcher selects the updated kernels
through its existing hybrid/packed/BF16 profile. No new flags are necessary.
Direct Host defaults remain FP32. See [hybrid configuration](PREFILL-HYBRID.md)
for explicit commands and precision boundaries.

## End-to-end measurements

Same M5 Pro 20-core GPU / 48 GiB, same five-bit checkpoint, exact saved prompts
and token-ID hashes. Batch one, greedy fixed 128 outputs with EOS ignored,
one excluded complete warmup and three measured requests per engine and length.
Engines run serially. Model/shader loading is excluded; prefix reuse is off.

| Input | Previous own prefill | Updated own prefill | Fresh MLX prefill | Own / MLX decode |
|---|---:|---:|---:|---:|
| 8192 | 3.734 s | 3.491 s / 2346.3 tokens/s | 2.962 s / 2765.9 tokens/s | 84.70 / 84.37 tokens/s |
| 32768 | 19.917 s | 17.979 s / 1822.6 tokens/s | 14.666 s / 2234.3 tokens/s | 74.84 / 71.46 tokens/s |

Own prefill time falls **6.51% at 8K and 9.73% at 32K**. The updated 8K trials
are 3.49136 / 3.49078 / 3.49204 seconds; 32K trials are
17.97476 / 17.98670 / 17.97580 seconds. Decode reports 127 intervals after first
emission. Own outputs are deterministic across the three measured requests;
output-ID identity across engines is not asserted.

**Prefill still takes 17.88% and 22.59% longer than this fresh MLX baseline.
Prefill parity has not been reached.** First-token wall timers remain in the
evidence: own prefill excludes a small first-emission cost, while MLX's prompt
timer includes producing the first token. Different activation precision also
prevents treating this as an identical numerical implementation.

## Validation and rejected experiments

Native AOT publishes without warnings/errors. Published executable and Metal
library hashes match both selected three-trial reports. Verification passes:

- 68 unit tests and 25 real-process/RPC checks.
- 511 Metal operator checks, including independent double oracles, expert
  routing/tails, operand rounding, original FP32 paths and DeltaNet.
- Twelve analytic BF16 attention cases cover one, three and 128 queries through
  the final native cache position, 262143. The 128-query cases exercise the new
  wide pipeline.
- Four real-model RPC checks: an 8K distant needle, native-limit rejection,
  independent short greedy IDs after long cache use, and repeated fresh-state
  arithmetic. The updated worker is restored on port 5100 for the CLI/TUI.

At position 4096 with 2048 input tokens, split-command diagnostics measure
expert gate/up at 259.2 -> 220.9 ms, expert/shared down at 148.1 -> 119.6 ms,
and attention at 147.9 -> 110.0 ms. Explicit boundaries perturb scheduling and
cache residency; these intervals are not production throughput. Experts remain
the largest measured stage, followed by projections and DeltaNet.

Smaller attention feature slices were slower and were reverted. Static padded
expert operands and static interior attention branches regressed the 8K model
result to 3.563 s and were reverted. Relaxed math had no consistent isolated
benefit and was not selected. Alternative sources/results remain under
`artifacts/reference/` and `artifacts/baselines/`.

No new full-window model generation, multi-thousand-output benchmark or broad
quality equivalence is claimed. Native-window operator checks establish bounds
and numerical cases rather than full-model quality at maximum context.

## Evidence and reproduction

`artifacts/baselines/qwen36-tuned-summary.json` audits raw result hashes, exact
prompt hashes, warmup/trial counts, published binary hashes and validation logs.
Fresh MLX reports include runtime module hashes matching the installed
distribution's RECORD entries. Versions are MLX 0.32.4 and mlx-lm 0.32.2.

Matched reports: `qwen36-tuned-release{8k,32k}.json` and
`qwen36-tuned-mlx{8k,32k}.json`; raw own requests are in the corresponding
`-data/` directories. Acceptance is `qwen36-tuned-acceptance.json`. Final
validation logs use `artifacts/tuned-final-*.log`.

```sh
artifacts/baselines/mlx-env/bin/python scripts/run-qwen36-context-suite.py \
  --gemm bf16-nax --gemm-impl hybrid --expert-impl packed \
  --attention-impl bf16-nax --kv bf16 --key-tokens 128 \
  --warmup 1 --trials 3 --cases 8192:128 --fixed-output \
  --prompt-file artifacts/baselines/qwen36-context-fixed-data/8192-4096-prompt.txt \
  --prefix qwen36-tuned-release8k
python3 scripts/summarize-qwen36-tuning.py
```

The runner refuses to replace an existing service; stop the owned idle model
first and run GPU jobs serially. The summary also requires the completed 32K,
MLX and validation reports listed above. Refresh measurements before auditing
a different native binary. `scripts/benchmark-metal-experts.sh` and
`scripts/benchmark-metal-bf16-attention.sh` are synthetic geometry sweeps,
not end-to-end throughput benchmarks.
