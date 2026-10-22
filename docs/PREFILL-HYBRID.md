# Prefill: hybrid expansion, packed expert input and BF16 attention

These measurements describe the preceding hybrid revision. See [current expert/attention tuning](PREFILL-TUNING.md) for the updated kernels and fresh matched measurements.

The optimization keeps the product on our original .NET/Metal executor.
MLX is used only in a separate, serial baseline process.

## Changes

- Diagnostic command-buffer interval profiling isolates a single 2048-token
  chunk at input position 4096. It identifies attention, projections and expert
  computation as the dominant stages. Splitting commands perturbs execution;
  these diagnostic intervals are never reported as production throughput.
  GPU submission accounting includes every diagnostic command buffer.
- Dense projections (including the 32-row decay/beta projections) and shared
  gate/up expand affine weights to BF16 once into bounded reusable workspace.
  MPP then consumes complete device-tensor matrices. Shared gate/up fuses SwiGLU.
  The normal expert path retains packed 5-bit storage and does not expand all
  experts into a persistent dense cache.
- Expert inputs are gathered/rounded once into compact 64-route BF16 tiles.
  Original expert GEMM uses 32 output rows and a 128-column inner tile, keeps
  dequantized weights in bounded threadgroup memory, and consumes device inputs.
  Output accumulators remain FP32, and scatter uses the original token/slot IDs.
- New original BF16 attention packs queries by KV-head group, consumes BF16
  Q/K/V through MPP, and retains FP32 online softmax, partial sums and output
  accumulators. Its matrix probability operands are rounded to BF16. Four query
  tokens and eight GQA heads share each 128-key tile. Query workspace remains
  bounded by the outer 128-query attention subtile. The cache limit is unchanged.
- Full expert expansion and cooperative-register dequantization were explored.
  They were not selected for the final low-memory configuration. The former
  costs over 1 GiB of workspace for little expert improvement; the latter
  regressed end-to-end timing. Experimental sources/results remain under
  `artifacts/reference/` and `artifacts/baselines/`.

The selected hybrid workspace is approximately 162 MiB (32 MiB expanded dense
weights, 2 MiB shared up weights, up to 128 MiB packed expert input), plus a
1 MiB BF16 query buffer. This is allocation, not physical residency. Residuals,
normalization, DeltaNet state/arithmetic, routing probabilities and decode
GEMV remain FP32. This is a mixed-precision implementation, not a full BF16
replica of MLX; precision changes can alter generated IDs.

## Run and reproduce

```sh
DEEPINFER_METAL_PREFILL_GEMM=bf16-nax \
DEEPINFER_METAL_PREFILL_GEMM_IMPL=hybrid \
DEEPINFER_METAL_PREFILL_EXPERT_IMPL=packed \
DEEPINFER_METAL_PREFILL_ATTENTION_IMPL=bf16-nax \
DEEPINFER_METAL_PREFILL_KEY_TILE=128 \
DEEPINFER_METAL_KV=bf16 \
./artifacts/native/osx-arm64/deepinfer demo --model qwen3.6-35b-a3b --stay
```

Requires M5 and macOS 26+. The existing CLI/TUI uses the coordinator normally.
The ordinary launch precision defaults remain unchanged; flags apply when the
worker loads the model. Stop any existing model service before launching another.

The managed benchmark runner accepts the corresponding `--gemm bf16-nax`,
`--gemm-impl hybrid`, `--expert-impl packed`, `--attention-impl bf16-nax` and
`--kv bf16 --key-tokens 128` options. `--warmup 1 --trials 3` records the warmup separately and
excludes it from measured trials. Exact saved prompt files and token hashes
allow comparison with `benchmark-qwen36-mlx.py --prompt-file ...`.

Validation includes independent double-precision product oracles with BF16
operand rounding, shared experts, input/output tails, route coverage and
indirect grids; an independent online-attention reference rounds both operands
and block probabilities; native-context analytic attention reaches position
262143. These operator checks do not establish long-model quality equivalence.
The full model acceptance checks an 8K distant needle, native-limit rejection,
short reference greedy IDs and repeated state-reset arithmetic. A new full
262144-token model generation and broad quality evaluation are not claimed.

Final measurements and native binary hashes are recorded in
`artifacts/baselines/qwen36-hybrid-summary.json`. Warmups, precision boundaries,
input hashes, fixed-output policy, baseline versions and limitations must be
read together with throughput. No MLX-parity claim follows from a different
prompt, dtype or warmup convention.

The same fused attention path is used for decode when BF16 attention is selected. `DEEPINFER_METAL_DECODE_ATTENTION=legacy` is a diagnostic override. The M5 `DeepInferMoE.command` launcher now selects this tested profile when the user has not explicitly chosen a GEMM precision; direct Host CLI defaults remain FP32.


## Final M5 Pro 48 GiB measurements

Same exact saved prompt files and token IDs, same packed 5-bit weights, batch
one, greedy fixed 128 outputs with EOS ignored, one excluded full warmup and
three measured requests per engine and length. Engines run serially. Model and
shader loading are excluded. No prefix reuse is used.

| Input | Own prefill | MLX prefill | Own decode | MLX decode |
|---|---:|---:|---:|---:|
| 8192 | 3.734 s / 2193.7 tokens/s | 2.924 s / 2801.3 tokens/s | 86.34 tokens/s | 85.50 tokens/s |
| 32768 | 19.917 s / 1645.2 tokens/s | 14.446 s / 2268.4 tokens/s | 76.24 tokens/s | 72.82 tokens/s |

Own 8K prefill trials are 3.732 / 3.733 / 3.738 seconds. The final-binary control
uses the previous staged BF16 GEMM/KV configuration, FP32 attention, and the same
warmup/trial convention: its mean is 11.485 seconds, a **3.08x** prefill speedup.
The same final executable/library hashes match the control and both new cases.
Own 32K trials are 19.915 / 19.914 / 19.922 seconds. Decode uses the 127 intervals
after first emission; aggregate rates above are ratios of total tokens to time.
**Decode matches or slightly exceeds this MLX baseline. Prefill remains 27.7%
and 37.9% slower in time, respectively. No prefill parity is claimed.**

MLX versions and model revision are recorded in the matched reports. Native
activation/residual precision remains different from MLX, and output ID identity
across engines is not asserted. The own prefill timer excludes a small
first-emission overhead; both full first-token wall timers are retained in the
summary so that timing boundaries remain inspectable.

Validation passes: 68 unit tests, 447 selected native operator checks, 25
process/RPC checks, and four real-model acceptance checks. The original FP32
path still passes 120 independent layer states and 248320 final logits with
unchanged thresholds and eight exact greedy IDs. BF16 attention is checked
against an independent online reference at the actual 128-key block size, and
analytic cases cover the last native cache address. Native AOT publish logs
contain no warning/error. The selected fast worker is restored at port 5100;
the existing CLI/TUI can use it. The restored service also passes a real 2+2 RPC.

Evidence: `artifacts/baselines/qwen36-hybrid-summary.json`,
`qwen36-final-control8k.json`, `qwen36-key128-release{8k,32k}.json`,
`qwen36-hybrid-mlx{8k,32k}.json`, their raw results and job logs,
`qwen36-key128-acceptance.json`, `artifacts/hybrid-*-oracle.log`,
`artifacts/*key128*-checks.log`, `artifacts/hybrid-final-*.log`,
`artifacts/hybrid-fp32-*.log`, and `artifacts/hybrid-live-smoke.json`.
