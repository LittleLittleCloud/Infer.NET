# M5 prefill: expert tails, fused gate/up and direct DeltaNet loads

The selected M5 launcher profile uses adaptive expert output tiles, one combined
gate/up matrix product and direct FP32 DeltaNet matrix loads. Product inference
remains the original .NET/Metal executor. MLX is only an independent benchmark.

## Implementation

`prefill_expert_packed.mm` retains the 64-route packing format and 32-output-row,
64-inner-element quantization tile. The last routing tile selects a 16, 32 or
64-column matrix operation according to its actual route count. The gather,
scatter, router decisions and buffer capacities are unchanged.

Gate and up weights occupy the two halves of a 64-row matrix. One matrix
operation consumes their shared input for each K tile. After the final K step,
the kernel reuses the same 8 KiB threadgroup weight workspace for the up result;
gate values remain in registers for FP32 SwiGLU. The resulting activation keeps
the existing FP32 route layout and separate BF16 packing before down. This is
different from the previous direct-BF16 intermediate experiment.

`prefill_delta.mm` loads W, U and causal matrices directly with SIMD matrix
loads and transpose operands. Complete 32-token output blocks use transposed
matrix stores directly into the output buffer; partial blocks retain guarded
stores. This removes redundant threadgroup copies and barriers. State remains
in FP32 matrix registers across blocks. The forward triangular solve, recurrence,
FP32 arithmetic, convolution history and normalization are unchanged.

No new persistent buffers, weight repacking, quantization changes, cached answers,
prefix reuse or model-runtime dependencies are introduced by the selected path.
Decode kernels and the 1024-key attention partition remain unchanged. BF16 GEMM
operands and attention/KV retain the previous profile's precision boundaries;
residuals, normalization, DeltaNet, routing and accumulators remain FP32.

## Measurements

Final results and audited evidence are in
`artifacts/baselines/qwen36-joy-summary.json`. Same M5 Pro / 48 GiB, AC power and
High Power mode; exact same five-bit checkpoint and saved 8192/32768-token
prompts. Each engine and length has three excluded complete warmups and three
measured requests, batch one, 128 fixed outputs with EOS ignored. Decode times
cover 127 intervals. Model loading is excluded. GPU workloads run serially.
The own A/B uses the same published executable and Metal library; order reverses
at 32K. Exact input hashes, all raw own payload hashes, output IDs, power settings,
allocation counters and validation evidence are audited by the summary script.

| Input | Control prefill | Selected prefill | Reduction | Fresh MLX prefill | Own / MLX decode |
|---|---:|---:|---:|---:|---:|
| 8192 | 3.468 s | 3.353 s / 2443.5 tokens/s | 3.32% | 2.967 s | 84.76 / 83.92 tokens/s |
| 32768 | 17.850 s | 17.344 s / 1889.3 tokens/s | 2.83% | 14.701 s | 74.88 / 71.56 tokens/s |

**Prefill parity has not been reached.** The selected implementation still takes
12.99% / 17.98% longer than fresh MLX at 8K / 32K.

All 24 own warmup/measured requests retain the exact same 128 output IDs for
their input length. The 88 own power observations retain AC and identical
High Power settings. Measured own requests have zero system swap-out pages;
small system swap-ins (0–72 pages per request) are recorded, without assigning
them to the worker. Allocated Metal memory and dispatch/submission counters
are identical across own A/B.

MLX prompt timing includes the first generated token; our prefill timing excludes
first-emission overhead. Both TTFT values are retained. Own A/B output identity
is checked; numerical or output equivalence across engines is not asserted.
These results do not establish full-window performance or long-output throughput.
Existing system swap allocation is not evidence of active swapping; the summary
records changes in system-wide swap-in/out page counters during measured requests.

## Reproduction and selection

`DeepInferMoE.command` selects the new flags for the standard M5 hybrid/packed
profile, respecting explicit overrides. Direct Host invocations retain existing
defaults. With the usual BF16 hybrid profile, the new switches are:

```sh
export DEEPINFER_METAL_PREFILL_EXPERT_TILES=adaptive
export DEEPINFER_METAL_PREFILL_EXPERT_GATE=fused
export DEEPINFER_METAL_PREFILL_DELTA_IMPL=device
```

Control settings are `fixed`, `split`, and `staged`. The selected profile keeps
`EXPERT_FLOW=staged`, `EXPERT_PIPELINE=serial`, and attention partitions at 1024.
Run `artifacts/baselines/mlx-env/bin/python scripts/sweep-qwen36-joy.py` with
port 5100 free for the serial same-binary A/B. It checks AC power and unchanged
power settings. Run `scripts/summarize-qwen36-joy.py` after the corresponding
MLX benchmarks and validation logs exist to audit and summarize the evidence.

## Validation and rejected experiments

- Independent double oracle for grouped/shared expert GEMM: 56 cases covering
  routing skew, hot/scattered experts, tails and 64–2048 inputs.
- 21 expert-flow cases retain the prior output fingerprint `c281d050974aeabd`.
- Independent double recurrence oracle: 60 DeltaNet cases, including zero/one
  decay, underflow, nearly identical keys, short fallback, block tails, state
  and convolution history.
- BF16 attention numerical and native-window analytic checks; attention is
  unchanged in the selected profile.
- Native AOT build, 68 unit tests, 25 process/RPC checks and four real-model RPC
  checks (distant needle, context rejection, independent short IDs and reset).

A 2048-key attention partition was implemented as an explicit experimental
option (`DEEPINFER_METAL_PREFILL_ATTENTION_PARTITION=2048`). It changes the BF16
online-softmax rounding path and changed the 8K generated sequence starting at
output token 41. It did not improve the later whole-model comparison enough to
select it. Default 1024 remains. Larger query tiles and larger gate execution
groups also failed to provide a general isolated advantage.

A full-precision MPP DeltaNet recurrence passed all 60 oracle cases, but the 8K
pilot improved only about 0.2% versus direct SIMD loads while requiring 64 MiB
of extra workspace. It was removed from product code; source and binary remain
under `artifacts/reference/delta-nax-rejected/` and
`artifacts/native/delta-nax-pilot/`. Earlier direct expert intermediate and
software double buffering remain unselected; see [expert flow](EXPERT-FLOW.md).
Pilot numbers are diagnostic and are not substituted for the final same-binary
comparison.
