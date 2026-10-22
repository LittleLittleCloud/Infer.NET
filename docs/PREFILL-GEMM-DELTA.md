# Original grouped expert GEMM and chunkwise Gated DeltaNet

This change implements the two missing batch paths in the own Metal executor.
No MLX/llama.cpp inference or external matrix library is linked or invoked.
Packed affine weights remain 5-bit/group64 (8-bit routers), with FP32
activations, matrix arithmetic, accumulation, recurrent state and KV.

## Expert grouping and GEMM

The GPU Top-8 router retains each token's own expert choices and probabilities.
A GPU-only compaction pass builds an expert-to-route list, where a route is the
original `(token, slot)` index. Experts selected by several tokens share a
16-token matrix tile. A second tiny GPU kernel builds indirect dispatch grids,
so the gate/up and down kernels launch only the occupied expert tiles. No
CPU readback or host sorting is used. Gate/up is fused with SwiGLU; down gathers
the corresponding route activation. Results scatter to the original slots;
the preceding weighted merge therefore retains exactly the same meaning.

Each threadgroup covers 64 output rows and 16 routed tokens, with a 32-column
inner tile. Eight adjacent 5-bit codes are unpacked together, and unpacked
weights are reused across the routed token tile. All sixteen expert inputs use
FP32 `simdgroup_float8x8` matrix operations. Tiny tails use zero padding, with
explicit row/route bounds. Unsupported expert precision retains the vector path.

## Chunkwise gated-delta rule

For one value head, the preceding sequential update is

```
S_t = a_t S_(t-1) + u_t k_t^T
u_t = beta_t (v_t - a_t S_(t-1) k_t)
o_t = S_t q_t
```

Within each 32-token block, define `A_t = product(a_0..a_t)` and
`D[t,j] = product(a_(j+1)..a_t)` for `j <= t`. Form the strictly lower matrix
`L[t,j] = beta_t D[t,j] dot(k_t,k_j)` for `j < t`. Then

```
H = inverse(I + L)
W = H diag(beta * A) K
U = H diag(beta) V
corrections = U - W S_start^T
O = diag(A) Q S_start^T + ((Q K^T) * D) corrections
S_end = A_last S_start + corrections^T diag(D[last,:]) K
```

The first kernel independently prepares all head/block matrices. Inversion uses
32 independent columns of a unit-lower-triangular forward solve. Direct products
of at most 32 decays avoid divisions by an underflowed cumulative decay. This
handles zero decay, unit decay and zero/full update gates.

The second kernel uses parallel FP32 matrix products for corrections, outputs
and state updates. Each threadgroup owns 32 value rows; the 128-column state
stays in matrix registers across blocks. Only the boundaries between 32-token
blocks remain recurrent. This is chunkwise parallelism, not an assertion that
the complete recurrent sequence executes independently in parallel. Convolution
history is updated once, after the block sequence, with the final three inputs.

The algebra follows the chunkwise/WY view in
[Gated Delta Networks (ICLR 2025)](https://arxiv.org/abs/2412.06464).
The shaders and dispatch implementation are original. An initial nilpotent
Neumann-expansion inverse was rejected: correlated keys produced large
cancelling powers and failed whole-model numerical tolerance. The stable
forward solve replaces it; tolerances were not relaxed.

## Larger outer prefill blocks and bounded attention workspace

Explicit outer chunk choices now include 256, 512, 1024 and 2048, in addition to
32/64/128. This is independent of the 32-token delta block and 16-route expert
tile. Full-attention queries are processed in tiles of at most 128, including
short attention and the legacy diagnostic path. Query, raw-gate and output
offsets and causal positions follow each tile. At native maximum context,
partition workspace remains at most 516 MiB; a 2048-token outer block no longer
requires an 8 GiB partition workspace. KV still depends on the total context.
Large trace buffers are allocated only during explicit tracing.

## Current defaults and remaining work

Default outer chunk is 2048; `grouped` experts and `chunked` DeltaNet are enabled.
For chunks shorter than 64 tokens the preceding vector/sequential paths remain.
Large outer chunks can increase cancellation latency: an already submitted
whole-chunk GPU command completes before cancellation is observed.

Shared-expert gate/up retains the preceding four-vector path; shared down and
ordinary projections already use tiled matrix kernels. Dense token tiles remain
at most 32. No explicit Metal 4 Neural Accelerator kernels are introduced.
Full-attention work is still quadratic in context length. These remaining
limitations must not be confused with having reached MLX prefill performance.

## Validation and controls

- `scripts/check-metal-prefill-experts.sh` checks the actual grouped/indirect
  route path against a double oracle: hot, scattered and skewed experts;
  64..2048 tokens; row and token tails; gate/up/SwiGLU and down; exact coverage,
  indirect grid counts, and reset of a deliberately stale GPU counter.
- `scripts/check-metal-prefill-delta.sh` checks every output, the complete final
  recurrent state and convolution history against a double sequential recurrence,
  including zero/unit/tiny decays, update gates 0/1, strongly correlated keys,
  nonzero initial states and lengths around 32/64/128/512 boundaries through 2048.
- The existing independent short FP32 oracle and the separate 65-token
  sequential-GPU differential fixture retain their original numerical thresholds.
  The latter is not an independent long-context MLX oracle.

Diagnostic environment switches select each implementation independently:

```
DEEPINFER_METAL_PREFILL_EXPERTS=vector|grouped
DEEPINFER_METAL_PREFILL_DELTA=sequential|chunked
DEEPINFER_METAL_PREFILL_CHUNK=32|64|128|256|512|1024|2048
```

The managed context benchmark runner accepts matching `--experts`, `--delta`
and `--chunk` options. It records these settings and checks the exact completed
GPU work counts, including each attention query tile, grouping/indirect grids
and chunkwise-delta preparation. Timings and output-identity limits are recorded
with the final measurements below.

## Completed measurements — M5 Pro 20-core GPU / 48 GiB, 2026-10-21

One GPU model at a time, one request at a time, exact saved 8192-token prompt,
128 fixed outputs (EOS explicitly ignored), no prefix cache, no excluded warmup.
The three controlled variants use the same candidate Host/Metal binary hashes;
only the explicitly recorded environment switches differ. The first trial in
each group is included. Loading/shader preparation are outside request timing.

| Outer chunk | Selected experts | DeltaNet | Three prefill trials (s) | Mean (s) |
|---:|---|---|---|---:|
| 128 | vector | sequential | 27.287 / 26.205 / 26.209 | 26.567 |
| 2048 | grouped GEMM | sequential | 16.952 / 15.627 / 15.639 | 16.072 |
| 2048 | grouped GEMM | chunkwise | 16.700 / 15.429 / 15.429 | 15.853 |

The extra end-to-end change from chunkwise DeltaNet is only about **1.37%** in
this comparison; it is not claimed as a significant independent speedup. The
large measured gain is from grouped GEMM plus larger outer batches. Earlier
one-trial ablations recorded 27.425 s at vector/128, 27.009 s at vector/512,
17.712 s at grouped/512, 17.672 s at grouped+chunkwise/512, 16.923 s at 1024
and 16.556 s at 2048. Increasing the batch alone was insufficient.

After selecting defaults, the final published Native AOT binary was remeasured:

| Input / output | Previous prefill | Published prefill | Published prefill rate | Speedup |
|---|---:|---:|---:|---:|
| 8192 / 128 | 26.567 s, three-trial control | 15.873 s, three trials | 516.1 tokens/s | 1.67x |
| 32768 / 128 | 126.731 s, one historical trial | 81.959 s, one trial | 399.8 tokens/s | 1.55x |

Published 8K trials are 16.824 / 15.394 / 15.400 seconds. Its aggregate decode is
70.06 tokens/s, versus 70.12 for the control; this is not a decode speedup.
32K decode over 127 intervals is 40.68 tokens/s. Both sizes reproduce all 128
output IDs of their respective controls; within each three-trial group output
IDs are identical. The intermediate grouped/sequential ablation differs from
the combined path at output 60, so exactness is not asserted for every
floating-point reduction configuration or arbitrary long prompt. The 32K
comparison uses the exact same prompt, output budget and 65536-slot cache as
its historical control, but is weaker evidence than a repeated paired A/B.

The final 8K request completes 91326 kernels / 131 submissions, versus
130006 / 191 for the 128-token control. Reported Metal allocation is about
23.54 GiB versus 23.04 GiB; larger activation scratch costs about 0.50 GiB even
though attention workspace is bounded. This counter is not physical residency.
The 32K request reports about 25.51 GiB of Metal allocations and 2.5 GiB FP32 KV.

Final checks:

- Release build and Host Native AOT publish succeed, with no warning/error in
  the captured logs; 68 unit tests pass.
- **270 native Metal checks** pass: 60 existing graph/projection/recurrence,
  33 analytic long-attention/reservation, 69 random attention/offset/short/legacy,
  60 chunkwise delta, and 48 grouped expert/indirect dispatch checks.
- Delta against a double recurrence: max output error 1.61e-7, max final-state
  error 3.71e-7, worst relative L2 1.85e-6, including identical adjacent keys.
  Grouped experts against a double oracle: max error 2.59e-8.
- Independent short FP32 oracle: 120 states and 248320 final logits within
  unchanged thresholds, eight greedy IDs exact; worst state relative L2 2.79e-6.
- The 65-token sequential-GPU fixture checks 200 states with both an outer
  2048 chunk and 64+1 chunks; worst state relative L2 6.01e-6, final argmax and
  eight short greedy IDs match. This tests state handoff and the short-tail fallback.
- 25 native process/RPC integration checks and four own-model long-RPC checks
  pass. The latter checks an 8K distant needle, native-limit rejection before
  prefill, independent short IDs after long cache use/rejection, and repeated
  fresh-state arithmetic. The default coordinator/worker service is restored.

Evidence: `artifacts/baselines/qwen36-grouped-delta-summary.json`, the matching
`qwen36-grouped-delta-{final-*,release8k,release32k}.json` reports and `*-data/`
folders, `qwen36-grouped-delta-acceptance.json`, native logs
`artifacts/grouped-delta-final-*.log`, and numerical comparisons under
`artifacts/reference/qwen36-grouped-delta-final*/comparison.json`.
All published binary hashes match the release reports. No matched long-prompt
FP32 MLX rerun or parity claim is made. The previous completed 240K/16K run
belongs to the previous implementation; it has not been rerun for this release.

Reproduce one final release request on free ports, with no competing model:

```sh
artifacts/baselines/mlx-env/bin/python scripts/run-qwen36-context-suite.py \
  --cases 8192:128 --trials 3 --fixed-output --restore-live --keep-awake \
  --prompt-file artifacts/baselines/qwen36-context-fixed-data/8192-4096-prompt.txt \
  --prefix local-grouped-delta
```

For the earlier control, explicitly use `--chunk 128 --experts vector
--delta sequential` and a different output prefix. Set `--base-port` to another
free pair only when the existing loaded model is stopped; do not load a second
GPU model for a performance comparison.
