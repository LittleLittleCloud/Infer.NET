# BF16 matrix operands and KV on Metal

This is an **opt-in mixed-precision path**, implemented in our own Metal shaders.
The default remains the existing FP32 executor. It is not a full BF16 model
implementation, and does not establish MLX performance or quality parity.

## What runs in which precision

| Component | Precision in this path |
|---|---|
| Stored model weights | Original packed affine 5-bit/group64; router 8-bit |
| Batched dense projections and grouped expert gate/up/down, chunks >=64 tokens | Unpack weights to BF16, round input operands to BF16, Metal 4 MPP matrix multiplication, FP32 accumulators/output buffers |
| KV cache, both prefill and decode | BF16, when `DEEPINFER_METAL_KV=bf16` |
| Residual/activation buffers, small projections, shared gate/up, classifier and decode GEMV | Existing FP32 |
| Norms, router probabilities, attention arithmetic and softmax | FP32 |
| Convolution history, chunkwise DeltaNet arithmetic and persistent recurrent state | FP32 |

The new GEMM tile is 64 output rows by 64 tokens/routes with a 64-column
inner tile. Four SIMD groups cooperate through `mpp::tensor_ops::matmul2d`.
Eight packed codes are unpacked together. Cooperative FP32 accumulators scatter
the final values directly using documented logical indices, fusing SwiGLU for
gate/up and avoiding an intermediate threadgroup result buffer. GPU-only route
compaction uses 64-route tiles for this path; the FP32 path retains 16-route
tiles. There is no persistent expanded weight cache, CPU routing, or inference
runtime dependency. Short chunks retain the existing FP32 kernels.

The MPP path requires macOS 26+ and Apple GPU family 10+ (M5). Explicit requests
fail on unsupported hardware; they do not silently use another GEMM precision.
Invalid precision option values are rejected. BF16 KV is separate from GEMM and
uses 16-bit Metal buffers with FP32 loads/attention accumulation. At the native
262144-token capacity, the ten full-attention layers require **5 GiB of KV**,
compared with 10 GiB in FP32. This is cache allocation, not a measured claim
about physical residency or total process memory. The context limit is unchanged.

The Apple API references are the [MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
and [Metal shading language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
These are API references; no third-party inference kernels are included.

## Run

Stop your existing model cluster before starting another one on the same GPU.
From the repository directory:

```sh
DEEPINFER_METAL_PREFILL_GEMM=bf16-nax DEEPINFER_METAL_KV=bf16 \
  ./artifacts/native/osx-arm64/deepinfer demo --model qwen3.6-35b-a3b --stay
```

The existing CLI/TUI connects to the coordinator at port 5100 as usual. Set both
variables to `fp32` to select the old precision explicitly. Options are read
when the worker loads its model; changing the parent terminal environment does
not change an already running worker.

```sh
bash scripts/check-metal-bf16.sh
artifacts/baselines/mlx-env/bin/python scripts/run-qwen36-context-suite.py \
  --gemm bf16-nax --kv bf16 --cases 8192:128 --fixed-output --trials 3 \
  --prompt-file artifacts/baselines/qwen36-context-fixed-data/8192-4096-prompt.txt \
  --prefix qwen36-bf16-release8k
```

`--gemm` and `--kv` configure both the owned worker and report metadata. The
runner refuses to replace an existing service. The Python environment is used
for tokenization and measurement only, and never performs product inference.

## Validation and measurements

Results and precise limitations are recorded after the final published-binary
runs in `artifacts/baselines/qwen36-bf16-summary.json`. Full BF16 activation
storage, low-precision attention/DeltaNet kernels, decode low-precision GEMV,
long-output quality evaluation, and a new full native-window run remain outside
this change. Numerical checks against rounded operands establish kernel
correctness; they do not establish identical outputs to the FP32 model.


Final M5 Pro 48 GiB results (same saved prompts, batch one, packed 5-bit model,
fixed 128 output tokens, EOS ignored, no excluded warmup):

| Input | FP32 prefill | BF16 operands + KV | Prompt throughput | Speedup |
|---|---:|---:|---:|---:|
| 8192 | 15.837 s, mean of 3 | 11.849 s, mean of 3 | 517.3 → 691.4 tokens/s | 1.34x |
| 32768 | 81.959 s, one historical trial | 65.587 s, one trial | 399.8 → 499.6 tokens/s | 1.25x |

The 8K A/B uses the same final published executable/library, with precision
flags as the difference. BF16 trials are 12.570 / 11.486 / 11.490 seconds;
FP32 trials are 16.707 / 15.395 / 15.408 seconds. Each variant is internally
repeatable in its 128 generated IDs. Between precisions, the 8K outputs first
differ at token 41. The 32K output matches all 128 IDs of the historical control.
This is evidence from these fixtures, not an assertion of general quality parity.
Aggregate 8K decode is 71.32 tokens/s; the one 32K trial is 43.10 tokens/s.
The measured 8K KV allocation drops from 0.625 to 0.3125 GiB; the 32K request
uses a 65536-slot cache of 1.25 GiB, compared with 2.5 GiB in FP32.

Validation: 68 unit tests, 210 FP32 native checks, 136 BF16 native checks,
25 process/RPC checks, and four actual-model acceptance checks pass. The BF16
acceptance includes an 8K distant 12-digit needle, native-bound rejection,
eight independent short reference IDs after long-cache use/rejection, and
repeatable fresh-state 2+2 generation. Original FP32 verification still checks
120 independent reference layer states and 248320 logits at unchanged
thresholds (worst state relative L2 2.67e-6). The final Host Native AOT publish
has no warning/error in its log. The live coordinator/worker has been restored
at port 5100 using the opt-in BF16 configuration; ordinary launch defaults
remain FP32.

Evidence includes `artifacts/bf16-{unit-tests,publish-final,integration,acceptance}.log`,
`artifacts/bf16-regression-*.log`, `artifacts/bf16-checks-final.log`,
`artifacts/reference/qwen36-bf16-fp32-regression/comparison.json`, and the
summary/report/raw result files. Final native binary hashes match all three
new final performance reports. The initial scalar-unpack experiment regressed;
it is exploratory, with one trial overlapping small hardware checks, and is
excluded from the final A/B evidence. The final tests and performance runs are
serial on one GPU/model.
