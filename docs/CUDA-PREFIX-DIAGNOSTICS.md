# CUDA prefix profile and rejected BF16 experiment — 2026-10-23

## Measured bottleneck

An actual fixed8192-token prompt on the pinned affine8/group64 model took **13.132s** in the native CUDA front20layers without instrumentation; the event-instrumented run took **13.233s** (+0.77%). One warmup, one unprofiled run and one profiled run were used for diagnosis, not a formal throughput cohort. Hidden output and all40final states are byte-identical across the three runs.

Source `46679c40ad49a9d4f4993ddd5ce3657e955597a1`; FP32 GEMM/KV,32-token layer-major chunks, RTX4090, no competing model process. The fixed token-ID JSON SHA256 is `e52710e57e42391d4330032372c8641ecd61b620bba4cf2be9b515b1796a75ce`; pinned prefix weights SHA256 `ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77`. Model revision and input prompt hash remain those in the fixed target document.

| Phase | CUDA event interval, ms |
|---|---:|
| norm_and_input_projections | 4456.732 |
| delta_convolution | 137.877 |
| attention | 942.950 |
| output_projection_norm | 1587.427 |
| router_shared_gate_up | 969.267 |
| expert_gate_up | 3161.452 |
| expert_down_shared_down | 1844.176 |
| merge | 27.364 |
| hidden_readback | 38.095 |

Dense/input/output projections and router/shared work dominate together, followed by routed experts. Attention is about0.94s at8K. Event intervals can include host dispatch gaps; the first phase also includes input upload/embedding. These are not hardware-counter kernel-only timings. The probe excludes network, Metal suffix, incremental KV export and decode. It cannot replace end-to-end throughput or predict128K by linear scaling.

Reproduce with `scripts/split/profile-prefix.py` at the recorded source, an idle GPU, the pinned prefix file and fixed token-ID JSON. Original phase logs and binary hashes are retained locally under `cuda-prefix-profile-46679c4-retry`. An earlier transfer attempt never began GPU measurement and remains marked INVALID.

## BF16: a failed hypothesis retained

The default FP32 cross-host check still passes1748tensor/logit checks. Merely storing KV in BF16 causes280failures in the84input/8output case. Both backends implement the correct round-to-nearest-even conversion of their own FP32 state: a tiny pre-rounding difference can straddle a BF16 midpoint and amplify through later layers.

An opt-in Metal reduction order experiment attempted to match CUDA's shuffle summation. It **did not fix the error**: BF16 failures increased to369; maximum relativeL2 rose from0.0023134 to0.0658194, maximum absolute error from0.0625 to0.1549683. All8greedy IDs still agreed, which demonstrates why matching a short answer is insufficient. Default FP32 and ordered FP32 both passed, with maximum relativeL2 about1.50e-5 and1.57e-5 respectively.

Source `421dd763a47f0443533ac6d913fa4d7cba59d6ad`, native math option `DEEPINFER_METAL_MATH=ordered`, off by default. Both native/AOT builds and existing test suites passed, but the full cross-host BF16 numerical gate failed. The experiment is rejected, not deployed. The reference, tolerances and8bit weights are unchanged. These runs used FP32 GEMM and compared KV storage formats; they do not validate the formal BF16 GEMM execution profile.

## CUDA attention operator, not a model speedup

A separate partitioned attention kernel passed independent double-oracle cases and retained CUDA attention comparisons, including causal tails and FP32/BF16 KV. At131072keys with32queries, median FP32 operator time was84.16→21.25ms (128-key tiles); BF16 KV was23.74→16.79ms (256-key tiles). Smaller8K/32K benefits are much smaller, and tiny inputs can regress. The partial-output workspace must be included in any integration memory budget.

Source `55afd1049be9e8d4f55a8a863c9cdd2159139b90`, `scripts/windows/check-prefill-attention.ps1`, one warmup plus five operator trials, hot data. This kernel is **not integrated into the deployed executor**; these numbers are not full-model tokens/s. Public records are in `site/prefix-diagnostics.json`, with raw logs retained locally.

The next measured experiment reuses dequantized dense weights across multiple input tokens while preserving FP32 arithmetic order. Formal BF16/131072-input gates remain open; no threshold has been relaxed. See [Issue3](https://github.com/LittleLittleCloud/Infer.NET/issues/3).
