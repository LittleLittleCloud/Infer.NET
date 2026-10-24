# Independent FP32 reference for the first 20 layer, 2026-10-23

This round tested the same set of 128 synthetic tokens for the Qwen3.6-35B-A3B 8bit model: Metal FP32 and CUDA FP32 matched each other in the [same-precision device comparison](BF16-FP32KV-CONTROL-20261023.md), but the possibility that "both sides deviate together from the true model" still needs to be ruled out. Product trajectories come from the same complete source code `1f42e6fab4d58a683c66a3d009a8465521ad4403` on both sides, and the CUDA native DLL SHA-256 `4128dd0a48fa25481733f7cbe8851264e006765da1e7af7c233dd398661f3edd`. The independent reference script and comparator are in the unmerged [PR #31](https://github.com/LittleLittleCloud/Infer.NET/pull/31) source code `7d4d8f6ea34ffd5d7bf4560123a8131e9cf87f22`; they only verify numerical values and do not participate in .NET/CUDA/Metal product inference.

The reference uses a fixed model revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, affine8/group64 weights, FP32 parameters and activations, and executes token-by-token, following the same path as the previously verified independent FP32 short-prompt path. Input IDs SHA-256 `8894963a38fa5ac5efc5991b5eb193536d712234d434a83cf8dc3cc47072f0e7`. The system wired-memory limit was not changed; the original Metal serve was paused during isolation sampling and restored to the `919259d` version afterward, with the coordinator showing Ready and the Windows agent connected. The reference saved all 128-token outputs for the first 20 layers and the final logits; this round only performed a three-way comparison for the first 20 layers.

| Relative to independent FP32 reference | Layers passed | Layer 0 relative L2 | Max layer relative L2 | Max absolute error across all layers |
| --- | ---: | ---: | ---: | ---: |
| Metal FP32 GEMM/KV | 20/20 | `3.83949e-7` | `9.13393e-7` | `7.62939e-6` |
| CUDA FP32 GEMM/KV | 20/20 | `3.33568e-7` | `6.73876e-7` | `5.72205e-6` |
| CUDA BF16 GEMM / FP32 KV | 0/20 | `0.009913724` | `0.0746134` | `0.150718` |

Both FP32 paths passed the project's hierarchical diagnostic thresholds relative L2 `<1e-4`, max absolute `<0.01` against independent references. CUDA BF16 GEMM candidates failed at the 0 layer, with max absolute error at that layer being `0.00564498`. KV fixed at FP32, so the 0 layer failure cannot be attributed solely to KV format. Previous analysis of token-16 QKV operand rounding and expert routing flips within the same CUDA provided finer clues for error sources; this round did not verify any compensation schemes.

There is one mandatory failure control: initially, a batch reference processing 128 tokens at once using MLX did not match the two FP32 product trajectories that had passed. Metal's layer 0 relative L2 was `0.0059983`, and only `0/20` layers passed. It was excluded from acceptance; the comparator now requires the independent reference metadata to specify **token-by-token** execution. The reason why batch and token-by-token results differ has not been located, so batch results cannot be interpreted as product regressions.

Valid token-by-token reference result JSON SHA-256 `de590f41b1f3afbaf8a260338a17eba6728a25b10052a6389bbf86230906cfa0`; comparing JSON SHA-256 `0c4a132ce2cb9d1e2864e0d59a0b4a9ae6c47dd2360204ebea205ec261072d34` yields the same SHA via offline recomputation. Failed batch comparison JSON SHA-256 `65780290f19af67dfe0e8e4e6c9d999048d773c81c12d4359fbd34f25cc1707d`. Large raw tensors are only saved on the dev machine `artifacts/dual-host/independent-prefix-oracle-{serial-,}20261023/` and are not available for public download.

During reproduction, first check that the service is idle, power supply is normal, and the original deployment identity is correct. Temporarily pause the Mac serve according to the deployment SOP, and immediately restore it after completing the reference, confirming that the agent reconnects. Use a new output directory:

```text
artifacts/baselines/mlx-env/bin/python scripts/split/reference-prefix-mlx.py --model MODEL_DIR --output NEW_ORACLE_DIRECTORY
artifacts/baselines/mlx-env/bin/python scripts/split/compare-prefix-oracle.py --oracle NEW_ORACLE_DIRECTORY --metal METAL_FP32_DIRECTORY --cuda CUDA_FP32KV_DIRECTORY --output NEW_REPORT.json
```

This round is the 128-token prefix value check. Independent logits have not been compared with product logits, nor have the reference KV/Delta states, post-20 layers, greedy outputs, real cross-machine data channels, target BF16 KV, 128K, memory, or throughput been verified; results cannot replace full acceptance. The next step is to determine whether compensation calculations that still satisfy the BF16 multiplication input constraints can reduce first-layer and routing errors, and to measure their real cost. Related to [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3).
