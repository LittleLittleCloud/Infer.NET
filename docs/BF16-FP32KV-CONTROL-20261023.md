# 4090 first 20 layers same precision FP32 comparison, 2026-10-23

Issue: Comparing the previous CUDA frontend with Mac Metal, a `4.086e-4` relative to L2 difference began appearing from the 3 layer (the first full-attention layer). In that round, although the CUDA mode 5 used FP32 GEMM, it still used BF16 KV; Mac used FP32 GEMM/KV, so the difference cannot be attributed to device implementation. The old observation and correction are retained in [previous experiment](https://github.com/LittleLittleCloud/Infer.NET/blob/b023318f821b94e28c1cd1a1551b6e6f37c95d91/docs/BF16-STAGE-PROBE-20261023.md). This round sets both ends' GEMM and KV to FP32 to isolate this variable.

Both ends use the complete source code `1f42e6fab4d58a683c66a3d009a8465521ad4403`, Qwen3.6-35B-A3B, fixed affine8/group64 weights, and the same set of 128 synthetic token IDs (SHA-256 `8894963a38fa5ac5efc5991b5eb193536d712234d434a83cf8dc3cc47072f0e7`). Windows RTX 4090 collected CUDA mode 5's FP32 GEMM/FP32 KV and mode 6's BF16 GEMM/FP32 KV; the routing and shared expert projections for both modes serve as diagnostic FP32. Mac M5 Pro collected FP32 GEMM/FP32 KV, 128-token outer block, disabled matrix path, and sequential DeltaNet. The CUDA frontend package SHA-256 is `ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77`, and the Windows CUDA DLL SHA-256 is `4128dd0a48fa25481733f7cbe8851264e006765da1e7af7c233dd398661f3edd`.

| CUDA vs Metal FP32 Comparison | Via Layer Output | Via State | Max Layer Relative L2 | Max State Relative L2 |
| --- | ---: | ---: | ---: | ---: |
| FP32 GEMM / FP32 KV | 20/20 | 40/40 | `8.53546e-7` | `3.40833e-6` |
| BF16 GEMM / FP32 KV | 0/20 | 0/40 | `0.0733243` | `0.103492` |

The maximum state absolute difference under same-precision FP32 control is `5.43594e-5`; all are below this project's diagnostic threshold of relative L2 `<1e-4` and max absolute `<0.01`. Compared to the previous round, same-precision control eliminated the difference at the 3 layer, supporting the hypothesis that the old difference stemmed from mixed KV dtypes. When keeping KV as FP32, the BF16 GEMM candidate still failed at the 0 layer: relative L2 `0.009913728`, max absolute `0.00564497`. Therefore, the first-layer difference of BF16 GEMM cannot be explained solely by KV format. This conclusion applies only to the isolated 128-token frontend sampling; Metal FP32 is a device comparison, not a fixed independent full-model FP32 logits reference.

Windows results JSON SHA-256 `c1a5ef1e419ab2ac52ed76ce96d08841c9ebb07cd367984ea4f8bb4d6a289cac`; Mac results `a20eadb1555ca920922827a84c6c9829af53111323bba09f2e6c1c624a525883`; offline comparison `3c67f4113e8622324f53675885e86c787a352c2e7039d49122c67a994068a08e`. Large raw trajectories and states are kept only on the development machine and are not available for public download. The original `919259d` Mac serve and Windows agent were both restored after isolated checks, with the coordinator showing Ready and the agent connected. Mac Release build passed; the earlier 125 Mac and 124 Windows tests belong to the previous source code checkpoint and cannot be considered new tests for this round.

During reproduction, the existing GPU service was kept idle and temporarily paused according to the deployment SOP; it was restored after collection. Each round uses a new output directory:

```text
Windows: powershell -File scripts/windows/check-bf16-prefix-profile.ps1 -Config PRIVATE_WORKER_JSON -Output NEW_CUDA_DIRECTORY -Kv fp32 -Router fp32 -Shared fp32 -RouteProbe 999:999
Mac: DEEPINFER_METAL_KV=fp32 DEEPINFER_METAL_PREFILL_GEMM=fp32 DEEPINFER_METAL_PREFILL_CHUNK=128 DEEPINFER_METAL_PREFILL_MATRIX=0 DEEPINFER_METAL_PREFILL_DELTA=sequential deepinfer split-check-fp32-prefix --weights MODEL_DIR --output NEW_METAL_DIRECTORY
Compare: python3 scripts/split/compare-bf16-prefix.py --metal NEW_METAL_DIRECTORY --cuda NEW_CUDA_DIRECTORY --metal-fp32 --output NEW_REPORT.json
```

Probe readback changes the time cost, so actual prefill throughput cannot be inferred from these seconds. This round involves no cross-machine transfer, post-20 layers, logits, greedy output, 128K, memory, or formal performance acceptance. The next step is to use a fixed independent full-model FP32 reference to locate MoE routing rounding issues in BF16 layer outputs, states, and logits, and to conduct real RPC and 128K threshold testing after passing numerical checks. Related to [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3) and the unmerged [PR #31](https://github.com/LittleLittleCloud/Infer.NET/pull/31).
