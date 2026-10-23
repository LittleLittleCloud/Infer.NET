# BF16 first-stage cross-backend probe, 2026-10-23

This is an isolated numerical diagnostic for the 8-bit 20/20 split. It does not pass the fixed BF16 correctness, memory, or 128K performance gates in [QWEN36-8BIT-TARGETS.md](QWEN36-8BIT-TARGETS.md), and the candidate was not deployed.

Both hosts built source `9f43459b50dad44a83a669a2308d9144f9539e12`. The Windows RTX 4090 ran the CUDA prefix with affine8/group64 weights, BF16 GEMM inputs and KV, and diagnostic FP32 router/shared projections; the M5 Pro ran the Metal prefix with the same profile. Both used the same 128 synthetic token IDs. CUDA used 32-token blocks; Metal used a 128-token chunk. Each probe copied token 0/layer 0 intermediate values to files, so these timings cannot measure serving throughput. The fixed CUDA candidate ran twice with the same trace SHA-256 `fc52acaa68f9205277124f8871bfec417277e5425961755ce327c34198800582`. The Metal trace SHA-256 was `7e820c3bf90f866c2efe879e1fc548072ec5748ef282ac61d690c46d1a12749ec`.

| Boundary for token 0, layer 0 | Relative L2 | Max absolute difference |
| --- | ---: | ---: |
| Embedding and embedding normalization | 0, bit-identical | 0 |
| First QKV projection | 0.000002695 | 0.00008965 |
| Delta normalization | 0.000003135 | 0.000003338 |
| Attention output projection | 0.000007553 | 0.000000954 |
| MoE input residual normalization | 0.000006665 | 0.00003910 |
| Router logits | 0.000002980 | 0.00008345 |
| Expert gate | 0.000195084 | 0.00018692 |
| Expert down | 0.000694768 | 0.00008685 |
| Layer merge | 0.000395050 | 0.00003073 |

The first non-identical arithmetic output is **QKV**, before DeltaNet and MoE. The first listed stage whose relative L2 exceeds the target's `1e-4` numerical threshold is **expert gate**. This threshold comparison is diagnostic: the target's formal gate applies to the specified reference states and logits, not every intermediate tensor. Token 0 selected the same top-eight experts on both hosts: `238, 112, 6, 173, 56, 106, 43, 200`. Thus its expert-gate discrepancy is not explained by a route flip. Of the 2048 MoE input-normalization values, 15 round to different BF16 operands across hosts under round-to-nearest-even; this could amplify the small upstream differences but does not yet establish the cause. A replay feeding *identical* BF16 operands into both expert kernels is needed to separate input drift from kernel arithmetic.

The Windows prefix result reports 0.5014 s for its baseline and 0.2934/0.3189 s for the candidate/repeat. These are single, instrumented, synthetic runs with readbacks and no full service path or formal warmup. They are **not** a prefill throughput result. This probe has no transfer, Mac suffix, decode, logits, long context, or swap acceptance. Previous full 128-token BF16 layer/state comparison still passed only 2 of 60 checks.

The existing Mac serve and Windows agent were stopped only during their respective isolated model loads and then restored. A short live request returned HTTP 200 and `OK` after both workers reported Ready. The candidate remains outside the live deployment.

Reproduce using the pinned model and new artifact directories, with active GPU workers paused and restored according to the deployment SOP:

```text
Windows: powershell -File scripts/windows/check-bf16-prefix-profile.ps1 -Config PRIVATE_WORKER_JSON -Output NEW_CUDA_DIRECTORY -Router fp32 -Shared fp32 -RouteProbe 1:93 -StageProbe
Mac: DEEPINFER_METAL_KV=bf16 DEEPINFER_METAL_PREFILL_GEMM=bf16-nax DEEPINFER_METAL_PREFILL_CHUNK=128 DEEPINFER_METAL_PREFILL_MATRIX=1 DEEPINFER_METAL_PREFILL_DELTA=sequential DEEPINFER_METAL_ROUTER_FP32=1 DEEPINFER_METAL_STAGE_DIR=NEW_METAL_DIRECTORY deepinfer split-check-bf16-prefix --weights MODEL_DIR --output NEW_METAL_DIRECTORY
Compare: python3 scripts/split/compare-bf16-stages.py --metal NEW_METAL_DIRECTORY --cuda NEW_CUDA_DIRECTORY --output NEW_REPORT.json
```

Local raw evidence is retained outside Git under `artifacts/dual-host/bf16-cuda-stage-9f43459`, `bf16-metal-stage-9f43459`, and `bf16-stage-compare-9f43459.json`. The comparison JSON SHA-256 is `ec4035c3f8b3a175bf1978d40fc415d1aeef0b54f9be44a32daf54d5a44c3f81`; Windows result SHA-256 `a16278faaf8804d72047ec865e4dbb293c370f7012bda087049da09d78bad3f7`; Mac result SHA-256 `a5e76538a78f909abc8f2945e4eab338d9bfb0510cfdf8670b6e5957a67f6a21`. Raw tensors are not published. The Windows DLL SHA-256 is `0a7ded1c3f365a837961d790265adc53c4ee42f5f2c90474baeace577a5478f3`; Mac dylib SHA-256 is `290e2d5b07ebc764b3ba3201ab3c86642aca087606c3234c241ed937b43f5ec1`.

Next: replay the first expert-gate operation with identical BF16 inputs and quantized weights on both hosts, then trace the first QKV projection with a scalar reference. Re-run full layer/state/logit and real split RPC checks before any performance claim or rollout.
