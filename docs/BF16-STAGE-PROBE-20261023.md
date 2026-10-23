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

The first non-identical arithmetic output is **QKV**, before DeltaNet and MoE. The first listed stage whose relative L2 exceeds the target's `1e-4` numerical threshold is **expert gate**. This threshold comparison is diagnostic: the target's formal gate applies to the specified reference states and logits, not every intermediate tensor. Token 0 selected the same top-eight experts on both hosts: `238, 112, 6, 173, 56, 106, 43, 200`. Thus its expert-gate discrepancy is not explained by a route flip. Of the 2048 MoE input-normalization values, 15 round to different BF16 operands across hosts under round-to-nearest-even.

## Offline quantized-weight reference

`scripts/split/analyze-bf16-stage-reference.py` reads the pinned safetensors directly, rounds dequantized weights and activations to BF16, and computes the token-0 QKV, output projection and selected expert gate/up products with float64 accumulation. This is a high-precision diagnostic reference, **not** the target's formal FP32 layer/logit reference. It also checked that the exported prefix bundle's QKV/output projections and all eight selected experts' gate/up weight, scale and bias bytes match the safetensors. The BF16 QKV weight operands were identical whether affine dequantization used separate float32 operations or one rounded fused operation: 0 of 16,777,216 differed.

| Comparison | Relative L2 | Max absolute difference |
| --- | ---: | ---: |
| Metal QKV vs same-input CPU reference | 0.000000342 | 0.00002031 |
| CUDA QKV vs same-input CPU reference | 0.000002676 | 0.00007445 |
| Metal expert gate vs own-input CPU reference | 0.000000367 | 0.000000635 |
| CUDA expert gate vs own-input CPU reference | 0.000000080 | 0.000000140 |
| Expert gate drift predicted from each host's BF16 input | 0.000195069 | 0.00018620 |
| Expert gate drift observed across hosts | 0.000195088 | 0.00018692 |

The residual between observed and input-predicted expert-gate drift has maximum absolute value `7.18e-7`. This strongly supports upstream input drift as the cause of the token-0 expert-gate mismatch; both expert implementations are close to the same mathematical operation on their *own* inputs. Identical-input GPU replay can still check backend accumulation order. For QKV, the inputs and quantized weights are identical, and CUDA is about 7.8 times farther from this reference in relative L2 than Metal. Accumulation order/rounding is now the leading hypothesis; the reference alone does not prove which CUDA operation is responsible or that changing it will make all 128 tokens and later states pass.

## CUDA QKV split-K A/B — local improvement, full-prefix failure

At `5c1dea1`, CUDA optionally resets its QKV FP32 MMA accumulator every 256 input columns and combines the partial results. The switch is `DEEPINFER_CUDA_QKV_SPLIT_K=1` or `-SplitKQkv` in the Windows diagnostic script; the default and deployed worker remain unchanged. Windows SM89 DLL built successfully (SHA-256 `1e3dfb188a3ec360ea4253934756409264ba923f33db6960f884bc85ae1b9928`). The fixed 128 synthetic tokens completed and the candidate was finite and deterministic. The live Windows agent was restored and reconnected to the original Mac serve after the isolated run.

| Diagnostic metric | Original QKV accumulation | Split-K 256 |
| --- | ---: | ---: |
| CUDA QKV vs CPU reference, relative L2 | 0.000002676 | 0.000000225 |
| Metal/CUDA QKV, relative L2 | 0.000002695 | 0.000000414 |
| Metal/CUDA output projection, relative L2 | 0.000007553 | 0.000021828 |
| Token-0 MoE BF16 operand differences | 15 / 2048 | 30 / 2048 |
| Layer-0 full trace, relative L2 | 0.000160545 | 0.000166285 |
| Passing whole-prefix layer/state checks | 2 / 60 | 2 / 60 |

The QKV arithmetic becomes much closer to the independent reference, but downstream BF16 rounding is discontinuous: the output projection's input-drift component rises from `6.07e-6` to `2.11e-5` relative L2, while its CUDA own-input error remains about `4.9e-6`. Layer 1 and some later layers improve, others regress. The fixed numerical gate still fails at layer 0. This A/B is a negative result for the *whole* BF16 prefix, so split-K stays opt-in. Probe readbacks, a single synthetic run and separate platform executions make the reported CUDA seconds unsuitable for throughput claims. No full logits, real split RPC, 128K or memory acceptance was performed.

The Windows prefix result reports 0.5014 s for its baseline and 0.2934/0.3189 s for the candidate/repeat. These are single, instrumented, synthetic runs with readbacks and no full service path or formal warmup. They are **not** a prefill throughput result. This probe has no transfer, Mac suffix, decode, logits, long context, or swap acceptance. Previous full 128-token BF16 layer/state comparison still passed only 2 of 60 checks.

The existing Mac serve and Windows agent were stopped only during their respective isolated model loads and then restored. A short live request returned HTTP 200 and `OK` after both workers reported Ready. The candidate remains outside the live deployment.

Reproduce using the pinned model and new artifact directories, with active GPU workers paused and restored according to the deployment SOP:

```text
Windows: powershell -File scripts/windows/check-bf16-prefix-profile.ps1 -Config PRIVATE_WORKER_JSON -Output NEW_CUDA_DIRECTORY -Router fp32 -Shared fp32 -RouteProbe 1:93 -StageProbe
Mac: DEEPINFER_METAL_KV=bf16 DEEPINFER_METAL_PREFILL_GEMM=bf16-nax DEEPINFER_METAL_PREFILL_CHUNK=128 DEEPINFER_METAL_PREFILL_MATRIX=1 DEEPINFER_METAL_PREFILL_DELTA=sequential DEEPINFER_METAL_ROUTER_FP32=1 DEEPINFER_METAL_STAGE_DIR=NEW_METAL_DIRECTORY deepinfer split-check-bf16-prefix --weights MODEL_DIR --output NEW_METAL_DIRECTORY
Compare: python3 scripts/split/compare-bf16-stages.py --metal NEW_METAL_DIRECTORY --cuda NEW_CUDA_DIRECTORY --output NEW_REPORT.json
CPU reference (NumPy diagnostic dependency): python3 scripts/split/analyze-bf16-stage-reference.py --model MODEL_DIR --metal NEW_METAL_DIRECTORY --cuda NEW_CUDA_DIRECTORY --bundle PREFIX_BUNDLE --output NEW_REFERENCE.json
```

Local raw evidence is retained outside Git under `artifacts/dual-host/bf16-cuda-stage-9f43459`, `bf16-metal-stage-9f43459`, `bf16-stage-compare-9f43459.json`, `bf16-cpu-reference-9f43459-v4.json`, `bf16-cuda-splitk-5c1dea1`, `bf16-splitk-stage-compare-5c1dea1.json`, `bf16-splitk-full-compare-5c1dea1.json`, and `bf16-splitk-cpu-reference-5c1dea1-v2.json`. SHA-256: baseline stage comparison `ec4035c3f8b3a175bf1978d40fc415d1aeef0b54f9be44a32daf54d5a44c3f81`; baseline CPU reference `d3a77f121f507f3d9a389cc1189d8dac575a3eb9ee24f59e6b7bfdb8ed39283e`; split-K stage comparison `622a3a2557248df92be5c31b2c95f8045f41b1dde801c9186bf9eca673dd34b5`; split-K full comparison `6ddde4bd68cb7eea3577ddb14377d65e14fae100c4fb5b434e7dbc9e947711c6`; split-K CPU reference `ce8c7c3282ede32815a31c14c00a7e6328e77db33e1299da611ecae8532fb2d3`; split-K Windows result `11dd7900af69a099e2fa710400e240e5751dfba00df0cd702893d2cd2a891957`. Windows baseline result SHA-256 `a16278faaf8804d72047ec865e4dbb293c370f7012bda087049da09d78bad3f7`; Mac result SHA-256 `a5e76538a78f909abc8f2945e4eab338d9bfb0510cfdf8670b6e5957a67f6a21`. Raw tensors are not published. The baseline Windows DLL SHA-256 is `0a7ded1c3f365a837961d790265adc53c4ee42f5f2c90474baeace577a5478f3`; Mac dylib SHA-256 is `290e2d5b07ebc764b3ba3201ab3c86642aca087606c3234c241ed937b43f5ec1`.

Next: isolate the output projection's accumulation and its BF16 input boundary crossings; if a controlled variant improves full layer/state parity, repeat logits and real split RPC before any performance claim or rollout.
