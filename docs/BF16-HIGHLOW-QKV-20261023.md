# BF16 High/Low Bit QKV Diagnosis, 2026-10-23

[Independent FP32 Reference](BF16-INDEPENDENT-ORACLE-20261023.md) has confirmed: Qwen3.6-35B-A3B 8bit model's Metal/CUDA FP32 frontend 20 layers all passed with 20/20, whereas the current CUDA BF16 GEMM/FP32 KV is 0/20. The first-layer error is related to BF16 operand rounding and MoE routing changes. This experiment asks: If we keep each Tensor Core multiplication input as BF16, but split the FP32 input and affine dequantized weights into BF16 high and low parts, using a three-term product to correct QKV, can we make the entire layer pass?

First, perform an offline CPU numerical feasibility check on the old token 16 and layer 0 QKV. Fix the model revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, reconstruct FP32 values from the original affine8/group64 weights, and split them according to BF16 round-to-nearest-even. The table below shows the relative L2 relative to FP32 QKV dot products; CPU uses float64 accumulation, indicating only theoretical error, with no implication for CUDA accumulation or speed.

| BF16 Product | QKV Relative L2 |
| --- | ---: |
| `W_hi·X_hi` | `9.92971e-4` |
| Plus `W_hi·X_lo` | `6.52460e-4` |
| Plus `W_lo·X_hi` | `1.48232e-6` |
| Plus `W_lo·X_lo` | `9.74416e-7` |

On Windows RTX 4090 SM89, the unmerged [PR #31](https://github.com/LittleLittleCloud/Infer.NET/pull/31) adds a default-disabled `-HighLowQkv` diagnostic. It performs the first three BF16 Tensor Core products on QKV, accumulating them into FP32; it does not add any resident weight or activation buffers. Comparison is made under the **same complete source code** `720105fb191e49fac4eefee2b06ea58bb5064129` and same CUDA DLL SHA-256 `ea43e25f05997b47356e3014f9087a6e243b3a7f74317c2ab31a5da87dcb7bac` between the switch-off and switch-on modes; 128 fixed synthetic tokens, affine8 weights, FP32 KV, and diagnostic FP32 router/shared, with identical probe readbacks. The hidden vectors, 20 layer trajectories, and 40 state hashes of the switch-off mode are completely consistent with previous default BF16 GEMM/FP32 KV runs. Both candidate modes are limited, and repeated runs determine byte-by-byte consistency; the original `919259d` Windows agent has been restored, and the Mac serve shows Ready and is connected.

| Numerical Diagnostic | Default BF16 QKV | Three-Product High/Low QKV |
| --- | ---: | ---: |
| Token 16 QKV Relative Internal FP32 QKV | `9.92009e-4` | `8.01099e-6` |
| Layer 0 Relative Independent Per-Token FP32 Reference | `0.009913724` | `0.007039659` |
| Independent Reference Passed Prefix Layers | 0/20 | 0/20 |
| Passed Metal FP32 Layers and States | 0/60 | 0/60 |
| Tokens Switching Expert Sets in First Layer | 5/128 | 2/128 |
| Total 2, 560 Token/Layer Expert Set Switches | 695 | 542 |

QKV local error reduced by about 124 times, but the entire layer still far exceeds the target relative L2 `<1e-4`; **this is a failed candidate segment, cannot be deployed or claimed as a speedup**. Token 16 still replaces the eighth expert from `217` to `78`. The first layer has two token sets (16, 91) that changed experts, contributing 94.6% squared activation error; 119 tokens with unchanged expert order merged relative L2 is still `0.001634`. Token 16's QKV is close to FP32, followed by Delta normalization still at `3.735e-4`, router logits at `0.001092`, layer output at `0.065666`. Other BF16 projections and downstream routing still need to be located. Isolation probe seconds switch off `0.3017`, on `0.3011`; these reads are dense, single runs without formal warming up **cannot be used for service throughput comparison**.

Original evidence retained only on dev machine: `artifacts/dual-host/bf16-default-qkv-720105f/` and `artifacts/dual-host/bf16-highlow-qkv-720105f/`. CPU feasibility JSON SHA-256 `1407eeaf00616dde7f13f5cec2858debe3bf8c7c0470643169f0be9a001437bd`; switch-off mode results `9411b5e674c5887f7cd97ea4654eba0deec81976ce0b7b4b519c38bda8136d9a`, routing summary `48ca387a4465ec7349550c0448d0628b77ede65556a9554e43f062cf2994fe4f`; switch-on mode results `a4d9854b027d8de810fd4fcc46953da9c44896162eafcc08339990b7da162128`, independent reference comparison `aeacc0ca3bbef93b802ca9e3fad103b3fd392163ba9acfcbb6b926abcafa61d0`, routing summary `53b792ca9ab2ea2804350135ca55a6871d672a61132ef258d65413a6de29c324`, stage summary `1bc28553292eb9592517d22ea64f25887a9053e93ef174c36eac6cad116583da`.

Reproduction: Isolate the GPU, stop the old agent with the verified PID, and run Windows `scripts/windows/check-bf16-prefix-profile.ps1 -Kv fp32 -Router fp32 -Shared fp32 -StageProbe -StageToken 16 -CompareStageModes -SaveRoutes` (with `-HighLowQkv` additionally enabled) in separate new directories using the same source code and DLL. Then restore the old agent and confirm that Mac shows Ready/connected. Run `scripts/split/analyze-bf16-mode-stages.py`, `analyze-bf16-route-gap.py`, and `compare-prefix-oracle.py` offline; configuration is in PR #31. The experiment did not run the target BF16 KV, independent state/full-model logits, real cross-machine requests, 128K, memory, or formal performance acceptance. Next, measure the compute required to correct other BF16 projections, then decide whether to implement broader compensation; do not switch online services until correctness is verified. Related to [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3).
