# BF16 高低位 QKV 诊断，2026-10-23

[独立 FP32 参考](BF16-INDEPENDENT-ORACLE-20261023.md)已确认：Qwen3.6-35B-A3B 8bit 模型的 Metal/CUDA FP32 前 20 层均为 20/20 通过，而当前 CUDA BF16 GEMM／FP32 KV 为 0/20。首层误差与 BF16 操作数舍入及 MoE 路由变化有关。本实验问：仍让每次 Tensor Core 乘法输入为 BF16，但把 FP32 输入和仿射反量化权重拆成 BF16 高、低两部分，用三项乘积修正 QKV，能否让整层通过？

先对旧的 token 16、层 0 QKV 做离线 CPU 数值可行性检查。固定模型 revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`，从原始 affine8/group64 权重重建 FP32 数值，按 BF16 round-to-nearest-even 拆分。下表为相对 FP32 QKV 点积的 relative L2；CPU 用 float64 累加，只说明理论误差，没有 CUDA 累加或速度含义。

| BF16 乘积 | QKV relative L2 |
| --- | ---: |
| `W_hi·X_hi` | `9.92971e-4` |
| 再加 `W_hi·X_lo` | `6.52460e-4` |
| 再加 `W_lo·X_hi` | `1.48232e-6` |
| 再加 `W_lo·X_lo` | `9.74416e-7` |

在 Windows RTX 4090 SM89 上，未合并的 [PR #31](https://github.com/LittleLittleCloud/Infer.NET/pull/31) 加入默认关闭的 `-HighLowQkv` 诊断。它对 QKV 执行前三项 BF16 Tensor Core 乘积，累加为 FP32；不新增常驻权重或激活缓冲。与开关关闭档在**同一完整源码** `720105fb191e49fac4eefee2b06ea58bb5064129`、同一 CUDA DLL SHA-256 `ea43e25f05997b47356e3014f9087a6e243b3a7f74317c2ab31a5da87dcb7bac` 下比较；128 个固定合成 token、affine8 权重、FP32 KV、诊断 FP32 router/shared、相同探针读回。开关关闭档的隐藏向量、20 层轨迹和 40 份状态哈希与先前默认 BF16 GEMM／FP32 KV 运行完全一致。两档候选均有限、重复运行逐字节确定；原 `919259d` Windows agent 最后已恢复，Mac serve 显示 Ready 且已连接。

| 数值诊断 | 默认 BF16 QKV | 三乘积高低位 QKV |
| --- | ---: | ---: |
| token 16 QKV 相对内部 FP32 QKV | `9.92009e-4` | `8.01099e-6` |
| 层 0 相对独立逐 token FP32 参考 | `0.009913724` | `0.007039659` |
| 通过的独立参考前段层 | 0/20 | 0/20 |
| 通过的 Metal FP32 层和状态 | 0/60 | 0/60 |
| 首层换专家集合的 token | 5/128 | 2/128 |
| 全 2,560 次 token／层中换专家集合 | 695 | 542 |

QKV 局部误差约降 124 倍，但整层仍远超目标 relative L2 `<1e-4`；**这是失败的整段候选，不能部署或宣称提速**。token 16 仍将第八专家从 `217` 换成 `78`。首层剩下两个换专家集合的 token（16、91）贡献 94.6% 的平方激活误差；119 个专家顺序不变的 token 合并 relative L2 仍为 `0.001634`。token 16 的 QKV 已接近 FP32，随后 Delta normalization 仍为 `3.735e-4`，router logits 为 `0.001092`，层输出为 `0.065666`。其他 BF16 投影及下游路由仍需定位。隔离探针秒数开关关闭 `0.3017`、开启 `0.3011`；这些读回密集、未按正式预热的单次运行**不能用于服务吞吐比较**。

原始证据仅留开发机：`artifacts/dual-host/bf16-default-qkv-720105f/` 与 `artifacts/dual-host/bf16-highlow-qkv-720105f/`。CPU 可行性 JSON SHA-256 `1407eeaf00616dde7f13f5cec2858debe3bf8c7c0470643169f0be9a001437bd`；关闭档结果 `9411b5e674c5887f7cd97ea4654eba0deec81976ce0b7b4b519c38bda8136d9a`、路由摘要 `48ca387a4465ec7349550c0448d0628b77ede65556a9554e43f062cf2994fe4f`；开启档结果 `a4d9854b027d8de810fd4fcc46953da9c44896162eafcc08339990b7da162128`、独立参考比较 `aeacc0ca3bbef93b802ca9e3fad103b3fd392163ba9acfcbb6b926abcafa61d0`、路由摘要 `53b792ca9ab2ea2804350135ca55a6871d672a61132ef258d65413a6de29c324`、阶段摘要 `1bc28553292eb9592517d22ea64f25887a9053e93ef174c36eac6cad116583da`。

复现：隔离 GPU，停止已核对 PID 的旧 agent，使用同一源码和 DLL，在不同新目录分别运行 Windows `scripts/windows/check-bf16-prefix-profile.ps1 -Kv fp32 -Router fp32 -Shared fp32 -StageProbe -StageToken 16 -CompareStageModes -SaveRoutes`（开启档另加 `-HighLowQkv`），然后恢复旧 agent 并确认 Mac 显示 Ready／connected。离线运行 `scripts/split/analyze-bf16-mode-stages.py`、`analyze-bf16-route-gap.py` 和 `compare-prefix-oracle.py`，配置见 PR #31。实验未跑目标 BF16 KV、独立状态／全模型 logits、真实跨机请求、128K、内存或正式性能验收。下一步先衡量修正其他 BF16 投影所需计算量，再决定是否实现更广泛的补偿；未通过正确性前不切换在线服务。关联 [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3)。
