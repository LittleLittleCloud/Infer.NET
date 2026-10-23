# 前 20 层全投影 BF16 补偿与 KV 精度诊断，2026-10-23

在 [PR #31](https://github.com/LittleLittleCloud/Infer.NET/pull/31) 的默认关闭诊断开关中，4090 的 QKV、输出与专家投影各用三项 BF16 Tensor Core 乘积补偿高低位，FP32 累加；权重仍为固定 affine8/group64。固定 128 个合成 token、20 层前段、诊断 FP32 router/shared。源码 `a20449e1e818af5760076a83f822f5bceec07967`，SM89 DLL SHA-256 `5d383567f4e63dc39b61fb040e1aca369c6156224dd8b6b0e028ed99474c4b54`。独立数值参考采用同量化权重的逐 token MLX FP32 实现；MLX **不参与产品推理**。Metal FP32 设备对照来自早前 `1f42e6f`，因此不能把它写成同完整源码的正式双机验收。

| 4090 固定 128 token 诊断 | KV | 独立 FP32 参考通过层 | 最差层 relative L2 | Metal FP32 层／状态通过 |
| --- | --- | ---: | ---: | ---: |
| 全投影高低位补偿 | FP32 | **20/20** | `2.0968e-5` | 60/60 |
| 全投影高低位补偿 | BF16 | **3/20** | `1.8929e-2` | 9/60 |
| FP32 GEMM 内部对照 | BF16 | 3/20 | `1.8783e-2` | 不构成纯 FP32 KV 对照 |

FP32 KV 档的前 20 层相对独立参考均低于固定 `1e-4` 门槛，最大绝对误差也通过；2560 次 token／层决策中专家**集合**无变化。BF16 KV 档和它的 FP32 GEMM 内部对照均从第 3 层，即第一个完整注意力层，开始失败：该层 relative L2 分别约 `4.0836e-4`、`4.0859e-4`。这支持 KV 存储／读取舍入是当前最先出现的剩余差值，但后续误差会累积，不能将所有后层偏差都归给 KV。

目标要求 BF16 KV，因此**本轮正确性仍未通过，不能将 FP32 KV 成绩算作达标**。读回密集的隔离探针耗时不代表正式吞吐；尚未验证独立 KV／状态参考、Mac 后段、logits、greedy token、真实 RPC、128K、显存及性能门槛。下一步先隔离首个注意力层的 BF16 KV 舍入，寻找不破坏目标内存预算的修正，再做两端同完整源码及端到端测试。

原始结果仅留开发机 `artifacts/dual-host/bf16-highlow-all-a20449e/` 与 `artifacts/dual-host/bf16-highlow-all-kvbf16-a20449e/`；不公开模型张量。两档结果 JSON SHA-256 为 `330feab45712cf7a46280b18c3e56cc9b7af6ae3af382290d4ae84827224a5bd`、`06e06ff5682c2f5d6775c8e6577c6b8a38c994230b0733cfabd157ba27592554`；独立参考比较 JSON 为 `326f424900aed37adb817e3505cb51cf7c8bf7771694376bd5fccec3d0c26355`、`615560f69d00c1522d434094ae82a2d4bbf8ed80dfe1648e1468600eaf1edd0f`。关联 [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3)。
