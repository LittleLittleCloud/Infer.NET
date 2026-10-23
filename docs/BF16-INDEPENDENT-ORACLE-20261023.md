# 前 20 层独立 FP32 参考，2026-10-23

本轮检验 Qwen3.6-35B-A3B 8bit 模型的同一组 128 个合成 token：Metal FP32 与 CUDA FP32 在[同精度设备对照](BF16-FP32KV-CONTROL-20261023.md)中相互吻合，但仍需要排除“两端一起偏离真实模型”的可能。产品轨迹来自两端相同完整源码 `1f42e6fab4d58a683c66a3d009a8465521ad4403`，CUDA 原生 DLL SHA-256 `4128dd0a48fa25481733f7cbe8851264e006765da1e7af7c233dd398661f3edd`。独立参考脚本及比较器在未合并的 [PR #31](https://github.com/LittleLittleCloud/Infer.NET/pull/31) 源码 `7d4d8f6ea34ffd5d7bf4560123a8131e9cf87f22`；它们只验证数值，不参与 .NET/CUDA/Metal 产品推理。

参考使用固定模型 revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`，affine8/group64 权重、FP32 参数和激活、逐 token 执行，与此前已验证的独立 FP32 短提示词路径相同。输入 ID SHA-256 `8894963a38fa5ac5efc5991b5eb193536d712234d434a83cf8dc3cc47072f0e7`。未改变系统 wired-memory 限制；隔离采样期间暂停原 Metal serve，结束后恢复 `919259d` 版本，协调器显示 Ready、Windows agent 已连接。参考保存前 20 层的全部 128-token 输出以及最终 logits；本轮仅对前 20 层做了三路比较。

| 相对独立 FP32 参考 | 通过层数 | 第 0 层 relative L2 | 最大层 relative L2 | 所有层最大 absolute error |
| --- | ---: | ---: | ---: | ---: |
| Metal FP32 GEMM/KV | 20/20 | `3.83949e-7` | `9.13393e-7` | `7.62939e-6` |
| CUDA FP32 GEMM/KV | 20/20 | `3.33568e-7` | `6.73876e-7` | `5.72205e-6` |
| CUDA BF16 GEMM / FP32 KV | 0/20 | `0.009913724` | `0.0746134` | `0.150718` |

两条 FP32 路径对独立参考均通过本项目层级诊断阈值 relative L2 `<1e-4`、max absolute `<0.01`。CUDA BF16 GEMM 候选在第 0 层即失败，最大绝对误差该层为 `0.00564498`。KV 固定为 FP32，因此第 0 层失败不能仅归因于 KV 格式。先前同一 CUDA 内部的 token-16 QKV 操作数舍入与专家路由翻转分析，为误差来源提供了更细的线索；本轮并未验证任何补偿方案。

有一次必须保留的失败控制：最初用 MLX 一次处理 128 tokens 的批量参考，与已通过的两条 FP32 产品轨迹也不吻合，Metal 第 0 层 relative L2 为 `0.0059983`、仅 `0/20` 层通过。它被排除在验收之外，比较器现要求独立参考元数据标明**逐 token**执行。批量与逐 token 结果为何不同尚未定位，不能把批量结果解释为产品回归。

有效逐 token 参考结果 JSON SHA-256 `de590f41b1f3afbaf8a260338a17eba6728a25b10052a6389bbf86230906cfa0`；比较 JSON SHA-256 `0c4a132ce2cb9d1e2864e0d59a0b4a9ae6c47dd2360204ebea205ec261072d34`，离线重复计算得同一 SHA。失败批量比较 JSON SHA-256 `65780290f19af67dfe0e8e4e6c9d999048d773c81c12d4359fbd34f25cc1707d`。大型原始张量只在开发机 `artifacts/dual-host/independent-prefix-oracle-{serial-,}20261023/` 保存，不作为公开下载。

复现时先检查服务空闲、供电与原部署身份，按部署 SOP 临时暂停 Mac serve，完成参考后立即恢复并确认 agent 重新连接。使用新的输出目录：

```text
artifacts/baselines/mlx-env/bin/python scripts/split/reference-prefix-mlx.py --model MODEL_DIR --output NEW_ORACLE_DIRECTORY
artifacts/baselines/mlx-env/bin/python scripts/split/compare-prefix-oracle.py --oracle NEW_ORACLE_DIRECTORY --metal METAL_FP32_DIRECTORY --cuda CUDA_FP32KV_DIRECTORY --output NEW_REPORT.json
```

本轮是 128-token 前段数值检查。尚未将独立 logits 与产品 logits 比较，也未核对参考 KV/Delta 状态、后 20 层、greedy 输出、真实跨机数据通道、目标 BF16 KV、128K、内存或吞吐；结果不能替代完整验收。下一步是定位能否用仍满足 BF16 乘法输入约束的补偿计算降低首层与路由误差，并测其真实代价。关联 [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3)。
