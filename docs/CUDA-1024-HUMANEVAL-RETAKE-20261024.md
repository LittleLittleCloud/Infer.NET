# CUDA 1024 分块复测：HumanEval 20 题与固定 8K 双机对照

关联 [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3)；复用已关闭的长块代码 [PR #30](https://github.com/LittleLittleCloud/Infer.NET/pull/30)。用户于 2026-10-23 明确要求重新测试 1024 块，并指定 HumanEval 20 题。本轮运行 ID 为 `chunk1024-retake-20261024T064506Z`；实验从当地时间 10 月 23 日深夜跨至 24 日凌晨。**结论：在这一套旧候选包及当前 Wi-Fi 拓扑下，1024 块不是端到端 prefill 改进，不建议切换部署。**

## 问题与受控条件

此前 1024 块让 4090 前段计算缩短，却因 Mac 等待首批 hidden 更久而使 8K 双机请求变慢。本次检查这种退化是否也出现在真实 Pi 编码工作负载。32/1024 两档都使用同一套不可变的 Mac 与 Windows Native AOT 包，源码 `47fe588160fc748a8a3f0125b46308c3f03d52d3`；只重启 Windows agent 并改变 `--cuda-prefill-chunk`。Mac serve 在两轮及固定输入复测期间保持 PID `83246`。coordinator 始终为原部署 `cd6f7da401fadf044dac13e87b415d27bb5eadfb`，不承载模型。两个候选包均为 split wire v5、affine8/group64 权重、FP32 GEMM/KV；没有测试 BF16 Tensor Core、Q8 KV 或最新 `main` 的推理代码。

Mac 经 `en0` Wi-Fi 与 Windows RTX 4090 通信，测量期间 Mac 为 AC；没有把这组数据标作有线结果。模型 revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`。Mac 包中的 `deepinfer` SHA-256 为 `64968e78583c311abb429a47f4e85b1a93109639a021f9073bc2399644cc4087`，Metal dylib 为 `1739ff5dd22a4b158f97d84d39268478a75d5683f4a9ad1a3cd0d1a8f79d4aef`；Windows exe 为 `15d79b0496c9ea597d1b681901b8d52a593824ee1e016bb7735491e987d531ef`，CUDA DLL 为 `2e6149805dd684b398107c3d18aa2e7bfd9f296f843c1b8ace3fb2c5be899061`。两端 package manifest 与文件散列已核对。短输入本机路由在候选 Mac serve 中显式设为 0，两个 HumanEval 轮次的所有请求均为 `cuda-metal-split`；每次请求重置状态，0% prefix hit。固定输入复测也走双机路径。

Pi 1.4.0 使用官方 HumanEval 数据集压缩文件 SHA-256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`、seed 0 的同一固定 20 题，context 32768、最大输出 4096、单题超时 300 秒。每轮串行运行；两端没有并发 GPU benchmark。监测进程每 100 ms 读取 serve 的 `completedRequests` 与 `lastRequest`，两轮计数逐一连续、与 Pi 请求数完全一致。prefill 速度为总输入 token 数除以 worker prefill 总时间；decode 为总 `N−1` 间隔数除以从 worker 报告的每请求速度还原的总 decode 间隔时间。Pi wall time 另列，包含工具执行。

## HumanEval 20 题结果

| CUDA 块 | 任务完成 / 语法有效 | worker 请求覆盖 | 输入 / 输出 token | prefill | decode | Pi 总耗时 |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 20/20、20/20 | 94/94 | 207,973 / 17,660 | **539.91 tok/s** | 67.15 tok/s | 654.33 s |
| 1024 | 20/20、20/20 | 101/101 | 233,889 / 19,994 | **445.13 tok/s** | 67.10 tok/s | 830.08 s |

按各自实际输入量归一化后，1024 的 worker prefill 吞吐低 **17.56%**；decode 差 **−0.07%**，基本持平；Pi 总耗时高 **26.86%**。两轮均无题目超时或 Pi 错误。只有 10/20 题的最终文件 SHA-256 相同，因此请求数、输入量和输出量并不一致；**Pi 总耗时差不能单独归因为 CUDA 块长**。本次只验证 20/20 语法有效，没有运行官方逐题断言；通过率和模型质量不能据此宣称一致。详情及逐题非敏感摘要见 [结果 JSON](../site/cuda-chunk1024-humaneval-summary.json)。

## 固定输入隔离复测

为排除 Pi 对话分叉，再用完全相同的 8192-token 固定输入文件（SHA-256 `defa12f5be86a88aed8a25aa479ab1b17860a661ff364865708e121cac17206d`）、128 个固定输出、忽略 EOS、每档预热 1 次＋正式 3 次，先测 1024、再回测 32。两档各四次输出 token 列表的 SHA-256 均为 `c63c2cf2159f018fb01f1fe836e3fc547b1a30ef83fac8b0d14913f9de0ad1cc`；有效载荷每次同为 267,812,864 字节。这里的 `prefillMs` 包括交接和 Mac 后段；各阶段可以重叠，不能直接相加。

| 指标，正式三次中位 | 32 块 | 1024 块 | 1024 相对变化 |
|---|---:|---:|---:|
| 完整 worker prefill | **11.595 s** | **12.730 s** | **慢 9.78%** |
| 客户端整次请求 | 13.735 s | 14.848 s | 慢 8.10% |
| 4090 前段纯计算 | 10.022 s | 9.239 s | 快 7.82% |
| Mac 等 hidden | 0.737 s | 1.839 s | 多等 1.102 s |
| Mac 后 20 层 | 10.304 s | 10.335 s | 约持平 |
| decode | 60.02 tok/s | 60.21 tok/s | 约持平 |

32 块正式 prefill 原始秒数为 `11.595405 / 11.907562 / 11.471479`；1024 块为 `12.412856 / 13.680834 / 12.729762`。预热分别为 `12.143249` 和 `13.897881` 秒，不计入中位。CUDA 计算确实缩短，但 1024 需要先算完较大的外层块才交付首批 hidden；Mac 等待增加，造成完整请求退化。固定输入结果与早先有线 8K/8 输出实验的方向一致，但**链路、输出长度和源码不同，数值不能并列求平均**。

## 证据、限制与恢复

原始运行文件仅在本机忽略目录 `artifacts/dual-host/chunk1024-retake-20261024T064506Z/`，不是公开下载链接。HumanEval 32/1024 的任务 JSONL SHA-256 分别为 `41f2e64485e5a18751e364b3ddc1e9a0d0b48761124c173ea69d95dac44a8c80` / `f74cc93f6f9c881713e7f5940eff30ec300a1c0911cc94b3fa9291c7cdc28067`；逐请求遥测分别为 `cec939b4831a68fc32cdd71c27f019e6fc4adfd04d60634542b8b082dc017f8e` / `6e8fc59b12b0dbbb8647be05c7d6db2f32ea47c00ce2e40ef4f8a4d27da5488a`；固定输入原始 JSON 分别为 `750f09ecfae6245570164dd79ca22647d183f22eaff1c183704d39d2c3547da1` / `c77ce7c914b454b75f8065c877416bc4edc690ff177d7bb505126fdb35dbfd83`。serve 原始日志 SHA-256 为 `a05d03cd8b45b8f2ec206f4ec612d5f462ac76fd2f162ac8a405b9585c8286a4`。首次启动时传给评测脚本的 `endpoint_build` 元数据少了一段 SHA；该未完成轮被立即中止、原文件保留并从对照排除，随后以正确完整 SHA 重启。

没有本轮独立的完整层/状态/logit 数值复核，也未采集每次请求的 GPU 峰值、Mac swap-out 或 128K 数据；这些结果不通过正式 [8bit 目标](QWEN36-8BIT-TARGETS.md)的正确性、内存和长上下文门槛。实验期间使用旧候选源码是为了让 32/1024 在同一可切换二进制上可比；不能把上表直接当作最新 `main` 或 Tensor Core 分支的速度。下一步若仍希望使用大块，需在最新源码上改善首批 hidden 交付或建立有序流水线，再用同一 20 题和固定输入复测，而不是直接切换默认值。

实验后候选 Windows agent 与 Mac serve 均已停止，原 `cd6f7da401fadf044dac13e87b415d27bb5eadfb` 部署恢复。Mac serve PID `94700`、Windows agent PID `38408` 均 Ready/已连接；经原 coordinator 的真实 558-token 双机请求返回 HTTP 200、`cuda-metal-split`。本结果 PR 只保存实验记录，不部署 1024 或修改 `main` 的计算路径。
