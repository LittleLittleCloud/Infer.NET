# RTX 4090 Tensor Core GEMM：同精度整段验证结果

实验 Issue [#38](https://github.com/LittleLittleCloud/Infer.NET/issues/38)，代码候选 [PR #51](https://github.com/LittleLittleCloud/Infer.NET/pull/51)。本记录是按 [实验 SOP](EXPERIMENT-SOP.md) 独立保存的结果；**没有将候选代码合入 main，也没有将 BF16 模式部署成在线 worker**。

## 问题与结论

问题是 affine8/group64 权重的 BF16 Tensor Core GEMM 在单算子变快后，整段 CUDA 前 20 层是否也比当前 main 的同精度实现快，以及是否比现行 packed FP32 路径快。结论分两层：

1. **同精度 BF16 对照成立。** 固定 512 个合成 token，main BF16 前段中位 1.189370 秒，候选 BF16 0.613646 秒，节时 **48.41%**。131-token 的逐层输出、最终 hidden 和全部 40 份 KV/Delta/卷积状态在两版 BF16 之间逐位一致。
2. **现行 FP32 路径没有被超越。** 同一候选 DLL 的 packed FP32 mode 5 中位 0.607222 秒，候选 BF16 mode 2 为 0.613646 秒，BF16 **慢 1.06%**。这是原生 4090 前段诊断，不含跨机传输、Mac 后段或 decode；不能写成用户可见 prefill 提速。没有运行 BF16 在线 HumanEval、正式 128K 或 Q8 KV 验收，**不建议据此采用 BF16 服务路径**。

## 版本、输入与设备

| 项目 | 本轮事实 |
|---|---|
| main 基线 | `99e97761df6ace50e5ca0894da5e3c51772ae20b`，mode 2，原 BF16 Tensor Core scalar tile loader |
| 代码候选 | `6e5a06131ff3c08231878d2f434a3ec18f909f6b`，mode 2，向量化 packed affine8 tile loader；mode 5 作 FP32 对照 |
| GPU | Windows RTX 4090，SM89，原生 CUDA DLL；实验前已核对驻留 agent 空闲，约 19,967 MiB 占用、29°C；测量时暂停该 agent，结束后恢复 |
| 权重 | Qwen3.6-35B-A3B revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`，affine 8bit/group64，前 20 层 bundle SHA-256 `ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77` |
| 状态与输入 | FP32 KV、FP32 累加/递归状态；mode 2 的 GEMM 乘法输入为 BF16。CUDA 原生块 32 tokens，合成 token ID 序列 SHA-256 `0040fef20910cc640cdc3ea957a5ce82d17b3a21cc436da9f148e0a05dfe447a` |
| 测量 | 每模式先 1 次完整 512-token 预热，再 3 次正式请求，重置 KV/Delta；Python 同一主机单调时钟，逐个 DLL 装载、释放，不并发占用 4090 |
| 环境限制 | 仅 native 前 20 层，含每个 32-token 块的 hidden 读回；无网络、Mac、Pi、TTFT、decode。GPU 运行峰值显存、Windows 供电模式和正式 swap 计数未采集，不能据此判断资源门槛通过 |

main 与候选 DLL SHA-256 分别为 `ccc5df7c21ca81eea11fc778139e9bde79668ac71c75b18741f4cd897baea199`、`5cb57922a9051a248690d5c4a262c6d5d3efca25d33bb59d39d534a4da29de79`。此次 `main` 的 CUDA native 源与更早的 `e017d84` 无差异，但本轮仍从当前 `99e9776` 独立检出并编译基线。两次完整 native A/B 在同一设备上得到相近结果；下表使用第二次运行 `tensor-core-precision-6e5a061`，第一次 `tensor-core-main-branch-52e0bac` 保留为复核证据。

## 算子、整段与正确性

算子脚本从候选源码构建，覆盖三种小形状与尾部，并以 double-BF16 oracle 检查；8192×2048 的大形状用独立 SIMT BF16 结果检查。候选 packed loader 与原 scalar Tensor Core loader **逐位一致**，大形状相对 L2 不超过 `7.23e-7`、最大绝对差不超过 `4.77e-7`。暖机后每个形状测三轮、每轮五次 launch 的 CUDA event 单算子中位数如下：

| 输入 tokens × 行 × 列 | 旧 Tensor Core | 候选 Tensor Core | 候选 SIMT BF16 |
|---|---:|---:|---:|
| 32 × 8192 × 2048 | 0.2105 ms | **0.0690 ms** | 0.6436 ms |
| 128 × 8192 × 2048 | 0.2517 ms | **0.1058 ms** | 2.5524 ms |
| 512 × 8192 × 2048 | 0.7490 ms | **0.3379 ms** | 10.1519 ms |

完整前 20 层在 131 个合成 token、混合 `1/7/17/32` 长度块上逐字节比较，两版 BF16 的所有层 trace、最终 hidden、40 份状态一致，无非有限值。512-token 三次正式测量如下；一次预热不计入中位数：

| 路径 | 正式三次，秒 | 中位秒 | 前段 token/s |
|---|---|---:|---:|
| main BF16 Tensor Core | 1.184539 / 1.190653 / 1.189370 | 1.189370 | 430.48 |
| 候选 BF16 Tensor Core | 0.613646 / 0.615330 / 0.611912 | **0.613646** | **834.36** |
| 同版候选 packed FP32 | 0.611114 / 0.607222 / 0.604531 | **0.607222** | **843.18** |

前一次 BF16-only 复测的 main/候选中位为 1.185164 / 0.610957 秒，方向和量级一致。两版 BF16 的 512-token hidden SHA-256 都是 `3fa37830bf771fcd327b4b3c9f6593eca03507158db464b6705c503cfcbefd05`；131-token trace SHA-256 都是 `206a0c37329c9c33f1f9c90aa75b9e985e51d27892a813f320d98dd356dfe113`。

**跨精度只作诊断。** 候选 BF16 与同版 FP32 的 131-token 全层 trace 相对 L2 为 `0.03972`、最大绝对差 `0.14984`，首处分歧在 token 0 / layer 0。BF16 与 FP32 本来可能不同，这不是同精度实现失败，也不直接证明回答质量退化。项目 v2 目标要求独立 BF16 同精度参考、预先固定的跨精度任务质量评价和真实后段/logits 验证；这些本轮没有完成。因此没有启动 BF16 在线 5/20 题 HumanEval，也没有 worker prefill、decode、TTFT 或通过率数据；首页趋势图不得把 native 512-token 速度画成这些指标。

## 恢复、证据与采用决定

原 Windows agent 由脚本核对 PID 与可执行路径后短暂停止，两轮测完均以原包/原配置在前台恢复。第二轮恢复后 PID 62216 报告 `Ready`，Mac `/ready` 与 coordinator worker 列表均就绪，真实短请求 HTTP 200、执行路径 `cuda-metal-split`。代码 PR 的 Mac Release 构建和 40 项集成检查在本地通过；首次 CI 因旧测试仍期望三种 `cuda-prefill` 选项而失败，代码分支已修正该断言，最新 CI 需以 PR 状态为准。

原始 `results.jsonl`、`result.json`、每层状态、trace、manifest、恢复日志仅保存在 Windows 的 `Infer.NET-cuda-gemm/artifacts/dual-host/` 下两个上述运行 ID，**不是公开下载链接**。第二轮 `result.json` SHA-256 为 `30808af0434a0ecfc8ea52aaae07793f81b56b21b153a0cefea3e54eb1c9c1af`；算子 JSONL 为 `940385c16dff78eb35471c170dc14afc2682daa72bfb2633a69f6361c79a7fb6`。

采用建议：**保留候选代码供进一步实验，不以本轮 native BF16 收益合入线上默认路径。** 若未来要评估服务收益，先提供可隔离的 BF16 agent 开关，完成同精度 CUDA→Metal 状态/logits 和任务质量检查，再按 SOP 运行固定 5 题迭代与 20 题正式对照；Q8 KV、128K、显存/交换和端到端门槛仍分别验收。
