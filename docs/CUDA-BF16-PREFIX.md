# CUDA 前段 BF16 操作数：隔离试验

2026-10-23；关联 [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3) 和 [Issue #1](https://github.com/LittleLittleCloud/Infer.NET/issues/1)。固定 Qwen3.6-35B-A3B affine8/group64，revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`。试验源码为 `70234e8a6bd18472f7b892676baebaec2f9f3cfa`，Windows CUDA DLL SHA-256 为 `91220b0fa2d7bdb617a39e910eaf319ce043eb1ed67aa2a22047d64e57db0660`。

## 问题与实现

固定目标要求 GEMM 使用 BF16 乘法输入、KV 为 BF16，FP32 残差、路由、Delta 状态和累加。先前 CUDA 试验模式 2 仅覆盖大矩阵，专家 gate/up/down 和小投影仍使用 FP32 GEMV。本轮增加**原生诊断模式 6**：大 dense 矩阵走既有 BF16 Tensor Core GEMM；专家和小 dense 矩阵在仿射 8bit 反量化后，将权重和输入各舍入一次到 BF16，以 FP32 累加。未修改线上 `split-agent` 的 `batch-fp32-packed` 模式；模式 2 的旧行为也保持不变。

`prefill_bf16_experts_checks.cu` 使用 1/17/32 tokens、67/129/256 行、192/256/512 列和路由/非路由输入，对 BF16 舍入后的双精度 CPU 参考比较。六组均通过，最大 relative L2 `1.23544603e-7`、最大 absolute `7.4505806e-8`；同时检查输出尾部哨兵没有被覆盖。Windows 上 `nvcc -arch=sm_89` 的完整前段 DLL 也编译通过。此检查只证明新专家投影算子的布局和数值，不证明完整模型或跨后端正确。

## 完整前段诊断

停止已核对的旧 agent 后，在同一张 4090 上加载固定前 20 层 bundle。131 个合成有效 token ID，32-token 分块，BF16 KV；同一 DLL 对照线上 FP32 packed 模式 5 与新模式 6，并重复模式 6。每层取 trace，结束时取全部 40 份 KV/Delta/卷积状态。所有 trace 有限；模式 6 两次的 hidden、trace、状态 SHA-256 均一致。

相对模式 5，差异在第 0 层就出现：relative L2 `0.004558`、max absolute `0.003231`。第 9 层的 relative L2 最大，为 `0.062914`；全层最大 absolute 为第 15 层的 `0.126917`。这不是新模式的正确性失败结论：BF16 与 FP32 操作数本来会产生不同结果，且 MoE 路由会放大差别。但它**绝不能被当成已达到固定 FP32 参考容差**。下一步需用相同舍入策略的独立 Metal/CPU 参考定位首个分歧，再验证 logits、greedy 和状态交接。

带逐层 trace 读回的三个诊断运行耗时为模式 5 `0.5030 s`、模式 6 `0.4011 / 0.3994 s`。没有预热与正式三轮，也没有网络、Mac 后段、decode 或客户端计时；这些秒数**不是吞吐 benchmark，不能据此宣称提速**。当前正式部署继续使用 affine8/FP32 KV，未启用模式 6。试验后已恢复原 Windows agent，Mac serve 未重启。

## 独立 Metal BF16 参考与首处分歧

随后在源码 `4f974a0e6d8006fc78d9909de89ca26487e419a8` 下，先暂时停止原 Mac serve，以本项目 Metal 后端对同一公式产生的 **128 个** token ID 进行一次 128-token BF16 GEMM/BF16 KV prefill。它输出前 20 层 trace、40 份状态以及模型 profile；原 serve 在测试退出时恢复。Windows 在独立 checkout、同一完整 SHA、相同 token ID 和 BF16 KV 下运行 CUDA 模式 6，然后也恢复原 agent。两端 trace 按 token/层/2048 浮点数比较；KV 按 BF16，Delta/卷积按 FP32 解读。

**60 项中仅 2 项通过**固定 `relative L2 < 1e-4` 且 `max absolute < 0.01` 的门槛。首个失败为第 0 层 hidden，relative L2 `0.00483155`、max absolute `0.00481211`；最大的 relative L2 出现在第 15 层卷积历史，为 `0.08883945`，max absolute `0.630859375`。第 0 层的多数 token 相对误差约 `0.0002–0.0012`，但 token 49 与 91 分别约 `0.0308` 与 `0.0467`，提示离散路由跳变。

额外的按需路由探针确认两处 top-8 截止位置不同：token 49 的第八专家 Metal 为 `161`，CUDA 为 `137`；token 91 的第八专家 Metal 为 `34`，CUDA 为 `241`。截止位置的归一化概率约为 `0.1071`、`0.0676`，两端数值非常接近。CUDA 的 FP32 与 BF16 操作数模式在这两处均与 Metal 不同，因此不是新 BF16 专家 GEMV 单独造成。打开路由探针前后的两端 trace SHA-256 各自完全相同，排除了探针改变计算结果。**目前只能确认近阈值路由分歧是早期误差的重要来源；尚未定位路由 logits 的首个算术差异，也没有验证 logits/greedy。**

本次比较是分开运行后对诊断产物进行的离线数值检查，不能冒充产品 RPC 的状态传输，也不是性能验收。Metal 与 CUDA trace SHA-256 分别为 `782d78d36f23369e1fc83a45f53ef2dd8484951c2ad041b4ece5f85e59a3be82` 与 `659c3ffc92273abcdb3651076d2f60355c71dcbc283174a65145c8b2b17c73ed`。完整比较 JSON 仅本机保留，SHA-256 `f49d0b5c8cc8cfa1b22577d9842f3aad0bebb8be92b1babfb761542eb612e3e4`。下一步对齐路由输入、投影与 top-8 的数值策略，再重测 128-token 全状态和真实双机短请求；不通过之前保持模式 6 关闭。

最后确认原线上 FP32 服务两端均回到 Ready；通过 coordinator 发出 572-token 输入、2-token 输出的真实请求，HTTP 200、`executionPath=cuda-metal-split`、`isSimulation=false`，客户端约 `1.85 s`。这只验证回退后的旧部署可用，不是 BF16 候选验收或吞吐基准。

后续路由与共享专家 FP32 诊断见 [2026-10-23 独立实验](BF16-ROUTER-SHARED-20261023.md)：第 0 层跨后端 relative L2 降至 `0.00016054`，但完整检查仍仅 `2/60` 通过，线上候选仍关闭。

## 复现与证据

仅在旧 agent 停止、4090 有足够空闲显存后，Windows 独立 checkout 使用 `scripts/windows/check-prefill-bf16-experts.ps1`，再用 `scripts/windows/build-prefix.ps1` 编译 DLL，运行 `scripts/split/check-bf16-prefix.py --weights PINNED_PREFIX --library BUILT_DLL --output NEW_DIRECTORY`。该脚本先验证 bundle SHA-256，结果目录不可复用。Mac 停止原 serve 后显式设置 BF16 KV/GEMM、128-token chunk、matrix=1 与 sequential Delta，再执行 `deepinfer split-check-bf16-prefix --weights MODEL_DIR --output NEW_DIRECTORY`。Windows 使用 `--tokens 128 --save-trace`，最后用 `scripts/split/compare-bf16-prefix.py` 比较两份诊断产物。公开逐层摘要在 [`site/bf16-prefix-summary.json`](../site/bf16-prefix-summary.json)；131-token 原始 JSON 仅本机保存，SHA-256 为 `a37abe830157ed8c270eb8218248b85cce2b3c0c413d87328d9f3fc1223405cc`。这些数据没有达到 `docs/QWEN36-8BIT-TARGETS.md` 的 BF16 正确性、128K 性能或资源验收。
