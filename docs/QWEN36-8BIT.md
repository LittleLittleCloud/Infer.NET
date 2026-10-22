# Qwen3.6-35B-A3B：8bit 验证与双机容量重估

2026-10-22，M5 Pro 20 核 GPU / 48 GiB，macOS 26.6.2，AC / High Power。

后续双机实现的固定性能门槛与计时规则见 [8bit 性能目标 v1](QWEN36-8BIT-TARGETS.md)。本文保留既有实测与容量估算，不作为双机达标证据。

## 检查点与实现

`mlx-community/Qwen3.6-35B-A3B-8bit`，固定 revision
`e06a74e6236a60c8367e1a3214e83d8b61b637b0`，affine 8bit / group64。
下载文件逐一验证 LFS SHA-256 或 Git blob hash；保留 LICENSE 和 provenance。
检查点的 MLX 格式由 C# 读取，实际产品执行仍为 Native AOT + 自写 Metal kernel。
MLX 仅用于独立数值参考，没有用于产品请求或下面的性能测量。

此次补齐 embedding、classifier、批量 GEMM、共享专家和路由专家的 8bit 路径，
同时保留 5bit 回归。模型量化配置和单矩阵覆盖值会校验。
KV 容量从 2 的幂增长改为 2048-token 页，避免 128K 输入加输出就申请 256K KV。
TUI 文案移除硬编码的“5-bit”。

## 正确性

独立 FP32 参考与自写引擎：120 组逐层状态、完整 logits、8 个 greedy token。
8bit 逐层最大 relative L2 = `3.559540353840437e-6`，
logits relative L2 = `2.2151446631205613e-6`；8 个输出 token 完全一致。
阈值保持 relative L2 < 1e-4、max absolute < 0.01。
5bit 实际模型回归同样通过。短数值参考不等于验证长上下文语义质量。
这证明实现复现了该 8bit 检查点，不是 8bit 对原始 BF16 模型的任务质量评测。

Release 构建和 Native AOT 发布成功；68 项 .NET 单元测试通过。
Metal 检查覆盖 5/8bit GEMV、FP32/BF16 专家 GEMM、M5 packed/fused 路径、
attention、DeltaNet 以及 KV 页边界；各检查原始日志在 `artifacts/qwen36-8bit-*.log`。
8bit 参考与比较结果在 `artifacts/reference/qwen36-8bit/`。

## 测量口径

单 worker、单请求，加载模型后预热，再测量；每个请求清空 KV/Delta 状态，
**没有跨请求 prefix cache hit**。权重 8bit，GEMM 输入和 KV 为 BF16，
残差、路由、Delta 状态和累加仍为 FP32。attention query batch=128，
key tile=128，partition=1024；hybrid dense GEMM、packed/adaptive/fused 专家、device DeltaNet。

128 个固定输出测量 127 个 decode 间隔。长输出也忽略 EOS 以得到固定长度吞吐，
不能当作模型自然生成了这么长的有意义答案。prefill 不包括模型加载，
首次请求的 GPU 驻留开销单独保留在预热记录中。
128-output 各档是一次预热、三次测量。4096-output 改用 512 分块，
完整预热一次再测一次，单独列为容量/持续解码验证，不与 2048 分块混算。

结果与原始请求见 `artifacts/baselines/qwen36-8bit-*`。
系统 swap 观测是整机计数，不能把别的进程的换入归因于 worker。

最终独立 classifier 提交版本的 128K / 128 输出：预热一次、测三次，
prefill 为 **135.775 / 136.170 / 135.954 秒**，均值 **135.966 秒**，
解码合并吞吐 **37.520 token/s**，三个输出 token 序列完全一致。
Metal allocation **37.938 GiB**，KV capacity **133120**。
首轮预热 prefill **166.358 秒**，不能与稳态混算。
三轮系统新增 Swapouts 页数为 **0 / 0 / 8428**（16 KiB/page，末轮约 131.69 MiB）；
Swapins 为 **29570 / 9315 / 12049**。这是整机环境中的结果，未关闭用户应用。

同一最终二进制、相同配置下的稳态结果（每档 3 次）：

|输入|固定输出|prefill 秒|prefill token/s|decode token/s|Metal allocation GiB|
|---|---:|---:|---:|---:|---:|
|8192|128|3.467|2362.7|66.34|35.351|
|32768|128|17.743|1846.8|60.51|35.868|
|131072|128|135.966|964.0|37.52|37.938|

长输出独立配置（512 分块，一次完整预热、一次正式测量）：

|输入|固定输出|prefill 秒|prefill token/s|decode 秒|decode token/s|Metal allocation GiB|
|---|---:|---:|---:|---:|---:|---:|
|131072|4096|164.157|798.5|105.123|38.95|37.541|

worker 首 token 到请求生成结束共 **269.355 秒**（含前置处理）；预热 prefill **206.333 秒**。
KV capacity **135168**，KV bytes **2,768,240,640**。两遍 4096 个输出 token 完全一致；
正式测量 16 个解码窗口约 **38.30–39.19 token/s**。
随后 4 项真实 RPC 验收通过：8K 远距离校验码、超限请求拒绝、长请求后短参考 token 一致、
重复算术请求确定性。验收不等于 128K 自然语言质量评测。

长输出正式测量整机新增 swap-out **16639 页 = 259.98 MiB**，换入 **39669 页**。
首次完整预热则新增 swap-out **620469 页 ≈ 9.47 GiB**，换入 **885357 页**。
这说明本机正常应用环境下加载/驻留 8bit 模型会造成明显内存压力，不能宣传成“48 GiB 下无 swap”。

8K 最后一轮整机新增 6340 个 swap-out 页（99.06 MiB），32K 三轮均为零。
所有档位内部的三次输出 token 序列一致。
本轮没有测量 MLX 的 8bit 性能，不能沿用历史 5bit 成绩判断框架间的差距。

## 128K 的内存边界

按实际 safetensors header 计算，语言权重 **34.298735 GiB**，
全部 10 层 full-attention 的 128K BF16 KV 为 **2.50 GiB**，
30 层 DeltaNet/卷积状态约 **62.81 MiB**，另需临时缓冲、运行时和输出容量。
Metal 返回推荐 working-set budget 为 `40,200,896,512` bytes（37.44 GiB），
并不是用户可任意独占的 48 GiB。

2048-token prefill chunk 的首轮 128K 在最后一个分块报 Metal
`kIOGPUCommandBufferCallbackErrorOutOfMemory`；129024 token 的中间进度不算完成。
原始失败记录保留，不能用其时长作为成功的基准。
512-token 分块仍在最后一步失败。最终修复是将 8bit 的词表投影单独提交：
每个请求只增加一次 GPU 提交，层间数据不回 CPU，计算精度不变。
同一提交不再同时引用 prefill 工作区和完整词表矩阵；128K 已用 2048 分块完成
prefill 和 decode。总 Metal allocation 仍可超过推荐 working-set 数值，
关键是提交时需要同时驻留的资源集合；推荐值不是总 allocation 的绝对上限。
失败记录分别保留在 `qwen36-8bit-128k-128out`、`qwen36-8bit-128k-chunk512-pilot`。
最终二进制测量用 `qwen36-8bit-split-*` 前缀，与旧结果分开。

128K + 4096 输出若仍用 2048 分块，预留容量从 133120 增至 135168，
额外 KV 加上工作区仍会触发首个提交 OOM。该失败记录在
`qwen36-8bit-split-128k-4096out`。长输出重测同时采用 **512 分块 + 独立 classifier 提交**，
报告前缀为 `qwen36-8bit-split-128k-chunk512-4096out`。
所以“128K 输入可用”必须连同输出预算和工作区配置说明，不能泛化为任意长度均可用。

## 4090 + Mac：先检查容量

目标是 4090 常驻前若干层，Mac 常驻完整模型并负责后半 prefill 和完整 decode。
因此不是把 24 GiB 与 48 GiB 当作透明的 72 GiB 显存池。
GPU 完成前半层 prefill 后，Mac 必须取得这些层的 KV 与最终 Delta/卷积状态，
才能继续完整模型解码；只传 hidden vector 不够。

128K 输入，GPU 前段含 embedding；状态按 BF16 KV / FP32 recurrence 计算：

|4090 / Mac prefill 层数|4090 权重|128K 状态|合计|距 24 GiB 余量|
|---|---:|---:|---:|---:|
|20 / 20|17.149 GiB|1.281 GiB|18.430 GiB|5.570 GiB|
|24 / 16|20.479 GiB|1.537 GiB|22.015 GiB|1.985 GiB|
|28 / 12|23.808 GiB|1.793 GiB|25.601 GiB|超出|

上述余量尚未扣 Windows 显示占用、CUDA 工作区、传输缓冲和 allocator 开销。
优先验证 20/20；24/16 须实测峰值后决定；28/12 已不满足容量。
8bit Mac 单机能否完成目标长度，是这个方案的前提。

每 token 边界 hidden vector 为 2048 × FP32 = 8192 bytes。
128K hidden 合计 1 GiB；加上 GPU 前段 KV 与最终 recurrence 状态，
20 层共 **2.281 GiB**、24 层共 **2.537 GiB**。
假设有效吞吐为 110 MB/s 或 1 GB/s，链路服务时间分别为：

|前段|110 MB/s|1 GB/s|
|---|---:|---:|
|20 层|22.26 s|2.45 s|
|24 层|24.76 s|2.72 s|

这是字节数除假设吞吐，不是家中网络实测。可分块重叠传输与计算，不能把全部网络时间
简单加到计算时长上；也不能忽略 CPU staging、GPU readback/upload 和流水线首尾开销。

若 90% prefix 在两端都已有正确缓存，仍需完整容量，但只需发送新后缀 hidden/KV
与最终 recurrence 状态，理论流量降至约 0.256 / 0.287 GiB（20/24 层）。
若 GPU 没有对应 prefix，需先补齐状态；解码新增状态也须在下一轮同步。
命中位置还必须有对应的 DeltaNet/卷积状态快照，只有 KV 或更长前缀的最终 recurrent 状态不能任意回退。
当前产品没有这套双端 prefix cache，因此这里只分析容量和协议，未声称测得 cache-hit 吞吐。

## 如何判断双机值得实现

设单机实测 prefill 为 T，假设同类层耗时可按层数拆分且两端流水线充分重叠，
20/20 的 Mac 后段约 T/2，24/16 约 0.4T。
这些是乐观的计算瓶颈估计，还需等待 4090、网络和首尾流水线开销，不能当作双机实测。
20/20 的 prefill 加速上限约 2 倍，24/16 约 2.5 倍；实际可能更低。

128K 下，按实际投影矩阵形状和 causal full-attention QK/AV 计数，
全模型约 2046.16 TFLOP；这不含 DeltaNet、反量化、归一化、路由和传输开销。
前 20/24 层分别约 1023.08/1227.69 TFLOP。
以“这些工作量 / 实测 Mac 后段估计时间”给出所需有效吞吐，
比直接套 4090 的广告峰值更有用。真正的下一道门槛是用相同 8bit 格式和分块
测出 4090 前段的实际时长、峰值显存及状态导出成本。
CUDA 原型应保留相同 affine 8bit 权重、BF16 乘法输入和 FP32 累加；
若改成 W8A8 或 FP8 算法，必须重新验证，不能直接当作相同精度的吞吐对照。

以本次 **135.966 秒** 代入，20/20 的 Mac 后段下限估计 **67.98 秒**，
24/16 为 **54.39 秒**。要追上它们，4090 对上述投影/attention 工作的有效吞吐
分别须达到约 **15.05 / 22.57 TFLOP/s**，并额外承担未计入的其他算子开销。
这些是原型的验收目标，不是对 RTX 4090 已有的实测结论。
千兆链路的 22–25 秒总服务时间可与计算重叠，所以不能仅凭该数字就判定千兆必然无收益；
10GbE 提供更大余量，但是否值得购买仍取决于实际 CUDA 前段与网络测量。

长输出的整次请求时间应看 `prefill + (输出数−1)/decode速度`。
如果 decode 完全留在 Mac，prefill 加速不会加速这一段；90% prefix 命中时，
剩余 prefill 更短，长输出场景的总耗时收益也会进一步变小。
冷 prefill 的最后 10% 分块耗时仅可作为 suffix 计算量的近似，不是实际 cache-hit 测量。

以**实际长输出的 512 分块配置**做同口径估算，prefill 164.157 秒、decode 105.123 秒：

|方案|Mac prefill 按层数估算|保持现有 decode 后的理想总时间|相对单机 269.28 秒|
|---|---:|---:|---:|
|20/20|82.08 s|187.20 s|1.44×|
|24/16|65.66 s|170.79 s|1.58×，显存余量小|

前提是 GPU 和网络能跟上且充分重叠，尚未计流水线首尾、状态导出/导入和调度开销。
因此优先做 20/20 前段 CUDA 微基准；以上数字不是实际双机结果。
这次 512 分块的最后 10% prefill 约 **24.09 秒**。假设未来正确命中 90% prefix，
沿用 105.12 秒长 decode，20/20 理想总时间仅从约 **129.21 秒**降到 **117.17 秒（1.10×）**。
这是根据分块时长推算的场景，不是 prefix-cache 实测。

当前 CUDA 后端尚未执行该模型的前段；跨机器层切分、KV/Delta 状态传输、
双端 prefix cache 均未实现。因此当前结论用于筛选原型，不能宣称已经获得双机加速。

若还要降低 Mac 的常驻内存，就需要两端在 decode 阶段也各跑自己的层，
而不是让 Mac 保留完整模型接管 decode。那是另一种方案，需要测每 token 的两段计算和网络往返；
本文的 68/54 秒估计只针对 prefill 分流，不能证明分层 decode 的速度。

## 复现

先运行下载器 `scripts/download-qwen36-mlx.py --bits 8`。
使用独立的 `artifacts/native/qwen36-8bit/deepinfer`，避免替换已有 5bit 服务二进制。
原始权重在 `models/qwen3.6-35b-a3b-8bit-mlx`。

```sh
artifacts/baselines/mlx-env/bin/python scripts/reference-qwen36.py \
  --model models/qwen3.6-35b-a3b-8bit-mlx --output artifacts/reference/qwen36-8bit
artifacts/native/qwen36-8bit/deepinfer verify-qwen36 \
  --weights models/qwen3.6-35b-a3b-8bit-mlx \
  --reference artifacts/reference/qwen36-8bit/reference.json \
  --output artifacts/reference/qwen36-8bit/own --prefill
python3 scripts/compare-qwen36.py --directory artifacts/reference/qwen36-8bit
artifacts/baselines/mlx-env/bin/python scripts/benchmark-qwen36-8bit.py --label my-run
python3 scripts/summarize-qwen36-8bit.py
```

基准需要 Python tokenizers 做输入核对，不导入 MLX 执行推理。
已有结果须先归档，驱动器拒绝覆盖同名报告。GPU 工作负载串行运行。

最终 8bit 产物 SHA-256：

- Native AOT host：`c6f28d325c85ebad4ba5f2767a9bc490d8b05a2d73ce5069dab6c07ebe8cb81b`
- Metal dylib：`4ea6561b9436f626c34c539c6680e03ae7f340423aafd6e3d20ecc8928d0794c`

所有报告记录二进制、tokenizer、输入及输出散列。汇总脚本标记当前 dylib 对应的结果，
保留早期成功和失败的证据，不把旧二进制的测量合并为新版本成绩。

测试结束恢复原有 5100 端口的 5bit 服务，保留用户正在运行的 TUI；
8bit 保存在上述独立产物中。本次没有把旧启动器的默认模型悄悄改成 8bit。
