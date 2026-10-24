# 实验 #61：Splash 8bit 完整常驻 / HumanEval 20

测量日期：2026-10-24（America/Los_Angeles）。[实验 Issue #61](https://github.com/LittleLittleCloud/Infer.NET/issues/61) · [独立观测适配器 PR #62](https://github.com/LittleLittleCloud/Infer.NET/pull/62) · [独立结果 PR #63](https://github.com/LittleLittleCloud/Infer.NET/pull/63)。初始默认预算启动失败；用户授权提高系统 GPU 限额后，同一份 8bit 完整常驻模型完成固定 20 题，20/20 官方断言通过。保留初始失败和后续整轮原始数据，不挑选最快请求。

## 40GiB 完整常驻正式题组

用户执行 `sudo sysctl iogpu.wired_limit_mb=41984`，随后明确授权继续 8bit 完整常驻、结束恢复默认限额并合并结果 PR。实测 Metal recommended working set 为 44,023,414,784 bytes（41GiB），Splash `--max-memory 40G` 的实际 hard budget 为 42,949,672,960 bytes（40GiB）。这次没有修改 Splash 原生二进制，没有降低权重量化精度，也没有启用专家 SSD 流式。

运行 ID `issue61-splash-human20-resident40g-20261024`；正式题组时间 **11:50:00–11:54:31 PDT**。Splash 实际源码 `1482f6bbc2abf12375332725302019594150b6cb`；观测适配器来自 `b0bad3155468961ca0cdf6ee55cb710b515dfd7c`；Pi harness 为固定 main `ce151663b4182bcb8371c9d0160fa20b9f8b1274` 的原脚本，其散列保存在 manifest。Windows 不参与。

|指标|整轮实测|
|---|---:|
|Pi 完成 / 官方断言通过|20/20 / 20/20|
|超时 / Pi 错误|0 / 0|
|请求数 / 原生指标覆盖|107 / 107（100%）|
|累计输入 / 输出 tokens|333,970 / 22,814|
|原生 command prefill 总时间 / 加权速度|108.905243 s / **3,066.611 tok/s**|
|原生 command decode 总时间 / 加权速度|153.202860 s / **148.914 tok/s**|
|原生 decode cycle（含 cycle 开销）速度|147.723 tok/s|
|原生请求生命周期 prefill / decode 速度|3,058.366 / 147.032 tok/s|
|客户端有效首输出 TTFT 中位数 / 覆盖|850.675 ms / 107/107|
|Pi 任务总耗时|270.630 s（4.51 分钟）|
|SOP 首末 token 的 N−1 decode 速度|未取得等价的逐 token 生成时钟|

原生 command 速度采用逐请求前后 `/status.metrics` 计数差，按 Σtokens / Σcommand wall seconds 加权，**不是逐请求速度的算术平均**。107 次均核对 submitted/completed 各增 1；prefill 总计数与每题 Pi usage、正式请求数、输入和输出 tokens 全部一致。decode command 分子包含该 command 产生的全部 token；不冒称为 Infer.NET 的 N−1 首末 token 间隔。请求生命周期采用最终 SSE 返回的原始 `metrics.request_latency`，decode 分子使用 `metrics.decode.tokens`，并非自行假定 N−1。`performance-resident40g.json` 保留每次两套原始指标和 command 差值。

所有正式请求使用独立 `cache_salt`，最终 cache matched tokens 全部为 0；Pi cacheRead/cacheWrite 同为 0。DFlash2 开启，verify-rows 为默认 auto；原生提示此 MoE 模式下 greedy 输出可能不完全重复，本轮未证明重复确定性。temperature=0、reasoning_effort=none。保持固定 target/draft、Q8 KV、FP32 recurrent state、单请求串行，`--expert-memory off` 且所有快照的 `streamed_experts=null`。`--max-context 32768` 实际生效，完整 target 常驻。

实际每请求输入 1,771–11,527 tokens，中位数 2,341；输出 13–1,699 tokens。窗口上限 32K 不代表每次请求输入达到 32K。最长 Pi 任务耗时 91.366 秒，低于预设的 300 秒上限。

启动加载 7.14 秒后首次 Metal pipeline 编译约数分钟；加载、引擎启动 warmup 和额外 24 输入 /31 输出 token 的独立预热均不计入正式题组。较长的 HumanEval/32 Pi 自检会话及其全部请求保留，没有重试题组或换题。

## 正确性与资源

固定官方断言脚本对同一批未修改的 `solution.py` 使用 pinned Docker 镜像 `python@sha256:5fb4bc53dd0180955e31e5245b648bbb24a4ac4ab03a1de81513642325220bcd`，每题独立、禁网、只读、无特权、限制 CPU/内存。20/20 官方断言通过；错误实现负对照失败。Pi 完成状态没有被当作正确性通过率。

正式区间边界 vm_stat 显示整机新增 swap-out **268,697,600 bytes（256.25MiB）**；每 10 秒及请求边界采集的 Metal allocation 峰值为 **38,626,592,128 bytes（35.974GiB）**。该峰值来自采样，不冒充连续精确峰值或进程 footprint；系统采样期间有 warning 内存压力，温度状态保持 nominal，供电均为 AC / High Power。开始采样到题组前另观测到 swap-out 338,821,120 bytes（323.125MiB），但初次资源采样晚于权重开始加载，不能称为完整加载阶段换页总量。整机换页包含其他应用，不归因于单一进程。

正式题组没有记录模型请求失败、任务超时、崩溃或 OOM。没有执行独立跨后端 logits/NaN、128K 长上下文、固定 4096 输出或长请求后 reset 验收；这些不能由 20 题通过替代。本次为一轮应用参考，不是 128K 正式性能门槛验收。与历史 Infer.NET 的 FP32 KV、无推测解码、不同分块和计时范围不等价，不作单项优化加速百分比归因。

## 复现与交付

在原参数基础上增加 `--max-memory 40G`，先核对上述系统 wired limit 和实际 `/status.memory_plan.budget`。复用 [观测适配器说明](https://github.com/LittleLittleCloud/Infer.NET/blob/b0bad3155468961ca0cdf6ee55cb710b515dfd7c/docs/SPLASH-HUMANEVAL.md)，固定 cohort/seed0、Pi1.4.0、context32768、max output4096、timeout300。manifest、逐请求计数、逐题结果和公式原始分量见 [公开性能摘要](evaluations/issue61/performance-resident40g.json)、[题组 manifest](evaluations/issue61/manifest-resident40g.json)、[官方断言摘要](evaluations/issue61/functional-resident40g.json)。原始 Pi 会话/答案、完整引擎日志、status 和资源样本在本机忽略目录 `artifacts/splash-human20-20261024/`；公开证据散列标明仅本机可用。

测试完成已停止本轮 Splash 和 observer，恢复原 worker，核对原 build `0.3.0+cd6f7da401fadf044dac13e87b415d27bb5eadfb`、Ready、协调器注册和真实聊天成功。用户在终端恢复系统默认限额，执行者独立读取确认 **`iogpu.wired_limit_mb=0`**，并在默认限额下再次完成真实协调器聊天。先前 GUI 管理员辅助命令被取消（-128），保留日志，没有伪称该辅助命令成功；系统默认恢复以真实 sysctl 读回为据。详见 [restoration 摘要](evaluations/issue61/restoration-resident40g.json)。

结果 PR 只保存第三方参考，不将 Splash 标为 Infer.NET 产品 main 基线。首页独立系列的标准 prefill/decode 图不混入 command 口径数值；选中该运行可查看原生速度、TTFT、20 题明细与证据。独立观测工具 PR #62 保留 review，不在本次“结果 PR 合入”授权中自动合并；不影响保存本轮原始结果。

以下保留最初默认预算失败阶段的配置、数字及恢复证据；不是后续 40GiB 题组的测量数据。

## 范围与版本

用户明确授权在本机用 Splash 测固定 HumanEval 20 题、保存结果 PR，并允许暂停已有 worker。main 基线为 `ce151663b4182bcb8371c9d0160fa20b9f8b1274`；观测适配器候选为 `b0bad3155468961ca0cdf6ee55cb710b515dfd7c`（完整 SHA 见公开 environment 摘要）。Splash 是独立第三方参考，不成为 Infer.NET 产品执行后端。

- M5 Pro，20 核 GPU，48 GiB；MacBook Pro，macOS 与 AC / High Power 证据见公开摘要。
- Splash 1.4.0，完整源码版本 `1482f6bbc2abf12375332725302019594150b6cb`，官方 Homebrew release；原生二进制 SHA-256 `20b36bcc051145b425403dee7489d06cdbb5060460dfdb2e0cfc8d49cebfc29f`，Metal library `e30124e8bceba1a9de7afe75cf1a10439bcfa5c1b777490c829ba4da0d0cd1cd`。
- Target `mlx-community/Qwen3.6-35B-A3B-8bit`，revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`，affine8/group64；复用现有模型，21 个文件逐一通过固定 provenance 的 SHA-256 核查。
- 自动选择 `incoai/Qwen3.6-35B-A3B-DFlash2`；完整 draft revision `51ef7b6923ad6c14cb1bb41c37a9041446496ab6`，文件散列保存在证据摘要，下载约 1.05 GB。原生加载后的 draft 分配为 480,165,888 bytes。
- 语言模式、Q8 KV、FP32 recurrent state、DFlash2 默认自动验证；`--expert-memory off`，要求完整权重常驻；SSD prefix cache 关闭；请求的 context 上限 32768。
- 固定 HumanEval 数据 revision `b9453e8f1865df0a0057a1383767caa400acfbcf`，SHA-256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`。计划 Pi 1.4.0 / cohort / seed0 / 每请求4096输出上限 / 每题300秒。

## 初始默认预算失败（正式题组之前）

Splash 读取权重后，在创建最小状态/KV 工作区时由内存规划器拒绝启动，退出，未进入 Ready。原始诊断：

|项目|bytes|GiB|
|---|---:|---:|
|物理内存|51,539,607,552|48.000|
|Metal recommended working set|40,200,896,512|37.440|
|规划器预留 margin|1,073,741,824|1.000|
|hard budget|39,127,154,688|36.440|
|target weights|36,901,766,144|34.368|
|fixed runtime|39,013,279,104|36.334|
|最小 required|39,279,945,088|36.582|
|缺口|152,790,400|0.142|

错误为 `resource_assembly/memory_planning` / `kv_pool_does_not_fit`：`hard budget cannot fit one lane's state and the KV runway`，随后 `native protocol reached EOF`。这是确定的启动容量拒绝，不是题目超时、解码崩溃或已运行后 OOM。即使只满足最小 required，也不代表能装入 32K KV，不能从 0.142 GiB 缺口推断稍增内存就能完成本测试。

|评测项|状态|
|---|---|
|已执行题数 / 计划题数|0 / 20|
|正式模型请求|0|
|prefill / decode / TTFT|未取得|
|官方任务通过率|未验证，不填 0%|
|Pi 题组总耗时|未取得|
|128K 性能与正确性验收|未执行|

初始阶段没有为了取得速度而自动降低 target 精度、启用专家 SSD 流式、修改 Metal 系统限制或绕过引擎内存预算。随后用户明确选择提高系统限额并继续完整常驻；后续配置和结果见上文。没有执行专家流式题组。

## 观测适配器验证和计时限制

PR #62 的适配器复用已有 Pi harness，不修改模型提示词，向每次请求加入独立 `cache_salt`、temperature=0、reasoning_effort=none。合成 HTTP/SSE fixture 验证了两次请求的不同缓存命名空间、响应透传、usage/timing 保存、角色事件不计作首 token，以及观测日志不保存提示词或答案。该 fixture 只验证采集器，绝不作为推理性能数据。

Splash 的 `/status.metrics` 为原生 executor command 累计计数；串行独占时可用逐请求前后差值取得计算吞吐。最终响应的 `metrics.request_latency` 则是原生请求生命周期；DFlash2 第一次 emission 可包含多个 token，decode 分子为 `metrics.decode.tokens`，不能直接改成 N−1。标准 SOP 的首末 token N−1 速度没有等价观测时必须标为未取得。两套口径应分开报告，不能将 HTTP 块间隔或 Pi wall time 当作 worker timing。

## 恢复和证据

测试前核对原 Mac worker Ready、completedRequests=0，原 PID 16997；通过其原 LaunchAgent 停用后确认 PID 退出。Splash 失败退出后恢复原 LaunchAgent，已核对 Ready、原 build `0.3.0+cd6f7da401fadf044dac13e87b415d27bb5eadfb`；新 PID 28444 保存在本机原始日志。Windows 不参与本次测量，没有修改其常驻版本。

公开摘要与证据散列见 [issue61 目录](evaluations/issue61/)。原始安装/启动诊断、模型文件散列、环境与采集器 fixture 仅保存在本机忽略目录 `artifacts/splash-human20-20261024/`，不是公开下载地址。公开记录排除本机账户、硬件序列号、私网地址、凭据以及完整 Pi 提示词/输出。

初始失败本身没有性能数值，四项趋势图不填零点；后续完整常驻 20 题另记为独立运行，保存参考结果不代表第三方代码被产品采用或主线性能基线更新。
