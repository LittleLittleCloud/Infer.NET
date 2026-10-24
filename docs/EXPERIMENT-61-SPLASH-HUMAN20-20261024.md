# 实验 #61：Splash 8bit / HumanEval 20 准备阶段失败

测量日期：2026-10-24（America/Los_Angeles）。[实验 Issue #61](https://github.com/LittleLittleCloud/Infer.NET/issues/61) · [独立观测适配器 PR #62](https://github.com/LittleLittleCloud/Infer.NET/pull/62)。本记录保存真实失败，不包含 HumanEval 实测速度。

## 范围与版本

用户明确授权在本机用 Splash 测固定 HumanEval 20 题、保存结果 PR，并允许暂停已有 worker。main 基线为 `ce151663b4182bcb8371c9d0160fa20b9f8b1274`；观测适配器候选为 `b0bad3155468961ca0cdf6ee55cb710b515dfd7c`（完整 SHA 见公开 environment 摘要）。Splash 是独立第三方参考，不成为 Infer.NET 产品执行后端。

- M5 Pro，20 核 GPU，48 GiB；MacBook Pro，macOS 与 AC / High Power 证据见公开摘要。
- Splash 1.4.0，完整源码版本 `1482f6bbc2abf12375332725302019594150b6cb`，官方 Homebrew release；原生二进制 SHA-256 `20b36bcc051145b425403dee7489d06cdbb5060460dfdb2e0cfc8d49cebfc29f`，Metal library `e30124e8bceba1a9de7afe75cf1a10439bcfa5c1b777490c829ba4da0d0cd1cd`。
- Target `mlx-community/Qwen3.6-35B-A3B-8bit`，revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`，affine8/group64；复用现有模型，21 个文件逐一通过固定 provenance 的 SHA-256 核查。
- 自动选择 `incoai/Qwen3.6-35B-A3B-DFlash2`；完整 draft revision `51ef7b6923ad6c14cb1bb41c37a9041446496ab6`，文件散列保存在证据摘要，下载约 1.05 GB。原生加载后的 draft 分配为 480,165,888 bytes。
- 语言模式、Q8 KV、FP32 recurrent state、DFlash2 默认自动验证；`--expert-memory off`，要求完整权重常驻；SSD prefix cache 关闭；请求的 context 上限 32768。
- 固定 HumanEval 数据 revision `b9453e8f1865df0a0057a1383767caa400acfbcf`，SHA-256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`。计划 Pi 1.4.0 / cohort / seed0 / 每请求4096输出上限 / 每题300秒。

## 实际结果

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

未为了取得速度而自动降低 target 精度、启用专家 SSD 流式、修改 Metal 系统限制或绕过引擎内存预算。是否允许专家流式读取已向用户提出；在答复前不执行依赖该选择的测量。流式读取即使权重仍为 8bit，也属于不同的执行条件，应单独固定配置和记录。

## 观测适配器验证和计时限制

PR #62 的适配器复用已有 Pi harness，不修改模型提示词，向每次请求加入独立 `cache_salt`、temperature=0、reasoning_effort=none。合成 HTTP/SSE fixture 验证了两次请求的不同缓存命名空间、响应透传、usage/timing 保存、角色事件不计作首 token，以及观测日志不保存提示词或答案。该 fixture 只验证采集器，绝不作为推理性能数据。

Splash 的 `/status.metrics` 为原生 executor command 累计计数；串行独占时可用逐请求前后差值取得计算吞吐。最终响应的 `metrics.request_latency` 则是原生请求生命周期；DFlash2 第一次 emission 可包含多个 token，decode 分子为 `metrics.decode.tokens`，不能直接改成 N−1。标准 SOP 的首末 token N−1 速度没有等价观测时必须标为未取得。两套口径应分开报告，不能将 HTTP 块间隔或 Pi wall time 当作 worker timing。

## 恢复和证据

测试前核对原 Mac worker Ready、completedRequests=0，原 PID 16997；通过其原 LaunchAgent 停用后确认 PID 退出。Splash 失败退出后恢复原 LaunchAgent，已核对 Ready、原 build `0.3.0+cd6f7da401fadf044dac13e87b415d27bb5eadfb`；新 PID 28444 保存在本机原始日志。Windows 不参与本次测量，没有修改其常驻版本。

公开摘要与证据散列见 [issue61 目录](evaluations/issue61/)。原始安装/启动诊断、模型文件散列、环境与采集器 fixture 仅保存在本机忽略目录 `artifacts/splash-human20-20261024/`，不是公开下载地址。公开记录排除本机账户、硬件序列号、私网地址、凭据以及完整 Pi 提示词/输出。

本轮尚未完成用户要求的 20 题测量。结果以无数值的失败实验记录展示，四项趋势图不填零点；保存这个结果不代表第三方代码被产品采用或主线性能基线更新。
