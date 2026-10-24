# 合并后 main Run：Pi 固定 20 题，2026-10-23

关联 [Issue #44](https://github.com/LittleLittleCloud/Infer.NET/issues/44)。用户 review 通过后，先合并结果 [PR #47](https://github.com/LittleLittleCloud/Infer.NET/pull/47)，再合并首页代码 [PR #46](https://github.com/LittleLittleCloud/Infer.NET/pull/46)，并确认 Pages 发布成功。用户随后要求补充合并后 main 的真实 Run。

## 实测结果

| 指标 | 本轮实测 | 覆盖 |
|---|---:|---:|
| Worker prefill | **531.199 token/s** | 92/92 请求 |
| Worker decode | **65.085 token/s** | 92/92 请求 |
| 客户端首个有效输出 TTFT 中位数 | **6986.970 ms** | 92/92 请求 |
| 官方逐题功能断言通过率 | **100%，20/20** | 固定完整 20 题 |
| Pi 整轮端到端任务耗时 | **711.369460043 s** | 20/20 题 |
| 累计输入 / 生成 tokens | **211383 / 19892** | 92 次模型请求 |

零任务超时、零 Pi 执行错误，20 份 `solution.py` 均通过隔离容器中的对应 HumanEval 官方断言。错误答案负对照实际以 `AssertionError`、退出码 1 失败，正对照退出码 0。每题耗时与累计 token 数保留在首页任务表；多轮重复上下文计入输入，不是净提示词长度。

Prefill 使用 `Σ inputTokens / Σ worker prefillSeconds`；decode 使用 `Σ(outputTokens−1) / Σ worker decodeSeconds`，不平均每请求 token/s。每个请求的输入/输出计数与 Pi 的独立 usage 逐项相符。两项速度与 TTFT 均达到整轮 92/92 采集覆盖。

这是 Pi 编码代理负载的一个真实 main 基线，不是全量 164 题榜单成绩，也不是 128K 正式性能验收。

## 版本、配置和观测边界

Mac 实际运行 coordinator、split-serve，以及 Windows split-agent 都重新构建、部署并核实为同一完整 main 源码：

`cd6f7da401fadf044dac13e87b415d27bb5eadfb`

Mac serve 在本轮期间 PID 保持 50216；Windows 实际运行代理 PID 为 69196，新包路径与版本已核对。原包和配置备份保留。Mac 两个组件由原用户 LaunchAgent 托管；Windows 沿用保留 SSH 会话的前台代理，没有新增 SCM 服务或管理员配置。新包校验为 Mac 9 项 / Windows 10 项；正式评测前稳定就绪后的真实部署验收 15 项通过。被测 source SHA 是实际二进制版本，不是本结果记录将来的合并 commit。

| 产物 | SHA-256 |
|---|---|
| Mac executable / Metal dylib | 见公开 observations.json 的 `binarySha256`，与包清单一致 |
| Windows executable | `76555a5f42392e8ff87bb8a59b8b8f794b41d4d063c10d1eefe293ee79636869` |
| Windows CUDA DLL | `30d789415cbd68e17fe329fd65dcf2d2c4aef7725666d8937e712e09b4e742eb` |

- 模型 Qwen3.6-35B-A3B；revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`；affine8/group64 权重，FP32 KV。
- 4090 前 20 层 prefill，Mac 后 20 层及完整 decode；CUDA `batch-fp32-packed`、chunk 32，suffix 128、Metal scratch 128，增量 KV，fused FP32 decode。短输入本机路由门槛为 0，全部正式请求均观测为 `cuda-metal-split`。
- Mac 插电、High Power，Windows stock 配置；Mac Wi-Fi / Windows Ethernet，原有 TLS 数据通道。未并行启动其他 GPU benchmark；没有声称桌面环境完全独占或全设备峰值资源验收。
- Pi 1.4.0；固定 seed 0 的同一 20 题；每题独立会话/目录；工具 read/write/bash/edit；thinking off；单题 300 秒，context 32768，单次输出上限 4096；自然 EOS。缓存读写均为 0。

当前 main 服务没有逐请求 `INFERENCE_METRIC` 日志。本轮使用两处独立 loopback 转发观测，产品的 .NET/CUDA/Metal 推理代码未改变：

`Pi → HTTP 观测客户端 → 同版本独立 .NET coordinator → 原始 worker stream 观测 → 实际 split-serve → TLS → 实际 split-agent`

Worker prefill/decode 数值直接取真实 `ServiceEvent.result.timings`；没有轮询 Dashboard 最近请求来代替逐请求记录。每次模型请求通过单并发窗口、task 起止时间、worker request ID 与 usage 对齐。采集器不存储请求正文、回答文本、凭据或生成代码的公开副本。

TTFT 从观测客户端开始向真实 .NET coordinator 发送已准备的请求，到收到第一段非空文本或有效工具调用内容，使用同一进程单调时钟。角色、心跳和 usage 帧不计。Pi 到观测器的派发在该边界之外；Pi 整题 wall time 则包含完整链路。当前工具调用 API 会缓冲输出，故 TTFT 可能接近该次生成完成时间，不能解释成 worker 内部首 token 时间。

新增观测跳数、packed/路由配置与旧历史不同，因此本轮使用独立可比配置；不将 711.37 秒与旧轮次总耗时之差归因于某项代码优化。旧 `e017d84` 记录和缺测状态完整保留。

## 证据、复现与页面验收

- [公开逐任务、逐请求及阶段数据](evaluations/main-cd6f7da-20261023/observations.json)：92 次真实 worker timings、客户端 TTFT、20 题时间窗口、官方断言结果、产物与本地原始文件散列。
- [新 Run 页面逐项核对](evaluations/main-cd6f7da-20261023/page-checks.json)、[桌面截图](evaluations/main-cd6f7da-20261023/home-desktop.png)、[320px 截图](evaluations/main-cd6f7da-20261023/home-mobile.png)：新 Run 默认选中、四项实测值、20 行任务时间/token 数、旧配置的缺测状态及深链接重载均通过，无脚本错误。站点构建为 3 页 / 46 篇笔记 / 3 个真实运行，18 项 Python 与 15 项 JavaScript 检查通过。
- [可复现的观测器](evaluations/main-cd6f7da-20261023/observe-main-run.py)：只将执行版本中的机器特定配置行改为参数；原实际执行文件的 SHA-256 留在原始证据索引中。使用本机已有受限配置目录与同源码 coordinator 包，端口 5260/5261/5262 须空闲。
- 原始任务 JSONL、Pi 事件、生成文件、HTTP 观测记录和失败的准备阶段记录仅保存本机忽略的 `artifacts/`；不是公开下载链接。公开数据不含私网地址、令牌或私有输入。

复现顺序：核实同 SHA 两端版本与空闲状态 → 部署验收 → 启动观测器 → 单条非题组预热并保留记录 → 执行固定 20 题 → 官方容器断言 → 对齐逐请求数据。命令入口：

```text
python3 docs/evaluations/main-cd6f7da-20261023/observe-main-run.py --config-dir PRIVATE_MAC_CONFIG --executable SAME_MAIN_COORDINATOR --output NEW_OBSERVER_DIR
python3 scripts/benchmark-pi-humaneval.py --cohort cohort --seed 0 --timeout 300 --context-window 32768 --max-tokens 4096 --endpoint http://127.0.0.1:5260 --endpoint-build cd6f7da401fadf044dac13e87b415d27bb5eadfb --model-revision e06a74e6236a60c8367e1a3214e83d8b61b637b0 --token-file PRIVATE_TOKEN_FILE --output NEW_RUN_DIR
python3 scripts/eval-pi-humaneval-docker.py --results ABSOLUTE_RUN_DIR --dataset PINNED_DATASET --output ABSOLUTE_RESULTS_JSON
```

准备阶段保留了三类失败及处理：首次 Metal LaunchAgent bootstrap 返回 5，确认该 job 未注册后仅清理本轮部分定义并重新安装；首次就绪边界验收一项未通过，稳定就绪后相同超限请求返回 400，重新执行完整 15 项通过；观测器最初未处理 chunked body 的请求返回 400，修正后 smoke 验证并在正式 20 题前完成。它们没有混入正式题组或删除正式慢轮次。

第一次容器调用使用相对挂载路径，Docker 以 125 退出，未执行任何断言；该错误不是 0/20 任务通过率。保留错误文件后改用绝对路径，同一份未修改答案完成有效 20 题测试，并额外核实正/负对照实际运行。未重新生成题目答案。

结果加入首页为默认选中的 main Run，按实际采集口径隔离旧配置。同步放宽了原测试中“生产数据永远只有两条”的假设：精确历史断言仍锁定原始运行 ID，新合法记录不再令测试误报；该测试调整由 Copilot CLI `gpt-6.1-sol` 完成，未改变生产绘图逻辑或推理后端。
