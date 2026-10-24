# 实验 #58：M5 Pro 全本地 Pi / HumanEval 20

测量日期：2026-10-24（America/Los_Angeles）。[Issue #58](https://github.com/LittleLittleCloud/Infer.NET/issues/58) · [代码 PR #59](https://github.com/LittleLittleCloud/Infer.NET/pull/59)。结果 PR：https://github.com/LittleLittleCloud/Infer.NET/pull/60。

用户先指定最新 main、固定 20 题、prefill 不使用 4090，随后明确批准增加全本地路由并测试候选 SHA。原 main 的聊天服务只允许 ≤128-token 本地 prefill，普通 worker 拒绝结构化聊天。候选把 `short-prefill-tokens` 可配置上限放开到模型窗口，显式设为 262144；默认 128、0 强制 split、模型精度与内核均保持原样。本轮没有与原始 main 做同配置 A/B，不给代码提速百分比。

## 版本与配置

- 固定最新 main 基线：`eaaf81f630614b4d6416bc62776c13d36331fa57`。
- 实际运行 coordinator / Mac serve / Metal 库：候选 `87c2b7dcaf4a418dcb18707e1913562717b43cdc`；Windows 不参与。代码 PR 后续 `19162c7` 仅更新集成测试，未冒充已测运行版本。
- M5 Pro / 48 GB，AC / High Power；独立 Native AOT 包与端口。原 Mac worker 空闲后通过 LaunchAgent 停用，测试结束恢复原包。无第二个模型 worker 争用 GPU。
- `mlx-community/Qwen3.6-35B-A3B-8bit`，revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`，affine8/group64，FP32 KV/状态，fused FP32 decode，Metal residency on。启动时逐文件验证固定权重散列。
- 保留部署路径的 managed prefill chunk 32、scratch 128；本轮仅改变路由，不切换到独立高吞吐 BF16/2048 实验配置。
- 固定 20 题、seed 0，Pi 1.4.0，新工作目录/会话；context 32768、每请求输出上限 4096、逐题 timeout 300 秒，单请求串行。每次请求清空状态，cacheRead/cacheWrite 均为 0。
- 官方数据 revision `b9453e8f1865df0a0057a1383767caa400acfbcf`，压缩文件 SHA-256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`。完整题目 ID 见公开配置与逐题表。
- 启动预热后另做两次 518-token 结构化聊天：均为 metal-local，输出 token IDs 一致；不计入 cohort。

## 结果

正式题组时间：10:51:21–11:11:24（America/Los_Angeles）。运行 ID：`issue58-m5pro-local-human20-20261024`。

|指标|实测|
|---|---:|
|Pi 完成 / 官方断言通过|20/20 · 20/20|
|超时 / Pi 错误|0 / 0|
|模型请求 / worker 指标覆盖|107 / 107（100%）|
|metal-local / CUDA 或跨机传输请求|107 / 0|
|输入 / 输出 tokens|243663 / 19104|
|prefill 总时间 / 吞吐|908.496 s / **268.205 token/s**|
|decode 间隔 / 总时间 / 吞吐|18997 / 285.955 s / **66.434 token/s**|
|客户端 TTFT 中位数 / 覆盖|10914.875 ms / 107/107|
|Pi 任务总耗时|1202.992 s（20.05 分钟）|

Mac executable SHA-256：`b4a15505fd2c86f0cd44185cd982eda666bfc5fff9a64e6f7cda494fbd2e9df8`；Metal library SHA-256：`1739ff5dd22a4b158f97d84d39268478a75d5683f4a9ad1a3cd0d1a8f79d4aef`。

prefill 使用 Σworker 输入 tokens / Σworker prefill 秒；decode 使用 Σ(N−1) / Σworker decode 秒。逐请求原始 `ServiceEvent.result.timings` 从真实 SSE 保存，并与每道 Pi 任务的请求数、输入/输出 token 数逐项对账；没有从 Dashboard 或 Pi wall time 推算吞吐。没有 CUDA 计算、传输或状态导入。

客户端 TTFT 由观察代理从转发前至实际收到首个有效文本/工具输出计时，忽略 role、心跳和 usage 事件。此配置增加两次 loopback 观察转发；Pi 到观察代理的派发不在 TTFT 区间内，因此该数字与其他计时边界分开解释。

## 正确性、资源与恢复

官方断言使用固定镜像 `python@sha256:5fb4bc53dd0180955e31e5245b648bbb24a4ac4ab03a1de81513642325220bcd`，逐题禁网、只读、无特权、限制 CPU/内存的独立容器。20/20 断言通过，错误实现负对照失败。首次调用由执行者传入相对路径，20 个容器均因 bind mount 路径不合法返回 125，未执行题目；保留 `functional-mount-error.json/.log`，不计为题目断言失败。使用同一批未修改的 solution.py 与绝对路径重新检查后全部通过；没有重新生成或挑选题目答案。

原 Mac LaunchAgent 已恢复，PID 改为 16997，build `0.3.0+cd6f7da401fadf044dac13e87b415d27bb5eadfb`、状态 Ready。候选模型、协调器、观察器和本轮 Pi 进程均已退出；未将候选包替换为常驻服务。

131 项 .NET 单元测试、44 项 Native AOT HTTP/进程集成检查通过。集成覆盖阈值 0/128/129/262144 的有效配置与 262145 的拒绝。模型配置/执行路径、重复长聊天状态 reset 与包散列验证通过。

整轮 10 秒资源采样均显示 AC；另在题组结束后约 0.117 秒、候选模型仍存活且原服务尚未重载时采集边界。Metal allocation 峰值 **34.730 GiB**。题组前后采样边界为 `2026-10-24T17:51:18.457138+00:00` 和 `2026-10-24T18:11:24.974177+00:00`，整机新增 swap-out **2.013 GiB**；加载/预热另有 **8.688 GiB**。这不是“无 swap”结果，也不能标为资源验收通过。swap-out 为整机计数，包含后台进程，不能归因于某一个进程；已有 swap 使用量不作为本轮新增值。

资源数值来自采样括住的题组区间，未取得逐请求精确 swap-out 或进程 footprint 峰值。实际 RSS 不作为 footprint 替代。内存压力与显著换页是这轮性能的限制条件；保留完整原始采样和所有正式请求，不排除慢样本。

这次是应用负载单次固定 cohort 测量，不代表 128K 正式目标验收，也不证明候选优于双机路径。不同拓扑、精度、分块或 TTFT 边界的历史点保留在不同配置系列，未伪造可比 A/B 基线。代码与结果均等待人工 review，不自动合并或关闭 Issue。

## 复现与证据

```sh
# 从代码 PR 的被测 SHA 构建、打包并验证后，在独立私有配置中设置：
# short-prefill-tokens=262144, kv=fp32, decode-attention=fused-fp32,
# metal-scratch-tokens=128, data-listen=loopback-only。
# 私有配置中 coordinator HTTP=5290、worker HTTP=5291，使用同一 token-file。
# 启动 coordinator / split-serve 并等待 ready 后运行观察器（5280/5281/5282）：
python3 docs/evaluations/issue58/observe.py --config-dir PRIVATE_MAC_CONFIG \
  --executable VERIFIED_PACKAGE/deepinfer --output NEW_RUN/cohort-observer
# 另一终端运行固定题组：
python3 scripts/benchmark-pi-humaneval.py --output NEW_RUN/cohort-20 \
  --token-file PRIVATE_TOKEN_FILE --endpoint http://127.0.0.1:5280 \
  --cohort cohort --seed 0 --timeout 300 --context-window 32768 --max-tokens 4096 \
  --endpoint-build 87c2b7dcaf4a418dcb18707e1913562717b43cdc \
  --model-revision e06a74e6236a60c8367e1a3214e83d8b61b637b0
python3 scripts/eval-pi-humaneval-docker.py --results NEW_RUN/cohort-20 \
  --dataset PINNED_HUMANEVAL_GZ --output NEW_RUN/functional.json
python3 scripts/import-issue58-local-humaneval.py --run-dir NEW_RUN --results-pr https://github.com/LittleLittleCloud/Infer.NET/pull/60
```

参数化观察脚本见 [observe.py](evaluations/issue58/observe.py)。公开数值见 [performance.json](evaluations/issue58/performance.json)，证据散列见 [evidence.json](evaluations/issue58/evidence.json)。逐题和逐请求数据保存在 [site/trends.json](../site/trends.json)。原始日志、Pi 会话与生成代码仅在本机忽略目录 `artifacts/humaneval20-m5pro-20261024-preflight/`，不是公开下载链接；公开摘要不包含凭据、私网地址、完整提示词或输出正文。

稳定测量点：`https://littlelittlecloud.github.io/Infer.NET/#experiment=humaneval20-m5pro-20261024&run=issue58-m5pro-local-human20-20261024`。该点在结果 PR 合入并且 Pages 部署成功后上线；本轮不自动发布。
