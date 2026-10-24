# Pi × HumanEval 固定 20 题复测：19/20，工具调用格式故障

日期：2026-10-23（America/Los_Angeles）。关联 [工具调用 Issue #32](https://github.com/LittleLittleCloud/Infer.NET/issues/32) 和 [评测工具 PR #42](https://github.com/LittleLittleCloud/Infer.NET/pull/42)。coordinator、Mac split-serve 和 Windows split-agent 来自 `e017d84e805c016d5912b5051e4c69f758056363`；新合并的 `15710714e537372a59b3c1ba999f31337a37017d` 仅提供评测脚本。它不是新推理引擎的性能测量。

## 结果

| 指标 | 本轮 |
|---|---:|
| Pi 任务完成 / 题数 | 19 / 20 |
| 语法有效解答 | 19 / 20 |
| 对应 HumanEval 官方 `test` 字段通过 | **19 / 20** |
| 超时 | 0 |
| 整轮串行 wall time | 774.288 秒 |
| 单题 wall 中位 / P90 | 31.798 / 64.703 秒 |
| 模型请求 | 103 |
| Pi 报告输入 / 输出 token | 241162 / 19949 |

失败题 `HumanEval/149` 在首个模型请求中用约 9.039 秒返回 `stopReason=error`、`errorMessage=terminated`，无 `solution.py`。同时 coordinator 日志记录 `System.IO.InvalidDataException: Malformed model tool call`，栈位于 `DeepInfer.Runtime.ChatTools.ParseOutput` → `DeepInfer.Host.CompatibleChat.Writer.End`。单独重试相同题目再次以同一异常失败，耗时 9.068 秒。因此本轮 19/20 是实际端到端结果，不是代码答案被断言判错；单题重试**未**拼入 20 题统计。服务随后保持 Ready、agent connected，其余题正常完成。

失败题提前结束，整轮 wall time 不能用于推断 worker prefill/decode 速度。逐题数据见 [公开摘要 JSON](../site/pi-humaneval-rerun-summary.json)。这仍是固定 20 题、有工具的 Pi 编码任务小样本，**不是**完整 164 题官方榜单 pass@1，也不是 128K prefill/decode 验收。

## 配置与验证

- 数据集：OpenAI HumanEval revision `b9453e8f1865df0a0057a1383767caa400acfbcf`，压缩文件 SHA-256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`；seed 0、固定相同的 20 题及顺序。Pi 1.4.0、每题独立工作目录与会话、300 秒单题超时、32768 context window、4096 单次输出上限、`read/write/bash/edit` 工具。
- 模型：Qwen3.6-35B-A3B revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`，affine 8bit 权重、**FP32 KV**。Mac M5 Pro 48 GiB 完整模型与 decode，RTX 4090 常驻前 20 层；Wi-Fi 跨机。Mac 插电、High Power。评测前 worker 为 Ready、agent connected，Windows GPU 空闲，完成后服务仍健康。用户同时打开了 TUI，因此不能声称环境完全独占。
- `scripts/benchmark-pi-humaneval.py` 在 `main` 的 SHA-256 为 `594f60571fabc5d05d7aa849cfd25ad668a3ee8fed7c5f278b6a7ea47b9d9c22`。本地离线 fixture 12 项通过、2 项 Windows 专用跳过；合并后 GitHub CI 的 Linux、Windows、macOS 任务均通过。
- 完整 `solution.py` 在断网、只读、无额外 capability、限 CPU/内存/PID、非 root 的 Docker 容器中分别运行对应的官方 `test` 与 `check(entry_point)`；19 份均通过。故意错误的负对照失败。评分脚本为 [`scripts/eval-pi-humaneval-docker.py`](../scripts/eval-pi-humaneval-docker.py)，容器镜像固定为 `python@sha256:5fb4bc53dd0180955e31e5245b648bbb24a4ac4ab03a1de81513642325220bcd`。模型生成文件不在宿主机执行。

复现命令（从仓库根目录，在隔离的本地环境中使用私有 token 文件和已启动的同版服务）：

```sh
python3 scripts/benchmark-pi-humaneval.py \
  --cohort cohort --seed 0 --timeout 300 --context-window 32768 --max-tokens 4096 \
  --dataset /PRIVATE/HumanEval.jsonl.gz --token-file /PRIVATE/token.txt \
  --endpoint-build e017d84e805c016d5912b5051e4c69f758056363 \
  --model-revision e06a74e6236a60c8367e1a3214e83d8b61b637b0 \
  --output /NEW/IGNORED/RESULT_DIRECTORY

python3 scripts/eval-pi-humaneval-docker.py \
  --results /RESULT_DIRECTORY --dataset /PRIVATE/HumanEval.jsonl.gz \
  --output /RESULT_DIRECTORY/functional-tests.json
```

原始事件、生成文件和完整评分日志保存在本机忽略的 `artifacts/humaneval-main-rerun-20261023/`，单题复现保存在 `artifacts/humaneval-149-retry-20261023/`，不提供公开下载。SHA-256：本轮 `manifest.json` `d7c01e21a28780b5f4a4d27b23826908f757fbd9a06f943d6b90a47a15984aad`，`tasks.jsonl` `7080013309968687a167413378f397e9357fc9a1815be3fbafabecdb1344277b`，`summary.json` `b4049089b744136dc06f777e71fe34e35cabc45f86595ac2b3e056c48d97617e`，`functional-tests.json` `0cb18ffb96d20a2a339765d5b9c1fb9e16fef356eae67773ec95f1f5851656b4`；单题重试 `tasks.jsonl` `a2dff920fcf109dc961976ee8a50b769d885f87762e8b637b39473150ff0d57c`。

下一步是定位 `ChatTools.ParseOutput` 无法接受的模型输出形态，并为真实模型输出与错误响应加回归覆盖。修复后应完整重跑同一 20 题，而不是用单题重试替换失败样本。
