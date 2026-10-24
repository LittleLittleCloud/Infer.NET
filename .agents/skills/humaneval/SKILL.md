---
name: humaneval
description: "用固定的 HumanEval 题组和 Pi 实测 Infer.NET 的 prefill、decode 性能，选择 5/20 题规模并记录运行版本与实验结果。仅用于 Infer.NET 的 HumanEval 性能评测。"
---

# Infer.NET HumanEval 性能评测

衡量真实 Pi 编码会话中的 Infer.NET 推理。**首要指标是 worker 的 prefill 与 decode 速度**；Pi 总耗时、请求数和工具调用开销另列。语法及官方断言可作健康检查，通过率不是本技能的主要优化目标，也不能用错误输出冒充有效的速度提升。

## 选择规模

- 小改动、冒烟检查、迭代调参：固定 5 题，`--cohort pilot`。
- 与 `main` 比较、决定是否保留优化、正式性能比较：固定 20 题，`--cohort cohort`。
- 默认不跑全量 164 题。只有用户明确要求全量时才用 `--cohort full`。

同一次 A/B 使用相同的官方数据集哈希、seed、题目 ID、Pi 版本、模型、上下文与输出上限、量化、缓存策略和尽量相同的设备/网络条件。默认 seed 为 0；5 题是 20 题的稳定子集。比较 `main` 时分别部署并核实实际运行的 Mac worker 与 Windows agent 完整 commit SHA；本地 checkout 的 HEAD 不代表正在服务的版本。不得并行启动会争用同一 GPU 的基准。部署及双机核查按 [Infer.NET 双机 skill](../infer-net-dual-host/SKILL.md) 执行。

## 执行与采集

在有效仓库 `/Users/littlecloud/repos/Infer.NET` 查当前脚本及参数。优先使用 `scripts/benchmark-pi-humaneval.py`，为每轮创建新的 `artifacts/` 目录，私密 token 只从本机受限文件读取。脚本的 `manifest.json` 固定题目与环境，`tasks.jsonl`、`summary.json` 保存 Pi 层结果。必要时先做一条非题组的预热请求，并排除预热。单次任务失败、超时和重试要保留，不挑最快结果。

确保 Mac split-serve 在整轮期间保存每请求 `INFERENCE_METRIC` 日志；它只含请求 ID、完成时间、build、输入/输出 token 数、prefill 毫秒、decode 间隔数/毫秒和执行路径，不含提示词。运行 `scripts/summarize-pi-humaneval-performance.py --run <本轮目录> --serve-log <服务日志> --mac-commit <已核实完整SHA> [--windows-commit <已核实完整SHA>] --change <改动简述> --output <本轮目录>/performance.json`。检查 `metrics_complete=true`、请求数与 Pi 相符、build 与预期一致；缺失时明确标成部分数据，修复采集后再声称整轮速度。旧服务没有此日志时先更新或另行获取等价的**逐请求 worker timings**；不能用 Dashboard 的 `lastRequest` 或 Pi 总耗时替代。

汇总速度用 `Σ输入 tokens ÷ Σprefill 秒` 和 `Σdecode 间隔数 ÷ Σdecode 秒`，不是简单平均每请求 tok/s。这里 prefill 包含跨机交接，decode 从首个到最后一个输出 token 计 `N−1` 个间隔；短到不足两个输出 token 的请求没有 decode 速度。另报 Pi 总耗时、题目完成数、请求数和 token 数，避免把应用层等待解释为 GPU 算子耗时。不要把不同精度、网络、执行路径或题组的速度直接归因于代码改动。

## 每轮必留记录

每轮结束后，把 `performance.json` 与原始日志保存在带运行 ID 的本地 `artifacts/`，并在仓库现有实验笔记或新的日期笔记中留下可追踪摘要，至少包括：运行时间/ID、目的与改动简述、**实际运行的 Mac 与 Windows 完整 commit SHA**（单机标明无 Windows）、题目数和 ID/seed、模型与权重/KV 精度、执行路径与网络、输入/输出 token 数、**prefill tok/s 与 decode tok/s**、指标覆盖率、Pi 总耗时及失败/正确性状态。没有可靠 worker 指标则写“未取得”，不可填估算。附本地原始证据位置或可访问链接，不公开 token、私网地址或私有提示词。

若是与 `main` 的 20 题比较，在同一记录内并排给出两端的上述数字、相对差值及条件差异；仅在配置与指标覆盖可比时写性能结论。HumanEval 属于应用负载，不替代 `docs/QWEN36-8BIT-TARGETS.md` 的 128K 正式性能和正确性验收。
