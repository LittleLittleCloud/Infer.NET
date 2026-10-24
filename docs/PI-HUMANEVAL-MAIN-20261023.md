# Pi × HumanEval 固定 20 题：main 部署基线（2026-10-23）

关联 [Issue #40](https://github.com/LittleLittleCloud/Infer.NET/issues/40) 与 [PR #42](https://github.com/LittleLittleCloud/Infer.NET/pull/42)。这次测量的三个**运行中**组件均来自源码提交 `e017d84e805c016d5912b5051e4c69f758056363`，而不是仅将 Git checkout 切到 `main`。Mac coordinator 与 split-serve 使用同一 osx-arm64 包；Windows split-agent 使用同源码 win-x64 包。旧包保留供回退。

## 实测结论

- Pi 编码任务 **20/20 完成**、**20/20 语法有效**；零超时、零脚本错误。
- 在隔离、断网、只读容器里，将各完整 `solution.py` 运行于对应 HumanEval 官方 `test` 字段，**20/20 通过**。本轮共涉及 139 条 `assert`；故意错误的负对照失败。此为固定 20 题的 coding-agent full-file 测试，不是 164 题官方榜单 pass@1。
- 串行整题 wall time：总计 **808.884 秒**（13 分 28.884 秒）；单题中位 **34.497 秒**，P90 **61.569 秒**，最快 **20.309 秒**，最慢 **83.232 秒**。P90 是 20 个样本的第 18 个顺序统计量。
- Pi 共发起 **98 次模型请求**；累计输入 **222744**、输出 **19212** token，缓存读取/写入均为 0。输入计数包含多轮重复上下文，不等于 20 个独立提示词的净长度。

## 固定配置与计时边界

- 模型 Qwen3.6-35B-A3B、revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`；权重 affine 8bit，KV **FP32**。这不是 Q8 KV 目标验收。Mac M5 Pro 48 GiB 完整模型常驻并 decode；Windows RTX 4090 前 20 层常驻；Mac 到 Windows 的数据链路此轮经 Wi-Fi。Mac 插电，`pmset powermode=2`。Windows agent 为 SSH 保持的前台进程，SCM 仍未安装。
两端包文件 SHA-256：Mac `deepinfer` `01a9777455e9de54fb23e06259390cd159a58fa8f5cc192ff13082752067d151`，Metal dylib `1739ff5dd22a4b158f97d84d39268478a75d5683f4a9ad1a3cd0d1a8f79d4aef`；Windows `deepinfer.exe` `a308532aca865617ff46c2007daa4b2fcc6be54db4cc66c71e7b56ce09fa1aa5`，CUDA DLL `b9abdf0dee1517874654bbbdb6efb54eb26e4927131d3854f5c70f821cf09311`。

- Mac serve 配置：短输入本机路由门槛 128 token、suffix chunk 128、Metal scratch 128、增量 KV 传输。升级时删去未合入 main 且原值为 `off` 的 `suffix-receive-overlap` 旧配置键；其余模型/鉴权配置复用。
- Pi 1.4.0，每题独立临时工作目录与会话，工具 `read,write,bash,edit`，thinking 关闭；20 题按 seed 0 哈希排序固定选择，单题 300 秒超时，context window 32768、单次输出上限 4096。运行前真实模型部署验收 15/15 通过，正式任务串行。
- Wall time 从 Pi 子进程启动到退出，含所有模型请求、prefill、decode、工具、文件操作和本机调度；不把它叫作 GPU 推理时间。20 题不是硬件性能极限或 `docs/QWEN36-8BIT-TARGETS.md` 的 128K 性能验收。

## 每题原始计时摘要

| ID | wall 秒 | 请求数 | 输入 token | 输出 token | 断言 |
|---|---:|---:|---:|---:|---|
| HumanEval/156 | 36.475 | 5 | 10989 | 1030 | passed |
| HumanEval/94 | 29.970 | 3 | 7293 | 1064 | passed |
| HumanEval/72 | 23.436 | 3 | 6314 | 657 | passed |
| HumanEval/80 | 22.064 | 3 | 5866 | 636 | passed |
| HumanEval/149 | 45.221 | 6 | 14558 | 1219 | passed |
| HumanEval/2 | 22.920 | 4 | 7447 | 436 | passed |
| HumanEval/35 | 49.374 | 4 | 7580 | 557 | passed |
| HumanEval/32 | 49.713 | 4 | 11557 | 1655 | passed |
| HumanEval/146 | 64.704 | 7 | 16882 | 1395 | passed |
| HumanEval/137 | 83.232 | 9 | 23691 | 1442 | passed |
| HumanEval/97 | 38.923 | 5 | 10166 | 804 | passed |
| HumanEval/154 | 61.569 | 9 | 22083 | 1181 | passed |
| HumanEval/6 | 27.591 | 4 | 8091 | 681 | passed |
| HumanEval/161 | 57.400 | 7 | 16449 | 1166 | passed |
| HumanEval/115 | 32.518 | 3 | 6899 | 921 | passed |
| HumanEval/131 | 29.482 | 5 | 9771 | 577 | passed |
| HumanEval/162 | 29.912 | 4 | 7912 | 712 | passed |
| HumanEval/50 | 20.309 | 3 | 5662 | 505 | passed |
| HumanEval/47 | 56.943 | 7 | 16965 | 1662 | passed |
| HumanEval/148 | 27.127 | 3 | 6569 | 912 | passed |

## 复现与证据

- 数据集：OpenAI HumanEval 提交 `b9453e8f1865df0a0057a1383767caa400acfbcf`，gzip SHA-256 `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`；任务 ID 顺序与逐题摘要见 [`site/pi-humaneval-main-summary.json`](../site/pi-humaneval-main-summary.json)。
- Pi harness 来自 PR #42 提交 `c5913cf354ce45614e31af6d579de355012261ea`，脚本 SHA-256 `594f60571fabc5d05d7aa849cfd25ad668a3ee8fed7c5f278b6a7ea47b9d9c22`。harness 不在本次被测 `main` 的运行代码内；只有测试驱动使用它。
- 复现：在两台设备部署上述 `main` 源码的已验证包并等到双级 Ready；从 PR #42 的该提交运行 `python3 scripts/benchmark-pi-humaneval.py --cohort cohort --seed 0 --timeout 300 --context-window 32768 --max-tokens 4096 --token-file <本机私有令牌文件> --output <全新本机结果目录>`。harness 自行获取并校验固定数据集。每次运行须用新结果目录。
- 原始 JSONL、事件、生成文件及功能测试结果仅存本机受忽略的 `artifacts/main-eval-20261023/artifacts/benchmark-humaneval-20-main-e017d84-20261023/`，并非公开下载链接；公开摘要只含任务 ID、计时和非秘密元数据。原始文件 SHA-256：`summary.json` 29fc26d8386b84705ed150d29bba55a2a270d33ba93012c8a048875875e9dde7；`tasks.jsonl` 71dd0e250de2db22e21106556fceea2c19a4cb9af2b8ec3e82ade09fb688a1a4；`functional-tests.json` 5d45d7b6e707bc209ccc91b5f927b5a69ddfec51a8108f2c04cc47aec4c14274。
- 构建与验收：Mac 128/128 单元测试，Windows 127/128 通过、1 项平台跳过；Windows CUDA 数值检查通过；两端 Native AOT 包 9/10 文件逐项校验；真实模型部署验收 15/15 通过。升级后 serve PID 保持不变地完成整个 20 题序列。

本次给出 `e017d84` 的**固定 20 题 Pi 端到端基线**。不同选题、网络、缓存、工具策略与并发条件会改变结果；后续版本应在相同 20 题和同等条件下 A/B 比较。
