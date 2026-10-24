# Infer.NET — Qwen3.6-35B-A3B

[项目进展](https://littlelittlecloud.github.io/Infer.NET/) · [实验笔记](https://littlelittlecloud.github.io/Infer.NET/experiments.html) · [任务与验收](https://github.com/LittleLittleCloud/Infer.NET/issues/1)

站点内容由 `site/` 与固定性能规范生成，运行 `python3 scripts/build-progress-site.py` 可在 `artifacts/pages/` 预览。推送相关变更到 main 后由 GitHub Actions 发布。项目名称为 Infer.NET；现有 `DeepInfer.*` 文件名与命令保留兼容。

**唯一支持的真实模型是 Qwen3.6-35B-A3B。** CLI、demo 和默认启动器均选择该模型，默认权重为 `models/qwen3.6-35b-a3b-8bit-mlx`。同一模型的 5bit 与 8bit 检查点继续支持；Qwen2.5 3B/7B、TinyStories 的加载器、下载器及启动入口已移除。`synthetic-v1` 仅用于不加载权重的 RPC 测试。

产品通过自写 C# / Metal / CUDA 路径执行推理，不调用 Python、Transformers、MLX、llama.cpp 或外部模型服务。Python 仅用于下载、独立参考和验收脚本。

## 直接启动

实验双机入口已在 `codex/split-prefill` 分支实现（[PR #8](https://github.com/LittleLittleCloud/Infer.NET/pull/8)）：4090 常驻前 20 层，Mac 执行后 20 层，接收完整状态后接管 decode。8bit 权重、**FP32 KV 诊断模式**的 17/8、61/32 和双端 Native AOT 验证通过；BF16 目标路径尚未通过严格数值门槛，当前输入加输出最多 2K，尚未接入 TUI。准备两端环境与权重后，在 Mac 执行：

```sh
python3 scripts/split/run-dual.py --native --verify --kv fp32
```

完整前提、构建、证据和失败记录见 [双机实验说明](docs/SPLIT-PREFILL.md)。这不是 128K 性能验收。

在这台 M5 Pro 48GB 上双击 **`DeepInfer.command`**，或：

```sh
./scripts/mvp.sh
```

首次使用先下载固定版本的 8bit 权重：

```sh
python3 scripts/download-qwen36-mlx.py --bits 8
./scripts/mvp.sh
```

下载环境需要 `huggingface-hub`。启动器启动一个协调器、一个 Qwen3.6 Metal Worker 和终端界面；Ctrl+C 退出并清理子进程。可用 `DEEPINFER_BASE_PORT=5500` 覆盖默认端口 5100–5101，或用 `DEEPINFER_WEIGHTS=models/qwen3.6-35b-a3b-5bit-mlx` 显式选择该模型的 5bit 权重。已有原生产物时使用 `artifacts/` 下的程序，否则构建托管版；修改源码后请重新构建或发布产物。

`DeepInferMoE.command` 保留为同一模型的 M5 调优入口，也使用默认 8bit 权重。显式增加 `--workers` 会加载额外完整模型副本，应先确认内存容量。

## Qwen3.6-35B-A3B：自写 Metal / C# 引擎

**8bit 性能目标 v1 已固定**：128K 无前缀缓存，双机端到端 prefill 最终目标 **2000 token/s**，128K + 4096 输出整次请求 **≤ 180 秒**，持续 decode **≥ 37 token/s**。阶段验收线为 1600 token/s / 200 秒；均为待实现目标。模型、计时、复测、正确性与内存门槛见 [性能验收规范](docs/QWEN36-8BIT-TARGETS.md)。

**8bit 新基线（独立产物）**：自写 Metal 已完成 8bit 权重支持与实际模型校验。M5 Pro 48GB、AC / High Power，8K / 32K / 128K 输入的稳态 prefill 分别为 **3.467 / 17.743 / 135.966 秒**，解码 **66.34 / 60.51 / 37.52 token/s**。**128K + 4096 输出**使用 512 分块完整跑通：prefill **164.157 秒**、持续解码 **38.95 token/s**。拆开最终词表投影降低单次提交的驻留峰值；长输出还需缩小工作区。整机确有换页，不能视为无内存压力。逐层校验、68 项单测及 4 项长请求后 RPC 验收通过；8bit 的复现、失败记录、内存限制与 20/20 双机容量/收益估算见 [8bit 验证报告](docs/QWEN36-8BIT.md)。当前启动器默认使用 8bit；以下保留此前 5bit 测量。

**M5 调优配置（5bit 历史测量）**：专家尾块自适应 16/32/64、gate/up 合并矩阵乘法、DeltaNet 直接矩阵读取。AC / High Power、相同二进制与精度、每档预热三次测三次，8K prefill **3.468 → 3.353 秒（减少 3.32%）**，32K **17.850 → 17.344 秒（减少 2.83%）**。两档 A/B 全部 128 个输出 token 一致，显存分配和提交次数不变。新鲜 MLX 对照 **2.967 / 14.701 秒**，prefill 仍慢 **12.99% / 17.98%**，尚未持平；本轮解码为 **84.76 / 74.88 token/s**。287 项 Metal、68 项单元、25 项 RPC 和 4 项真实模型检查通过。`DeepInferMoE.command` 已选择新配置，详见 [实现与完整证据](docs/PREFILL-EXPERT-DELTA.md)。下文保留历史轮次，各自条件不能混用。

此前专家中间布局实验：已实现 gate/up 直接输出 down 所需 BF16 布局，8K 同轮 prefill 耗时降低约 **1.4%**；32K 初测及补测分别慢约 **1.1% / 0.5%**，因此保留原默认路径。反量化双缓冲在 8K 反而慢约 **7.1%**，未启用。精度及输出不变，详见 [实现、High Power 实测与验证](docs/EXPERT-FLOW.md)。

双击 **`DeepInferMoE.command`**，或运行 `./scripts/mvp.sh`。先用 `python3 scripts/download-qwen36-mlx.py --bits 8` 获取固定版本的量化权重；该下载器的 Python 环境需要 `huggingface-hub`，运行产品无需 Python。

支持该系列检查点的 5-bit 或 8-bit/group64 affine 权重、8-bit router、256 专家 Top-8、共享专家、30 层 Gated DeltaNet 与 10 层 gated full attention。默认完整 forward 在自写 Metal GPU 执行器中完成，包括量化 GEMV、专家 SwiGLU、归一化、卷积、RoPE、两种注意力、路由和 argmax；C# 负责分词、生成控制和 RPC。FP32 激活、状态和累加。**没有调用 MLX、llama.cpp、Ollama 或外部模型服务。** `mlx` 仅出现在所读取的检查点格式/目录名中。

**上一轮 M5 配置（混合精度，历史测量）**：相同五位权重、相同输入、固定 128 输出、预热一次再测三次。8K prefill **3.734 → 3.491 秒**，32K **19.917 → 17.979 秒**，耗时分别降低 **6.5% / 9.7%**；本轮 MLX 复测为 **2.962 / 14.666 秒**，仍未达到 prefill 持平。8K 解码 **84.70 / 84.37 token/s（自写 / MLX）**，32K 为 **74.84 / 71.46**。优化专家分块与线程数、注意力寄存器占用，保持上一版精度边界；68 项单元、511 项 Metal、25 项 RPC 和 4 项真实模型验收通过。详见 [该轮实现、限制与原始证据](docs/PREFILL-TUNING.md)。下列早期测量保留各自配置，不能混用。

**后续插电验证**：模型级 prefill 分块为 **2048**；128 是 Attention query 调度批次。128/512/1024 每档、每种长度均完整预热三次再测三次，132 次供电观测均为 AC，输出逐 token 一致。保留 128 时，8K/32K prefill 均值 **3.431 / 17.856 秒**，解码 **86.13 / 74.90 token/s**。扩大到 512/1024 在 8K 略慢、32K 仅快约 1.2%/0.8%，分别多占 390/910 MiB 的 32K 缓冲区，因此默认保持 128。该轮未重测 MLX，不能与上段历史数据作新的持平结论。详见 [分块层级、AC 复测与证据](docs/ATTENTION-BATCH.md)。

2026-10-21 本机 M5 Pro 20 核 GPU / 48GB 短输入复测（当时 32-token 分块），worker 首 token **217.1 → 92.3 ms**，缩短 **57.5%**，解码保持 **96.60 token/s**。相同 24-token 输入、128-token 输出、各预热一次测三次：MLX **FP32 首 token 93.9 ms / 解码 85.27 token/s**；MLX 默认 **BF16 为 87.3 ms / 90.59 token/s**。同精度 FP32 的短输入首 token 已基本持平，BF16 的首 token 仍快约 5 ms。128 个输出 token 与 MLX FP32 完全一致。所有权重继续紧凑存储，激活/状态/累加保持 FP32；没有通过降低精度或缓存已有答案提速。每个解码 forward 为 **675 个 kernel / 1 次提交与等待**；整段 24-token prefill 也只需一次提交。

```sh
./artifacts/native/osx-arm64/deepinfer demo --model qwen3.6-35b-a3b --stay
# 另一个终端：
./artifacts/native/osx-arm64/deepinfer infer --model qwen3.6-35b-a3b --prompt "用一句话介绍你自己。" --json
python3 scripts/benchmark-qwen36-own.py
```

首轮长输入优化（同日，128-token 分块和自写 FP32 融合注意力）：相同 8K 输入三次测量，prefill 平均 **37.87 → 26.92 秒（1.41×）**；相同 32K 输入一次历史对照为 **257.64 → 126.73 秒（2.03×）**。8K 的 decode 均值为 69.9 → 67.5 token/s，本轮没有解码提速结论，也没有新的长输入 MLX 持平结论。原始数据、输出差异、正确性校验和复现步骤见 [长序列基准](docs/LONG-CONTEXT-BENCHMARK.md)。

此前的 FP32 优化已默认启用所选专家批量 GEMM 和分块 DeltaNet，并将外层 prefill 块扩大到 2048 tokens。8K 输入三次均值 **26.57 → 15.87 秒（1.67×，516 token/s）**；32K 一次历史对照 **126.73 → 81.96 秒（1.55×，400 token/s）**。本轮两种输入的 128 个输出 token 均与各自对照一致；FP32 精度不变。主要增益来自专家 GEMM，DeltaNet 的独立变化仅约 1.4%，不宣称显著提速。270 项 Metal 检查、68 项单元测试及 RPC 端到端验收通过，尚未宣称达到 MLX prefill 水平。详见 [批量 GEMM 与 DeltaNet 实现及测量](docs/PREFILL-GEMM-DELTA.md)。

目前为单机 GPU 执行基线：默认最多 2048-token 分块批量 prefill、所选专家分组 GEMM、分块 DeltaNet、长输入使用自写融合注意力、贪心非思考模式；已开放配置中的 262,144 token 总窗口、最多生成 32,768 token，输入输出共用预算。开放容量不代表最大长度已经实测；长上下文测试见 [长序列基准](docs/LONG-CONTEXT-BENCHMARK.md)。不支持图像、MTP、YaRN 或跨设备模型切分。正确性验证、实测速度和复现方式见 [自写引擎基线](docs/QWEN36-OWN-BENCHMARK.md)。

## Pi + HumanEval wall-time harness

`scripts/benchmark-pi-humaneval.py` measures the **real Pi CLI** using
`infernet/qwen3.6-35b-a3b`, serially, with standard read/write/bash/edit tools.
It never starts, restarts, or deploys a server. Python 3.9+ and a separately
installed Pi are required; no Python packages are needed.

**Dataset pin:** official [OpenAI HumanEval](https://github.com/openai/human-eval),
revision `b9453e8f1865df0a0057a1383767caa400acfbcf`,
[`data/HumanEval.jsonl.gz`](https://raw.githubusercontent.com/openai/human-eval/b9453e8f1865df0a0057a1383767caa400acfbcf/data/HumanEval.jsonl.gz),
compressed-file SHA-256
`b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`.
The runner verifies the digest and all 164 IDs before running any task. Use
`--dataset /absolute/path/HumanEval.jsonl.gz` for an offline copy of those exact
bytes. The dataset is not bundled; its upstream MIT license applies.
Only task IDs and prompts are retained after loading; tests and reference
solutions are never supplied in Pi's prompt or workspace.

### Reproduce

Use a disposable environment with only the intended local Infer.NET endpoint
reachable. Pi's bash tool runs generated commands: **a fresh workspace is not
a security sandbox**. Do not run this with unrelated credentials, private
files, or a local HumanEval checkout accessible to the agent. No official
evaluator or generated solution is executed by the harness; Pi itself may run
commands or syntax checks while solving the task.

The Pi CLI interface was checked against upstream
[`136b87e8a7f5a3e5186de0289523ba327dc592fc`](https://github.com/badlogic/pi-mono/tree/136b87e8a7f5a3e5186de0289523ba327dc592fc/packages/coding-agent)
(package version 1.4.0). Install a Pi release supporting the recorded flags,
including `--no-context-files` and `--no-approve`; unsupported flags are task
failures, not grounds to silently weaken isolation. Keep the **same exact Pi
version** for baseline and candidate. `--pi /absolute/path/to/pi` selects the
executable. On Windows, where npm normally installs a `.cmd` wrapper, pass
`--pi C:\absolute\npm\node_modules\@earendil-works\pi-coding-agent\dist\cli.js`
(use the actual installed package path). The harness invokes that entry point
with `node` from PATH, without shell quoting of prompts. `.cmd`/`.bat` wrappers
are rejected. Bash must also be available for Pi's bash tool.

Start and validate the authenticated local server separately using its normal
deployment instructions. Its OpenAI-compatible chat endpoint must support
tool calls. Supply the existing token **file path**, never a token in command
arguments. The runner does not read your normal Pi config: it generates a
fresh `models.json` for the loopback endpoint using the
`$INFERNET_BENCH_TOKEN` key reference and explicit HTTP authentication. It disables
discovery of extensions/MCP/skills/context files,
disables retry/compaction, uses `--thinking off`, and uses `--no-session`.
Each task gets a new temporary workspace, home, and agent directory, removed
after collection. The child environment is restricted, but the local API
token is necessarily available to Pi (and hence its tools).

The generated `infernet/qwen3.6-35b-a3b` model uses `openai-completions` with
explicit endpoint compatibility: `maxTokensField="max_tokens"`,
`supportsDeveloperRole=false`, `supportsReasoningEffort=false`, and
`supportsStore=false`. The current Infer.NET endpoint requires `max_tokens`
rather than `max_completion_tokens`, does not support the developer role, and
rejects `reasoning_effort` and `store` parameters (including `store=false`).
These settings prevent Pi from sending unsupported parameters; they match the
working Pi model configuration without reading or modifying your normal config.
The offline fixture checks these fields in each task's generated `models.json`.

From the repository root (replace paths/build labels with verified local values):

```sh
python3 scripts/benchmark-pi-humaneval.py \
  --cohort pilot --seed 0 --timeout 600 \
  --endpoint http://127.0.0.1:5200 --token-file /absolute/private/token \
  --endpoint-build VERIFIED_BUILD --model-revision VERIFIED_MODEL_REVISION \
  --output artifacts/pi-humaneval/baseline-pilot

# Run only after reviewing pilot evidence; use a fresh directory every time:
python3 scripts/benchmark-pi-humaneval.py \
  --cohort cohort --seed 0 --timeout 600 \
  --token-file /absolute/private/token --output artifacts/pi-humaneval/baseline-20

python3 scripts/benchmark-pi-humaneval.py \
  --cohort full --seed 0 --timeout 600 \
  --token-file /absolute/private/token --output artifacts/pi-humaneval/baseline-164

# Explicit IDs override cohort selection:
python3 scripts/benchmark-pi-humaneval.py \
  --ids HumanEval/0,HumanEval/1 --timeout 600 \
  --token-file /absolute/private/token --output artifacts/pi-humaneval/explicit

# Offline fixtures, also run in CI; no Pi, model, network, or GPU:
python3 scripts/test-pi-humaneval.py
```

Selection sorts IDs by SHA-256 of UTF-8 `SEED:HumanEval/N`, ascending;
`pilot` takes 5, `cohort` takes 20, and `full` takes 164. Thus the pilot is
contained in the fixed 20-task cohort for the same seed. Explicit IDs preserve
the supplied order. The seed controls **selection only**, not server sampling.
Defaults are a 32,768-token context declaration and 4,096 output tokens per
request; set `--context-window` / `--max-tokens` to the real endpoint limits.
Do not change them between comparisons. Build/model revision fields default
to `null`: supply verified non-secret provenance when available, not guesses.

### Evidence and interpretation

The new output directory contains:

- `manifest.json`: exact source/digest, ordered IDs, seed, Pi version,
  endpoint/build/model revision, limits, timeout, and harness digest.
- `tasks.jsonl`: one record per attempted task, including failures/timeouts;
  prompt hashes (both dataset and complete user prompt), exact Pi command,
  version/provider/model, UTC start/end and monotonic process wall time,
  return/finish status, authoritative assistant-message usage, request and
  turn timings, tool lifecycle records, solution hash and syntax status.
- `HumanEval-N/events.jsonl`: raw Pi event objects wrapped with harness-observed
  monotonic receive offsets; stderr and malformed lines are retained separately.
  API credentials are redacted before writing, including JSON-escaped values.
  Streaming usage snapshots are **not** summed: only assistant `message_end`
  usage is counted, once per response, not duplicate `turn_end`/`agent_end` data.
- `HumanEval-N/solution.py`: collected implementation, never evaluated. Syntax
  means only `ast.parse` succeeded, not that the answer is correct. Original
  bytes are hashed; if credentials are redacted, `stored_solution_sha256`
  records the different stored bytes.
- `summary.json`: task/completion/syntax/timeout/error counts, median,
  nearest-rank p90 and total task wall time, token totals, request count and
  observed request-time statistics. All failed/timed-out tasks remain in the
  wall-time denominator. Partial usage is retained when reported; missing
  usage is not estimated. In-flight requests lacking `message_end` have no
  completed-request usage/duration record. Nonzero exit indicates task failure.

Wall time runs from Pi process launch through exit/timeout cleanup, including
startup, model requests, tools, and orchestration; dataset download, directory
setup, and the harness's post-run syntax parsing are outside that interval.
`observed_seconds` measures assistant `message_start` to `message_end` receipt;
`duration_ms` is Pi's optional reported request duration. Turn duration includes
tools. Neither is server-only prefill/decode telemetry or GPU kernel timing.
Older Pi streams may omit reported durations and `agent_settled`; the parser
also accepts terminal `agent_end` with a final successful assistant response.
Windows runs each capture in a private, kill-on-close Job Object. A gated Python
launcher starts Pi only after job assignment succeeds; Pi and its descendants
are terminated even if Pi exits first. Breakaway is not enabled. Cleanup waits
for the job to empty and reaps the launcher, using bounded three-second waits;
cleanup errors are reported as `cleanup_failed:<exception type>`, not suppressed.
Windows termination is asynchronous: zero active job processes is not a barrier
for every retained process handle to signal. The Windows regression verifies
membership of the launcher, Pi stand-in, and descendant in the exact capture job
(and excludes an unrelated process), then requires their exit signals within a
shared one-second teardown deadline after capture, with total elapsed time under
five seconds. It covers both a sleeping Pi and Pi exiting before its descendant.
Launcher startup and all cleanup remain included in wall time. Windows hosts
whose existing job restrictions prevent nested job assignment fail before Pi
starts; there is no uncontained fallback. POSIX uses a process group; descendants
that deliberately escape it still require external disposable-environment
cleanup. Neither mechanism contains work delegated to an existing external
service. Unrelated processes/services are never stopped.

**This is not HumanEval pass@1, even for the full run. A subset is not a standard
leaderboard score.** No correctness claims are made without the official
evaluator in an appropriate isolated execution environment. No live-model
results are asserted by these fixtures. Before publication, the orchestrator
must review the five-task pilot, then run baseline/candidate 20-task cohorts
with matched IDs/order, Pi version, harness digest, token/context limits,
model revision/precision, server state/cache policy, hardware, power and load.
Keep failed evidence and publish only validated results.

## 独立 MLX 参考结果

已在本机 M5 Pro 20核 GPU / 48GB 上测量 Qwen3.6-35B-A3B 的 MLX community 5-bit 权重：短输入解码 **90.45 tokens/s**，4K输入 **86.91 tokens/s**，16K输入 **79.97 tokens/s**。单请求、每种输入预热一次再测三次，每次固定生成256 token。

这是独立 **MLX-LM** 引擎先前的 256-token 参考结果；最新 128-token 短输入复测已按 FP32 / BF16 分别列在上方。短输入结果不能推广为长上下文性能持平。下载器、固定依赖、原始记录位置和计时定义见 [外部参考基线](docs/MLX-BASELINE.md)。

## CLI

```sh
python3 scripts/download-qwen36-mlx.py --bits 8
dotnet run --project src/DeepInfer.Host -c Release -- demo --stay
# 另一个终端，默认 Qwen3.6-35B-A3B / Metal：
dotnet run --project src/DeepInfer.Host -c Release -- infer --prompt "用一句话介绍你自己。" --json
```

`deepinfer` 可替换为构建后的程序路径，或 `dotnet run --project src/DeepInfer.Host -c Release --`。独立运行各角色：

```sh
deepinfer coordinator
deepinfer worker --id qwen-mac --listen http://127.0.0.1:5101 --backend metal \
  --weights models/qwen3.6-35b-a3b-8bit-mlx --model qwen3.6-35b-a3b
deepinfer infer --prompt "写一句关于月亮的诗。"
```

Worker 仅加载 `mlx-qwen36-affine` 格式，分词器默认为权重目录中的 `tokenizer.json`。Metal 执行完整模型，CUDA 前段通过下文的常驻双机服务接入。CPU 数值算子保留用于验证，不再提供 CPU 模型后端。每个真实模型 Worker 容量固定为 1；无权重的 Worker 只探测设备。

单轮接口使用非思考聊天模板、贪心解码、独立请求状态；结构化多轮与流式接口见 [聊天 API](docs/CHAT-API.md)。输入输出共享 262,144 token 窗口，实际可用容量取决于内存；配置上限不等于通过全部长度验收。

## 构建与验证

需要固定的 .NET SDK 10.0.100、Python 3.9+。构建 Metal bridge 需要含 macOS 26+ SDK 的 Xcode/命令行工具；M5 BF16/Metal 4 路径要求相应 GPU 与操作系统。`global.json` 固定 SDK，依赖使用锁定还原。

```sh
dotnet restore DeepInfer.slnx --locked-mode
dotnet build DeepInfer.slnx -c Release --no-restore
dotnet test tests/DeepInfer.Tests -c Release --no-build
python3 scripts/integration.py
python3 scripts/chat-integration.py
# 已下载权重并构建 Metal bridge 后：
python3 scripts/tui-smoke.py --qwen36
deepinfer verify-qwen36 --weights models/qwen3.6-35b-a3b-8bit-mlx \
  --reference artifacts/reference/qwen36-8bit/reference.json
```

独立参考文件需预先生成，见 [8bit 验证报告](docs/QWEN36-8BIT.md)。旧模型的验证及性能记录保留为历史证据，相关加载器和复现脚本已移除。

Native AOT 发布：

```sh
dotnet publish src/DeepInfer.Host -c Release -r osx-arm64 -o artifacts/native/osx-arm64
dotnet publish src/DeepInfer.Tui -c Release -r osx-arm64 -p:PublishAot=true \
  -p:TreatWarningsAsErrors=false -o artifacts/tui-native/osx-arm64
```

发布时将 `libdeepinfer_metal.dylib` 与 `deepinfer` 放在一起，构建会自动复制。部署方式与两端验收见下文。

## 基准与限制

```sh
deepinfer bench --requests 3 --warmup 1 --max-tokens 128 --output artifacts/benchmark-qwen36.json
```

`timings.prefillMs` 测 prompt forward；`timings.decodeTokensPerSecond` 测首个到最后一个输出 token 之间的 N−1 个生成间隔。`tokensPerSecond` 包含 prefill 和 RPC。非流式接口的内部首 token 时间不能当作网络可见 TTFT。

服务不静默回退到其他模型或后端。跨机访问须显式启用 `--allow-remote` 并配置统一 token；使用可信私网/VPN 或 TLS 终止。完整的 128K 性能与资源门槛仍按 [固定规范](docs/QWEN36-8BIT-TARGETS.md) 验收。

Qwen3.6 检查点的 Apache-2.0 许可证与固定版本摘要保留在对应模型目录，权重不提交 Git。仅支持该模型的文本路径；图像、MTP、YaRN、GGUF 与通用 rank 通信尚未实现。纯模拟测试显式使用 `demo --backend simulation` 或 `infer --backend simulation --model synthetic-v1`。

M5 调优入口选择 hybrid GEMM、packed experts 与 BF16 attention。精度边界及历史测量见 [BF16 Metal](docs/BF16-METAL.md) 和 [调优记录](docs/PREFILL-TUNING.md)。

## Persistent 8bit deployment

The `split-serve` / `split-agent` commands keep Metal and CUDA weights loaded across requests, warm the complete Mac graph before readiness, reset request state, and use pinned mutual TLS for cross-host state transfer. The local coordinator exposes both complete and streamed responses; CLI and TUI share it. Current limits: affine8 weights, FP32 KV, 2048 total tokens and 256 output tokens, one active request plus a bounded queue. BF16/128K performance acceptance remains open.

See [deployment instructions](docs/DEPLOYMENT.md) for private configuration, Native AOT packaging, Mac LaunchAgents, Windows Service installation, upgrade/rollback and real-model acceptance. Windows Service installation requires administrator PowerShell; the foreground SSH runner is only for development.

## Chat playground 与 HTTP 兼容接口

TUI 默认连接当前 5200 coordinator，支持 Microsoft.Extensions.AI 流式多轮对话、无边框紧凑布局。HTTP 提供 Chat Completions、Claude Messages 和 Responses 的文本流式/非流式格式。用法、鉴权与明确的能力边界见 [Chat API / TUI](docs/CHAT-API.md)。
