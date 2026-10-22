# Infer.NET — Qwen dense / Qwen3.6 MoE

[项目进展](https://littlelittlecloud.github.io/Infer.NET/) · [实验笔记](https://littlelittlecloud.github.io/Infer.NET/experiments.html) · [任务与验收](https://github.com/LittleLittleCloud/Infer.NET/issues/1)

站点内容由 `site/` 与固定性能规范生成，运行 `python3 scripts/build-progress-site.py` 可在 `artifacts/pages/` 预览。推送相关变更到 main 后由 GitHub Actions 发布。项目名称为 Infer.NET；现有 `DeepInfer.*` 文件名与命令保留兼容。

**默认模型已换成 Qwen2.5-3B-Instruct。** CLI / RazorConsole → 协调器 → 独立 .NET Worker → 实际 Metal 计算 → 中文回答。支持 Native AOT。

3B / 7B 采用官方 BF16 safetensors，权重保持 BF16 存储，Metal 矩阵运算使用 FP32 累加。分词、聊天模板、注意力、归一化、RoPE、激活和 KV cache 在 C# / CPU 上执行。Qwen3.6 MoE 使用下文的自写量化执行路径。产品不调用 Python、Transformers、MLX、llama.cpp 或外部模型服务。

## 直接启动

在这台 M5 Pro 48GB 上双击 **`DeepInfer.command`**，或：

```sh
./scripts/mvp.sh
```

首次下载约 6.2GB 官方模型；已有文件会验证 SHA-256 并复用。启动一个协调器、一个 Qwen 3B Worker 和终端界面。默认输入“用一句话介绍你自己。”，Tab 选择 Generate，Enter 生成，Ctrl+C 退出并清理进程。

原生版本已放在 `artifacts/`；没有原生文件时启动器会构建托管版。默认端口 5100–5101；可用 `DEEPINFER_BASE_PORT=5500 ./scripts/mvp.sh` 覆盖。默认使用一个模型副本；显式增加 `--workers` 才加载更多副本。

## 试用 7B 与测量解码

双击 `DeepInfer7B.command`，或运行 `DEEPINFER_MODEL_SIZE=7b ./scripts/mvp.sh`。它加载官方 Qwen2.5-7B-Instruct 原始 BF16 权重（约 15.2GB），默认一个副本；原有 `DeepInfer.command` 继续使用 3B。

```sh
python3 scripts/download-qwen.py --size 7b
deepinfer demo --model qwen2.5-7b-instruct --stay
# 另一个终端：
deepinfer infer --model qwen2.5-7b-instruct --prompt "用一句话介绍你自己。" --json
deepinfer bench --model qwen2.5-7b-instruct --requests 3 --warmup 1 --max-tokens 128
```

`timings.prefillMs` 是 worker 的 prompt forward 时间；`timings.decodeTokensPerSecond` 只测首个输出 token 到最后一个输出 token 之间的 N−1 个生成间隔，排除加载、prefill、首 token 和末尾 EOS 探测。它包含 CPU 运算、GPU 调度和同步，属于实际引擎解码速度。非流式 API 的 `firstTokenMs` 是 worker 内部时间，不能当作用户看到首 token 的网络延迟。`bench` 同时保留包含 prefill/RPC 的 `tokensPerSecond`，不能将两者混用。

两种模型的相同条件实测及复现命令见 [解码速度记录](docs/DECODE-BENCHMARK.md)。

## Qwen3.6-35B-A3B：自写 Metal / C# 引擎

**8bit 性能目标 v1 已固定**：128K 无前缀缓存，双机端到端 prefill 最终目标 **2000 token/s**，128K + 4096 输出整次请求 **≤ 180 秒**，持续 decode **≥ 37 token/s**。阶段验收线为 1600 token/s / 200 秒；均为待实现目标。模型、计时、复测、正确性与内存门槛见 [性能验收规范](docs/QWEN36-8BIT-TARGETS.md)。

**8bit 新基线（独立产物）**：自写 Metal 已完成 8bit 权重支持与实际模型校验。M5 Pro 48GB、AC / High Power，8K / 32K / 128K 输入的稳态 prefill 分别为 **3.467 / 17.743 / 135.966 秒**，解码 **66.34 / 60.51 / 37.52 token/s**。**128K + 4096 输出**使用 512 分块完整跑通：prefill **164.157 秒**、持续解码 **38.95 token/s**。拆开最终词表投影降低单次提交的驻留峰值；长输出还需缩小工作区。整机确有换页，不能视为无内存压力。逐层校验、68 项单测及 4 项长请求后 RPC 验收通过；8bit 的复现、失败记录、内存限制与 20/20 双机容量/收益估算见 [8bit 验证报告](docs/QWEN36-8BIT.md)。现有 MoE 启动器仍保留原 5bit 配置，以下为此前 5bit 测量。

**当前 M5 启动配置**：专家尾块自适应 16/32/64、gate/up 合并矩阵乘法、DeltaNet 直接矩阵读取。AC / High Power、相同二进制与精度、每档预热三次测三次，8K prefill **3.468 → 3.353 秒（减少 3.32%）**，32K **17.850 → 17.344 秒（减少 2.83%）**。两档 A/B 全部 128 个输出 token 一致，显存分配和提交次数不变。新鲜 MLX 对照 **2.967 / 14.701 秒**，prefill 仍慢 **12.99% / 17.98%**，尚未持平；本轮解码为 **84.76 / 74.88 token/s**。287 项 Metal、68 项单元、25 项 RPC 和 4 项真实模型检查通过。`DeepInferMoE.command` 已选择新配置，详见 [实现与完整证据](docs/PREFILL-EXPERT-DELTA.md)。下文保留历史轮次，各自条件不能混用。

此前专家中间布局实验：已实现 gate/up 直接输出 down 所需 BF16 布局，8K 同轮 prefill 耗时降低约 **1.4%**；32K 初测及补测分别慢约 **1.1% / 0.5%**，因此保留原默认路径。反量化双缓冲在 8K 反而慢约 **7.1%**，未启用。精度及输出不变，详见 [实现、High Power 实测与验证](docs/EXPERT-FLOW.md)。

双击 **`DeepInferMoE.command`**，或运行 `DEEPINFER_MODEL_SIZE=moe ./scripts/mvp.sh`。本机权重已下载。其他机器先用 `scripts/download-qwen36-mlx.py` 获取固定版本的量化权重；该下载器的 Python 环境需要 `huggingface-hub`，运行产品无需 Python。

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

## 独立 MLX 参考结果

已在本机 M5 Pro 20核 GPU / 48GB 上测量 Qwen3.6-35B-A3B 的 MLX community 5-bit 权重：短输入解码 **90.45 tokens/s**，4K输入 **86.91 tokens/s**，16K输入 **79.97 tokens/s**。单请求、每种输入预热一次再测三次，每次固定生成256 token。

这是独立 **MLX-LM** 引擎先前的 256-token 参考结果；最新 128-token 短输入复测已按 FP32 / BF16 分别列在上方。短输入结果不能推广为长上下文性能持平。下载器、固定依赖、原始记录位置和计时定义见 [外部参考基线](docs/MLX-BASELINE.md)。

## CLI

```sh
python3 scripts/download-qwen.py
# 启动后保持运行：
./artifacts/native/osx-arm64/deepinfer demo --stay
# 另一个终端，默认就是 Qwen 3B / Metal：
./artifacts/native/osx-arm64/deepinfer infer --prompt "用一句话介绍你自己。" --json
./artifacts/native/osx-arm64/deepinfer infer --prompt "计算2加2，只回答数字。"
./artifacts/native/osx-arm64/deepinfer status
```

实际生成的回答：

> 我是Qwen，由阿里云开发的超大规模语言模型，旨在提供各种领域的信息和帮助。

这是单轮聊天：框架添加官方默认 system / user / assistant 模板，使用贪心解码，遇到 EOS、长度或上下文限制停止。每次请求拥有独立 KV cache。目前应用上下文上限 **2048 tokens**，生成上限 **256 tokens**；并未开放模型配置中的完整长上下文。

## 独立运行各个角色

```sh
deepinfer coordinator
deepinfer worker --id qwen-mac --listen http://127.0.0.1:5101 --backend metal \
  --weights models/qwen2.5-3b-instruct --model qwen2.5-3b-instruct
deepinfer infer --prompt "写一句关于月亮的诗。"
```

`deepinfer` 可替换为上述原生程序路径，或 `dotnet run --project src/DeepInfer.Host -c Release --`。`--weights` 指向目录时加载 `hf-qwen2`，分词器默认为目录中的 `tokenizer.json`。CPU 后端也支持该格式，但本机 3B 验收重点是 Metal；Python 只用于独立验收，不是 Worker 后端。

请求通过 HTTP JSON RPC；协调器按已加载的 model/backend 路由，Worker 是独立进程。真实模型 Worker 容量固定为 1，更多 Worker 提供完整模型副本。当前分发整个请求，尚未拆分模型到多台设备。

## 构建与验证

需要 .NET 10.0.100（接受后续 10.0 补丁）、Python 3.9+；Metal 需要 macOS 15+、Apple GPU 和构建时的命令行工具。Metal shader 在运行时编译，无需单独安装离线 Metal 编译器。

```sh
dotnet restore DeepInfer.slnx --locked-mode
dotnet build DeepInfer.slnx -c Release --no-restore
dotnet test tests/DeepInfer.Tests -c Release --no-build
python3 scripts/download-qwen.py
python3 scripts/qwen-inference.py
python3 scripts/tui-smoke.py --qwen
# 原有小模型和故障场景回归：
python3 scripts/integration.py
python3 scripts/real-inference.py
```

独立 Transformers FP32 参考仅用于测试，可选：

```sh
# 在独立虚拟环境安装 torch / transformers / tokenizers / safetensors 后：
python scripts/reference-qwen.py
# 对照 5 组分词/聊天模板、151936 维 logits、完整生成 token IDs：
deepinfer verify-qwen --reference artifacts/qwen-reference.json
```

本机使用的参考版本及测试结果见 [验证记录](docs/VALIDATION.md)。没有参考文件时，验收仍验证真实 BF16 Metal 生成，但不会声称做过独立 logits/token 对照。

Native AOT 发布：

```sh
dotnet publish src/DeepInfer.Host -c Release -r osx-arm64 -o artifacts/native/osx-arm64
python3 scripts/qwen-inference.py --executable artifacts/native/osx-arm64/deepinfer \
  --output artifacts/qwen-e2e-native.json
dotnet publish src/DeepInfer.Tui -c Release -r osx-arm64 -p:PublishAot=true \
  -p:TreatWarningsAsErrors=false -o artifacts/tui-native/osx-arm64
python3 scripts/tui-smoke.py --qwen --host artifacts/native/osx-arm64/deepinfer \
  --tui artifacts/tui-native/osx-arm64/deepinfer-tui
```

发布时把 `libdeepinfer_metal.dylib` 与 `deepinfer` 放在一起，构建会自动复制。普通 Release 构建无警告；原生 TUI 保留一条上游 IL2104 裁剪警告，使用到的组件已实际验证。Linux、Windows、NVIDIA GPU 和真实多机网络尚未在本机验证。

## 基准与限制

```sh
deepinfer bench --requests 3 --warmup 1 --max-tokens 48 --output artifacts/benchmark-qwen.json
```

结果统计真实生成 token、RPC p50/p95、已完成 GPU 命令数量与 GPU 时间。请求耗时包含聊天模板、prefill、decode、CPU 运算、GPU 同步和 RPC，不能视为纯 decode 吞吐。Qwen3.6 已实现 GPU attention 和批量 prefill；流式返回尚未实现。

RPC 默认等待上限为 5 分钟，以容纳 3B 模型。心跳间隔 2 秒、租约 8 秒；过载返回 429，无可用模型返回 503，失联 Worker 返回 502。不会把 CUDA 请求回退到其他后端。服务器默认仅监听本机；跨主机必须显式使用 `--allow-remote` 与统一的 `DEEPINFER_TOKEN`。HTTP 未加密，应使用可信私网/VPN 或 TLS 终止。

Qwen2.5-3B 使用 **Qwen Research License**，不是 Apache/MIT；下载器保留 [上游许可证](models/qwen2.5-3b-instruct/LICENSE) 和 [固定版本/摘要](models/qwen2.5-3b-instruct/provenance.json)。实际权重不提交 Git。Qwen3.6 检查点保留其 Apache-2.0 许可证。CUDA 推理、其他 Qwen3 变体、GGUF、持久多轮会话和跨设备模型分片尚未实现。

TinyStories 15M 仍保留为小模型回归：`demo --backend metal --model stories15m --weights models/stories15M.bin --tokenizer models/tokenizer.bin`。纯模拟测试须显式使用 `demo --backend simulation` 或 `infer --backend simulation --model synthetic-v1`。

Metal BF16 matrix operands and 16-bit KV are available as an opt-in M5 path; precision boundaries, measurements and commands are documented in [BF16 Metal](docs/BF16-METAL.md).

The M5 MoE launcher selects the measured hybrid GEMM, packed expert input and fused BF16 attention profile. See [current tuning and matched MLX measurements](docs/PREFILL-TUNING.md).
