# Infer.NET 双机部署

本入口用于 Qwen3.6-35B-A3B **8bit 权重、FP32 KV**，候选版本移除 2048 context / 256 output 人为上限，模型窗口为输入加输出 262144 tokens，实际请求仍受内存和 deadline 约束。长窗口整模可用性与吞吐需要独立实测，不由上限声明保证。它提供常驻执行与服务生命周期；BF16、128K 和性能目标仍需另行验收。代码位于部署实现分支，未自动合并上游 PR。

## 进程与网络

- Mac coordinator：loopback HTTP 5200，CLI/TUI 的统一入口，不加载模型。
- Mac `split-serve`：loopback HTTP 5201 + LAN TCP 5210。持有唯一完整模型；负责后 20 层 prefill、接收前 20 层状态、全部 decode。
- Windows `split-agent`：主动连接 Mac 5210，前 20 层权重常驻。无需开放新的 Windows 入站端口。
- TCP 5210 强制双向 TLS + 固定证书 SHA256 校验，内部继续使用 SplitWire v3 身份/布局/位置/序号/HMAC 检查。客户端逐 token 输出走本地 HTTP SSE。v3 不兼容旧 v2，CUDA 原生 ABI 也升至 3；升级时两端必须一起切换，旧包保留供成对回退。
- API 默认只对 Mac 本机开放；其他设备不能直接把它当作安全公网 API。跨机数据通道已经加密；HTTP 的远程 TLS 网关不在本版本安装器范围。

每台机器只允许一个部署 worker 持有相应配置的独占锁。不要同时运行旧 demo、单机模型或另一份完整模型来做正式性能验收。历史 demo 的 Qwen3.6 默认权重已改为 8bit。

## 构建、配置、打包

Mac：

```sh
dotnet publish src/DeepInfer.Host -c Release -r osx-arm64 -o artifacts/deploy/build-mac
python3 scripts/deploy/package.py --publish-dir artifacts/deploy/build-mac --rid osx-arm64 --output artifacts/deploy/releases/mac-VERSION
python3 scripts/deploy/verify-package.py artifacts/deploy/releases/mac-VERSION
```

Windows 普通 PowerShell：

```powershell
& scripts\windows\build-deployment.ps1
python scripts\deploy\package.py --publish-dir artifacts\deploy\build-windows --rid win-x64 --output artifacts\deploy\releases\windows-VERSION
python scripts\deploy\verify-package.py artifacts\deploy\releases\windows-VERSION
```

打包要求源码干净，且 `deepinfer version` 的源码 SHA 与当前 checkout 一致。清单包括文件散列、平台、模型 revision、KV 精度与能力上限。程序运行不需要 SDK、Python 或 nvcc；配置生成/安装辅助脚本分别使用 Python/OpenSSL 或 PowerShell。GPU 驱动、系统框架与 MSVC 系统运行依赖需在目标机预检。

使用 `scripts/deploy/prepare.py --help` 生成独立的私有配置目录，明确传入 Mac LAN 地址、两端模型位置、Windows 配置目标目录。该工具生成两套固定证书、随机集群 token 与配置；证书有效期 365 天，换证时排空请求并成对更新 pin。新建目录使用 0700，文件 0600；Windows 配置目录应仅允许专用账号、SYSTEM、Administrators 访问。

只将生成的 `windows/` 内容安全传输到指定 Windows 配置目录；不要传 Mac 私钥，不把配置/凭据放入 GitHub。源码仍通过 GitHub 同步，模型单独校验；Windows prefix bundle 与 Mac 完整模型实际使用的文件均在启动时校验固定散列。

## 直接启动和使用

先启动三个独立进程（路径使用生成的绝对配置路径）：

```sh
deepinfer coordinator --config /PATH/mac/coordinator.json
deepinfer split-serve --config /PATH/mac/worker.json
```

Windows：

```powershell
.\deepinfer.exe split-agent --config C:\PATH\worker.json
```

普通前台模式需要保留启动终端；SSH 子进程可能随会话退出。`scripts/windows/run-deployment.ps1` 是保留 SSH 会话的开发验证工具，绝不代替 Windows Service。该普通账号不具备 WMI 脱离会话创建进程的权限，开发流程不修改此权限。

用户请求：

```sh
deepinfer infer --endpoint http://127.0.0.1:5200 \
  --token-file /PATH/mac/token.txt --stream \
  --model qwen3.6-35b-a3b --prompt "你好" --max-tokens 128
```

`--stream` 实时打印 token，最后显示 finish reason 和 decode 吞吐；不带该参数返回完整响应。自然生成遇到 EOS 结束，测试 API 可显式设置 `benchmarkIgnoreEos=true`。TUI 默认连接同一 5200 coordinator，通过 Microsoft.Extensions.AI 的 IChatClient 流式显示多轮对话；在仓库根目录无参数启动时读取现有部署配置，也可用 `DEEPINFER_TOKEN_FILE` 或 `DEEPINFER_TOKEN` 指定凭据。TUI 自身不加载模型。兼容 Chat Completions、Messages、Responses 的文本接口见 [HTTP chat API 指南](CHAT-API.md)。

`GET /health` 可见 Loading/Warming/Ready/Busy/Faulted、PID、build、模型与容量。`GET /ready` 需要 `X-DeepInfer-Token`，模型完成实际预热且 Windows 已认证连接才返回 200；加载完成并不等于 Ready。首次预热执行全部 40 层及 32 个 decode 步骤，之后重置请求状态，无 prefix cache 命中。

## 系统托管

Mac 使用用户 LaunchAgent：

```sh
python3 /RELEASE/deploy/install-macos.py --executable /RELEASE/deepinfer --config-dir /PATH/mac
```

登录后启动、异常退出重启；退出登录/休眠不是无人值守可用性保证。停用使用相同参数加 `--uninstall`，只操作 `com.infernet.coordinator` 与 `com.infernet.metal`。日志在配置目录 `logs/`；应按保留周期归档，当前安装脚本不做自动日志轮转。

Windows 使用专用标准本地账号。先停止开发前台 agent，再在**管理员 PowerShell**执行：

```powershell
& C:\RELEASE\deploy\install-windows.ps1 -Executable C:\RELEASE\deepinfer.exe -Config C:\PATH\worker.json
```

安装脚本在本机交互请求服务账号凭据，不接收聊天中的密码。账号需要“作为服务登录”权限及模型/配置/程序目录的相应访问权；组织策略禁止时须由管理员处理，脚本不修改驱动或防火墙。服务名 `InferNetCuda`，自动启动，失败按 10/30/60 秒重试。删除使用同命令加 `-Uninstall`。Windows Service 通过 .NET 的 SCM 生命周期取消任务；系统安装、重启后的 GPU 可见性及自动恢复必须在实际服务账号下验收。

## 运行与升级边界

单 GPU executor，默认最多 2 个排队请求；超出返回 429。默认 300 秒 deadline 包括排队，可配置 1..1800 秒；客户端断开会取消当前请求。取消是协作式的，已经提交的单个 GPU 工作不能瞬时中断。每个请求重置 KV/Delta/卷积状态，传输缺失、重复或位置错误时阻止 decode。GPU 错误退出进程交给系统重启，已输出部分 token 的请求不会自动重放。

升级：停止接入、等待请求完成、停旧 worker、校验新包、切换绝对路径、重新启动并等待 ready；失败恢复旧包和旧兼容配置。两端 split 协议、模型 revision、KV 精度及 ABI 必须匹配。Mac 不并行加载新旧完整模型，本版本接受维护窗口，不承诺无停机滚动升级。

API 返回 prefill/decode/分 token profile、Metal allocation、GPU command 时间与提交计数；`--log-file` 可将输出写入指定持久日志。系统 swap 与供电仍由实验监控采样，不把既有 swap 总量误当作本次新增换页。

自动验收入口：

```sh
python3 scripts/deploy/acceptance.py --token-file /PATH/mac/token.txt --output artifacts/deploy/acceptance-UNIQUE
```

它验证真实连续请求、输出确定性、流式文本与 token 一致、取消后恢复、自然 EOS、容量拒绝、队列背压和 PID 稳定。需另外执行 agent 断连/重启测试、服务安装/重启测试，以及 `docs/QWEN36-8BIT-TARGETS.md` 的完整性能门槛。

## Metal 权重驻留候选

`split-serve` 默认启用 `metal-residency=on`，为已加载权重建立 Metal residency set，预算不超过建议工作集与物理内存 80% 的较小值减 512 MiB。权重超出预算会明确失败，不静默退回；尚未固定驻留 KV/工作区。权重上传完成后、预热与 Ready 之前提交驻留请求。默认 `residency-refresh-ms=1000`，空闲时续驻留，不提交空 GPU 推理；资源最后一个图/矩阵引用释放时撤销登记。

`metal-residency=off` 为同一候选版本的对照开关，`residency-refresh-ms=0` 可测仅申请一次的行为。这些配置只作用于应用，不修改系统 wired limit。`RESIDENCY` 日志记录预算和登记大小，不保证操作系统一定满足驻留请求；仍须测空闲后的真实请求与换页。

新增 `PREFILL` 日志分别记录 reset、hidden 等待、Mac suffix、状态等待/导入及交接字节数；hidden 等待含远端计算和传输，不称为纯 CUDA 时间。`METAL_WAIT` 对超过 100 ms 的命令记录 commit、wait 和 GPU 时间。日志用于定位，不改变 API 已有计时定义。

驻留参数通过显式 C ABI 从 C# 传给 Metal；不能依赖 `Environment.SetEnvironmentVariable`：Unix 的托管环境副本不会更新原生 `getenv`。0bcbbb6 的首次上线因此没有启用驻留，保留为失败实验，不能算驻留性能结果。原生 `RESIDENCY` 日志及跨边界测试才证明该选项生效。历史 `Profile` 的其他环境参数同样需要逐项审核；不要将托管赋值当成原生生效证据。

### Small-request routing and CUDA readback

`short-prefill-tokens` defaults to `128` (range `0..128`). The count includes the complete chat template and history. Small requests run full prefill and decode in the existing resident Mac model; larger requests retain the CUDA-prefix / Metal-suffix path. Set it to `0` to force split execution for comparisons. This is an explicit routing policy, not prefix caching; each request still resets its state. RPC results and latest worker metrics expose `executionPath` (`metal-local` or `cuda-metal-split`), also shown in the TUI.

`cuda-readback` defaults to `token`. The `batch` candidate is opt-in: measured 512-token native-prefix A/B was 3.55% slower with FP32 KV and 3.95% slower with BF16 KV, despite exact hidden/state agreement. Batch mode writes each final-layer hidden vector into a 256 KiB device buffer and performs one host readback per chunk (up to 32 tokens). It does not change arithmetic, precision or wire layout, and it is not batched GEMM. Native CUDA ABI is now 5; rebuild both managed/native components. Use `scripts/split/benchmark-readback.py` for exact hidden/state A/B and a synthetic-token prefix-only timing check, with the deployed agent stopped first so there is only one CUDA model.


### Layer-major CUDA prefill

`cuda-prefill=batch-fp32` (default) processes all tokens of each 32-token chunk layer by layer. Projections/experts keep the original FP32 GEMV arithmetic; convolution, attention and routing are batched, and Delta recurrence keeps each state row in registers across the chunk. This is distinct from the experimental Tensor Core BF16 GEMM primitive. `cuda-prefill=serial` retains the original token-major path for A/B. The agent logs its actual policy on readiness. Source fingerprints now cover all CUDA include files as well as the main translation unit.

A standalone pinned-model test compared 131 synthetic token IDs across irregular chunk sizes, all 20 layer traces and all 40 prefix state tensors: byte-exact and finite for FP32 and BF16 KV. The 512-token prefix-only median improved 3.44x (FP32 KV) / 3.42x (BF16 KV). These tests do not pass cross-host BF16 accuracy or 128K end-to-end performance gates. Use `scripts/split/benchmark-prefix-chunk.py` with the deployed agent stopped to avoid loading another model on that GPU.

When the CUDA agent is disconnected, an enabled short-request route remains available as `ReadyLocal`; longer requests return a bounded peer-unavailable error. `short-prefill-tokens=0` retains the original all-split readiness behavior. No long request silently changes route.

`suffix-chunk-tokens=128` is now the selected default (32 and 512 remain available for A/B). CUDA still sends 32-token hidden chunks; Metal combines them before executing the suffix. Native scratch is explicitly configured to 512 tokens rather than relying on managed environment changes reaching libc. Rebuild/package the updated Metal bridge with the managed executable. The selected 128-token policy passed 234/714-token layer/logit checks and repeated 8K end-to-end comparisons; 512 did not improve the complete pipeline. See `SUFFIX-BATCHING.md`. These results do not pass BF16/128K targets.

`cuda-send-overlap=on` is selected after same-package A/B (off remains available). A bounded two-frame queue overlaps one CUDA chunk producer with one authenticated sender. All exits join the producer before reset/disposal; no background retry. Final KV/recurrent state transfer still happens after hidden production. See `PREFIX-SEND-OVERLAP.md` for the 4.27% 8K prefill gain and real mid-prefill cancellation/recovery check.

`kv-transfer=final|incremental` selects final full KV handoff or experimental per-chunk lossless KV transfer. Default remains `final` pending end-to-end A/B. The incremental mode sends ten attention KV slices after each 32-token hidden chunk; Metal coalesces and imports them after each suffix submission. The thirty DeltaNet/convolution states still transfer last. Decode remains blocked until every tensor is complete. The two-item send queue now holds complete bounded chunk bundles, and every failure joins the GPU producer before reset. This change requires **both hosts rebuilt with split protocol v4 and CUDA prefix ABI 6**; v3 peers fail authentication. Deploy the pair together, retain the previous immutable pair for rollback. No precision or model change.
