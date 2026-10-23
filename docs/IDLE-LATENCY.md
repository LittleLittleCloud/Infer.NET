# 空闲后首请求停顿：驻留与换页诊断

2026-10-22（America/Los_Angeles）。关联 Issue #9 / #13。结论：这些十几至九十秒的停顿不是已经加载、预热的推理服务应有的稳态延迟。当前证据强烈指向 Metal 资源重新驻留和内存压力；几千 token 的慢 prefill 另有部署计算路径未批量化的问题。尚未实施修复或完成驻留策略 A/B。

## 同进程复现

现有双机后端源码 `506481445d3670d129ff3d101251f96357efb971`，Qwen3.6-35B-A3B affine8 / group64，revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`，FP32 KV。Mac M5 Pro 48 GiB / macOS 26.6.2，RTX 4090，既有家庭 LAN。Mac 插电，未重新核对 High Power；RTX 4090 只读快照显示 19427 MiB 使用、0% GPU。没有修改系统设置、关闭其他应用、替换后端或加载第二个模型。

服务 Ready 后，经 coordinator 的 `/v1/infer/stream` 连续请求两次。每次新 HTTP 连接，固定 system `You are a helpful assistant.`、user `你好`，maxTokens=32，profile=true，自然 EOS。模型 PID 不变，两次都是 24 输入 / 13 输出，全部 token ID 一致。没有使用 TUI，说明无需重开 TUI 也会出现停顿。

|指标|空闲后第一次|紧接着第二次|
|---|---:|---:|
|客户端首段可见文本|3.709 s|1.327 s|
|内部 prefill（含交接）|3.348 s|1.314 s|
|首个 decode 调用，即输出 token 2|16.393 s|0.410 s|
|客户端总耗时|20.270 s|1.903 s|
|Mac 累计 GPU 执行计时|225.76 ms|225.50 ms|
|整机 swap-in 页计数换算|4.201 GiB|4.828 MiB|
|整机 swap-out 页计数换算|0.699 GiB|0|

输出 token 1 是完整的 `你好`，不是空 UTF-8 片段；这次客户端在 3.709 秒已经收到首字，接着等待约 16.4 秒才收到第二个 token。不能把整段等待叫做 prefill。第一次的 token 3 起恢复约 13.6–14.2 ms/token。

这只是两次诊断，不能据此给出 p95、正式吞吐或驻留优化的收益承诺。第一次还做了 1 秒 sample 采样，存在观察开销。vm_stat 是整机计数，不能把所有换页归给该进程；页面数×16384 是计数折算量，不是 SSD 实际传输字节数。解压页计数折算为约 30.6 GiB，也不代表额外读取了 30.6 GiB 磁盘数据。

## 停顿在哪里

第一次请求约第 8 秒的栈采样显示：

- 推理线程：`SplitService.Handle → Qwen36Gpu.Forward → di_graph_forward → q_complete → waitUntilCompleted`。
- Metal 提交线程：`_submitAvailableCommandBuffers → IOGPUCommandQueueSubmitCommandBuffers → iokit_user_client_trap`。
- 同期 swap-in 页持续增长；到请求完成约 4.20 GiB。
- 该首步 decode 的墙钟为 16392.604 ms，GPUStartTime 到 GPUEndTime 为 17.769 ms。

因此主要等待位于 Metal/驱动提交与完成路径，且和大量换页同现；不是这一步 GPU kernel 真算了 16 秒。它也不是等待 Windows KV：源码先完成状态导入，再进入此 Forward。进程未重启、容量保持 6144，采样栈不在模型加载、tokenizer、网络接收或 KV Grow。

Mac 模型 Metal allocation 为约 35.22 GiB；设备 recommendedMaxWorkingSetSize 为 37.44 GiB，物理内存 48 GiB。测试前 vmmap 的 IOAccelerator graphics 显示约 30 GiB 在 SWAPPED 列，而系统 swap 总使用约 11.6 GiB：两者不是同一口径，不能声称模型有 30 GiB SSD 数据。当前内存压力、压缩和换入证据，足以把驻留问题列为首要嫌疑，但尚未用驻留开关 A/B 排除其他驱动等待。

分流设计会放大这个转换：prefill 时 Mac 只执行后 20 层；首步 decode 才在 Mac 执行全部 40 层，扩大需要访问的权重范围。当前 Metal buffer 使用 shared storage，却没有显式 residency set 管理。shared 表示 CPU/GPU 共享内存，并不保证空闲期间持续驻留。

## 与主流框架及 Apple 文档对照

- [Apple residency sets 文档](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets)：默认在提交时使资源驻留，可能推迟 GPU 开始执行；可用 requestResidency 提前准备，但其他应用争用内存仍可能延迟处理。
- [llama.cpp PR #17766](https://github.com/ggml-org/llama.cpp/pull/17766)：2025-12-05 合入 residency keep-alive，处理空闲后重新 wiring 的延迟。这是已有工程问题，并非正常推理算力成本。
- [MLX Issue #4609](https://github.com/ml-explore/mlx/issues/4609)：报告空闲后资源失去驻留、下一次操作显著变慢。该报告是其他设备上的复现，不能直接把其时间或倍率套到本机。
- [MLX residency 实现](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/resident.cpp) 已提供可配置的 residency refresh；[set_wired_limit 文档](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html) 说明该预算用于保持内存驻留。只研究机制，没有把 MLX 接入产品后端。

## 为什么 prefill 本身也慢

部署 `SplitService.Profile` 固定 chunk=32、FP32、matrix=0、Delta sequential。CUDA `di_cuda_prefix_run` 内层明确按 token 循环，使用 GEMV，并对每个 token 的 hidden 做同步 D2H。32 是传输/调用块长，不意味着已经执行 32-token 批量 GEMM。它没有继承单机大分块优化的全部路径，因此当前约 150–164 tokens/s 的用户观测不能当成 4090 的能力上限。

此外，每次请求都交接前 20 层状态。即使只有 24 个输入 token，也有 15 层 Delta/卷积固定状态，共 32,931,840 bytes，加 491,520 bytes KV，约 31.875 MiB；另有约 0.188 MiB hidden。暖请求约 1.3 秒的 prefill 包含这些计算、CPU staging、Wi-Fi/TLS 传输和导入，有真实固定开销。没有分阶段测量前，不能把这 1.3 秒全部归到网络，也不能把几千 token 的 12–26 秒视为模型固有延迟。

## 修复优先级与验收

1. 对 Metal 权重与工作区实行有预算的 residency set 管理，验证空闲后是否需要定期续驻留；预热与 readiness 要覆盖完整 decode，保留 OS/其他应用内存余量。先做局部 A/B，不盲目增大全局内存限制。
2. 补齐 reset、前段计算、等待 hidden、后段计算、状态接收/导入、Metal commit/wait 和 first-text 分段指标。
3. CUDA prefill 改成批量 GEMM；重新验证 Mac 后段批量路径与大分块。短请求可另评估显式单机路由，避免每次支付固定状态交接。

验收建议：同一模型、输入、精度、供电和后台环境下，分别空闲 0/2/10/60 秒再请求，比较 TTFT、token 1→2 间隔、后续 decode、swap 增量与 token IDs。驻留前后多轮比较，再判断收益；不能靠持续发空推理维持繁忙来掩盖问题。

[公开诊断摘要](../site/idle-latency-summary.json)。原始 JSON、逐秒 vm_stat、采样栈、vmmap、decode 日志在本地忽略目录 `artifacts/dual-host/idle-latency-20261023T061238Z`，未公开进程映射或本机路径明细。没有达到正式性能门槛，也没有宣称问题已修复。
