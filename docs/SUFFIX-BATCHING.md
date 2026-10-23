# Mac 后段合批：128 优于 32，512 未进一步改善整次请求

2026-10-23，继 CUDA 按层批量执行后，保持前段 32-token wire chunk，Mac 把多个 hidden 块合并后执行后 20 层。`suffix-chunk-tokens=32|128|512`，本轮选择 **128**。不改变模型 affine8/group64，也不改变 FP32 GEMM/KV 的部署诊断精度。

Native scratch 通过明确的 C ABI 设置为 512 tokens，避免依赖 Unix 上 .NET 环境变量与 libc 环境不一致的行为。有效请求的 Metal allocation 从前一版 37,995,495,424 降为 37,572,984,832 字节，少约 403 MiB。正式 A/B 的三档全部使用相同 512 scratch，不把 scratch 改动算成合批收益。

## 数值与资源

诊断先在完整 Mac 模型按 32-token 块生成参考，再取第 20 层 hidden 输入到后段，比较后 20 层全部 trace、8 步完整 logits 和 greedy 输出，并导入相同前段状态验证后续解码。234、714 两种实际模板后长度覆盖小尾块、128 跨块、512 跨块；32 对照完全一致，128/512 最大 relative L2 为 `1.1783111e-5`、max absolute `3.4570694e-5`，均通过固定 `1e-4 / .01` 门槛，8 个 greedy token 一致。没有使用文本相似代替数值验证。

完整 8K API 的 16 次有效请求全部 128 个输出 token 与之前 CUDA 串行基线相同。每次新增整机 swap-out 为 0，Metal allocation 约 34.99 GiB，Windows 整 GPU 500ms 采样峰值 19,585 MiB 以下。Mac `vmmap -summary` 在每档结束时采集，physical footprint / peak 记录在 JSON；初始两档 peak 约 35.6G。采样不等于所有分配事件的精确峰值。

## 同包双机 A/B

两端 Native AOT source `a8ce0250b959b9279be8ccce99e9922c07f47f22`，同一 CUDA agent，Mac 每档按新配置重启；模型预热一次、正式三次，每次 reset，greedy 固定 128 输出，0% prefix cache。输入仍为固定 8192 tokens，SHA256 `defa12f5be86a88aed8a25aa479ab1b17860a661ff364865708e121cac17206d`。Mac AC / High Power，既有 Wi-Fi ↔ Ethernet LAN / TLS，未用 SSH 隧道。

|Mac 后段块|正式 prefill 秒（三次）|中位 prefill 秒|客户端首 token 中位秒|客户端整次中位秒|decode 中位 token/s|
|---|---|---:|---:|---:|---:|
|32|18.125 / 18.402 / 18.408|18.402|18.414|20.682|55.94|
|128|17.314 / 17.134 / 17.168|17.168|17.181|19.457|55.78|
|512|17.536 / 17.978 / 17.621|17.621|17.632|19.904|55.86|
|128 确认|17.355 / 17.142 / 17.403|17.355|17.366|19.628|55.82|

128 初轮减少 6.71% prefill 时间，确认轮仍减少约 5.69%。完整演进是早先串行 52.503s → CUDA 批量 18.642s → 当前约 17.2–17.4s；跨版本值附各自配置，不能替代上表同包 A/B。

512 的 Mac 计算本身更快（中位 8.12s，128 为 10.84s，32 为 14.98s），但接收端等待分别约 6.75 / 3.51 / 0.47s。更大的块增加流水线填充和等待，不能只选 GPU 计算最短的一档。最终状态接收仍约 2.8s。

先 32、再128、再512、最后128确认，非随机次序。一组确认轮与物理 GPU 单测有潜在重叠，已整组排除并另做独占确认；排除 run `suffix-e2e-128-20261023T082656Z` 及原因保留，未挑选单次最快数据。

## 构建、CI 与复现

- 两端 AOT 包散列在 `site/suffix-batching-summary.json`；Mac / Windows 本地各 107 单测通过，35 项 HTTP 集成检查通过。原生图检查增加两组显式 scratch 配置，原 60 项硬件检查通过。
- CI `f751653` 三平台通过：[运行记录](https://github.com/LittleLittleCloud/Infer.NET/actions/runs/39923894758)。macOS runner 实测名称为 Apple Paravirtual device，无法创建 residency set；该硬件用例明确跳过，CI 显示 106 passed / 1 skipped。物理 M5 本机同用例仍实际运行，107 passed / 0 skipped。没有捕获所有 residency 异常并静默当通过。
- 单机诊断：停止已核对的驻留 Mac 模型后，以进程环境 `DEEPINFER_METAL_KV=fp32 DEEPINFER_METAL_PREFILL_CHUNK=32 DEEPINFER_METAL_PREFILL_GEMM=fp32` 运行 `deepinfer split-check-suffix --weights MODEL_DIR --prompt-file PUBLIC_PROMPT --output NEW_DIRECTORY`；最多1024输入，保存原始 trace、state、logits 与比较结果。不要在同 GPU 加载第二份模型。
- 整次测量：同包分别配置 `suffix-chunk-tokens` 为32/128/512，对固定输入调用 `/v1/infer/stream`，`maxTokens=128, profile=true, benchmarkIgnoreEos=true`；每档 warmup 1 + 正式3，验证实际 PREFILL 日志中的 suffixChunk。
- 本地原始证据未公开：`artifacts/dual-host/suffix-check-{9d3304b,a8ce025}`、`suffix-e2e-32-20261023T081727Z`、`suffix-e2e-128-20261023T082142Z`、`suffix-e2e-512-20261023T082422Z`、`suffix-e2e-128-20261023T082851Z`。

本轮依旧是 FP32 / 8K / 128 输出。**BF16 跨主机数值、128K prefill ≥1600→2000 token/s、4096 输出 decode ≥37 token/s、整次 ≤200→180秒均未验收。** 不运行已知超出内存预算的 128K FP32 请求，不放宽精度门槛。下一步让前段计算和网络发送重叠，再处理最终集中 KV 交接。
