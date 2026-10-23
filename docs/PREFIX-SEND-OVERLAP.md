# CUDA 计算与 hidden 发送重叠

2026-10-23，继后段128-token合批，新增 `cuda-send-overlap=on|off`。同包 A/B 后选择 **on**：8K prefill 中位 17.365 → 16.624 秒，减少4.27%；客户端整次19.647 → 18.956秒，decode保持约56 token/s。模型仍为固定 affine8/group64，FP32 GEMM/KV；不是BF16或128K验收。

## 实现与安全性

一个生产任务独占 CUDA session，逐32-token块计算并读回；一个发送任务按原顺序通过已有TLS/HMAC协议发送。最多排队两个frame，另有当前生产/发送的frame；不缓存整个请求，不并发执行同一GPU模型。最终40个状态张量仍在全部hidden发送结束后交接，本轮没有提前传KV。

取消、发送错误或GPU错误时，先取消排队并等待生产任务结束，才返回调用者，避免下一次reset与在途GPU调用竞态。没有后台重试。四项新增测试覆盖顺序、背压、取消后join/dispose、生产异常传播，本机111项单测通过；36项HTTP集成检查通过。

## 同包实测

两端Native AOT `2760e449364fbfd18f192d80aa66e41eb734a5da`；Mac进程与配置不变，只重启Windows agent切换显式开关。Mac后段128、scratch512；CUDA按层batch-fp32，wire32。固定8192输入、128输出，greedy忽略EOS、每次reset、0%prefix cache。每档预热1次、正式3次。Mac AC / High Power，既有Wi-Fi/Ethernet LAN及TLS，未用SSH tunnel。

|发送重叠|正式prefill秒|中位prefill秒|客户端首token中位秒|客户端整次中位秒|decode中位token/s|
|---|---|---:|---:|---:|---:|
|off|17.592 / 17.365 / 17.228|17.365|17.374|19.647|55.88|
|on|16.624 / 16.547 / 16.790|16.624|16.636|18.956|55.84|

8次请求的全部128个输出token与之前串行基线相同，每次新增整机swap-out为0；每次Metal allocation为37,572,984,832字节。Windows整GPU500ms采样峰值19,585MiB。逐次预热、资源、包散列、阶段数据见 `site/send-overlap-summary.json`。

CUDA前段计算/readback wall time仍约13.4秒；原累计约0.8–1.1秒发送现在能与下一块计算重叠。`computeMs`并非纯GPU事件时间，`sendMs`包括framing/TLS，重叠时间不能相加当总耗时。前段hidden全部发送约14.35 →13.59秒；最终状态交接仍约3秒。off先、on后，非随机次序。一个驱动启动时日志文件尚未建立的尝试发送了0个推理请求，保留但不计入数据。

## 真实取消及部署验收

开启重叠时，8192-token prefill进入Busy后约1.07秒主动断开客户端，2.38秒后新鲜遥测确认两端恢复。随后8192输入/8输出与基线前8个token相同，连续Mac短请求也确定性；Mac与Windows PID均未变。原始证据 `artifacts/dual-host/overlap-cancel-20261023T084752Z`。

另有15项部署验收全部通过：真实连续请求、流式一致、首token后取消与恢复、自然EOS、超限/鉴权及队列背压。切换前也验证a8ce025的ReadyLocal行为：Windows离线时短请求200（0.293s），长请求503（5.005s）。Windows仍以前台SSH控制方式运行，未冒充SCM服务。

CI三平台通过：[2760e44运行](https://github.com/LittleLittleCloud/Infer.NET/actions/runs/39926055286)。虚拟GPU的residency测试覆盖边界沿用后段合批报告；物理M5实际运行该测试。

## 复现及边界

同一包保持Mac `suffix-chunk-tokens=128`，Windows分别 `--cuda-send-overlap off/on`，以agent Ready日志核对实际开关，再对固定8192输入执行 `/v1/infer/stream`，`maxTokens=128, profile=true, benchmarkIgnoreEos=true`，每档完整预热1+正式3。输入SHA256 `defa12f5be86a88aed8a25aa479ab1b17860a661ff364865708e121cac17206d`，revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`。

本地原始证据未公开：`artifacts/dual-host/overlap-e2e-off-20261023T084130Z`、`overlap-e2e-on-20261023T084415Z`、`overlap-release-2760e44`。公开JSON不含凭据或私网地址。

从最初同精度串行的52.503秒到当前16.624秒，8K prefill累计约3.16倍，但128K prefill≥1600→2000 token/s、4096输出decode≥37 token/s、整次≤200→180秒仍待验收；不把短请求decode成绩当作长上下文通过。下一步将KV交接与计算重叠，并验证更长输入/输出。
