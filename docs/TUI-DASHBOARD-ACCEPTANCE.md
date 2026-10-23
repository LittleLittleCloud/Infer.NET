# 鼠标 Dashboard、Auto 输出与上下文边界验收

2026-10-22。Issue #13，PR #14（叠加 #12，未合并）。运行源码 `506481445d3670d129ff3d101251f96357efb971`，两端同 SHA 的 Native AOT，M5 Pro 48 GiB / Windows RTX 4090 24 GiB，固定模型 revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`，8bit 权重、FP32 KV。Windows 仍是前台 agent，由现有 SSH controller 保持；Mac 为用户 LaunchAgent。未安装 SCM、未做主机重启验收。

## 改动与用途

- 默认 Dashboard、独立 Playground；不使用边框。真实 SGR 鼠标点击 Tab/worker/按钮/输入框，滚轮分页和翻看历史，保留 Tab/Enter 键盘操作。切换 Tab 保留草稿、历史和正在生成的请求。
- Dashboard 显示心跳、状态、附属 agent 连接、队列、完成数、模型量化/窗口和最近完成请求的 prefill、decode、Metal/KV 分配。未知数据显式标为未上报；Windows agent 没有冒充独立注册 worker。
- TUI 默认 Auto：不固定 128/256 输出，直到 EOS 或剩余模型窗口。Chat/Responses 省略 budget 同样使用 Auto；Messages 保留必填 max_tokens。用户仍可设置明确输出上限。
- 移除部署的 2K context / 256 output 人为上限。CUDA attention 使用按容量分配的工作区，前段 KV 按请求扩容；Metal decode 扩容时复制已有 KV，保留 Delta/卷积状态。SplitWire v3 在分配前检查形状/长度，CUDA ABI 3，两端必须同时升级。
- 模型原生窗口仍是 262144 tokens。实际内存、HTTP 请求大小与 deadline 仍有限；没有声称无限上下文或 262K 整模已可用。

## 实测证据

- Mac 与 Windows 各 106 项单测、Native AOT 构建通过。[最终源码 CI](https://github.com/LittleLittleCloud/Infer.NET/actions/runs/39906897249) 通过。
- 最终 Mac AOT + 官方 SDK + 实际 PTY 共 59 项受控检查通过；使用模拟 worker 的协议检查不算 GPU 推理。
- CUDA GEMV/Delta 数值检查以及 17、2048、4097、8193 tokens × FP32/BF16 的 8 项 attention 检查通过，attention 最大绝对误差 3.27e-8。
- Metal 38 项硬件检查通过：包括 attention 到最后位置 262143，以及 KV 扩容前后字节保留。它们是算子/缓冲检查，不代表 262K 整模型验收。
- 真实双机：旧 API 15 项、Chat/Claude Messages/Responses 15 项、真实 TUI 8 项、上下文/输出 7 项通过。
- 1962 输入 / 300 固定输出：解码越过原 2048 边界；与已预留 4096 容量的第二次运行相比，全部 300 个 token ID 一致。两次总耗时 19.142 / 17.401 秒。
- 3012 输入 / 32 输出：CUDA prefill、跨主机状态导入与 Metal decode 完成，总耗时 19.189 秒。之后短请求 token ID 与旧部署基线一致。Auto 短回答自然 EOS。
- 断开 agent 后，serve 心跳仍健康但状态变为 WaitingForAgent、不可接单，模型从可用列表撤下；重连后输出 token ID 不变，Mac 模型 PID 不变。

这些是功能测试，未按正式性能门槛控制供电、预热与重复次数；不把耗时作为新速度基准。当前家庭网络仍是 Mac Wi-Fi → 路由器 → Windows 1GbE。KV 仍经 CUDA D2H、主机缓冲、TLS/TCP 与 Metal shared-buffer 拷贝，不是 GPUDirect。

## 失败与修复

候选 `317e17f` 完成长请求后暴露心跳采样问题：已 Dispose 的 TcpClient.Client 为 null，导致心跳任务退出、协调器租约过期。`5064814` 增加安全连接探测和回归测试，重新构建/部署两端，重复长请求及全部正式功能回归通过。最早测试在服务 Warming 期间得到 503；后续先等待 worker 与 coordinator 两级就绪再测试。失败日志保留，没有隐藏为成功轮次。

## 复现

先部署两端匹配的 v3 包；旧包成对保留用于回退。使用 `scripts/deploy/acceptance.py`、`scripts/chat-acceptance.py` 和 `scripts/chat-integration.py --sdk --tui`。前两者测真实部署，最后一个用独立端口模拟 worker 检查协议与鼠标行为。

扩容案例需新启动的 serve：向 `/v1/infer` 发送 `prompt = " a" * 1950`、`maxTokens = 300`、`benchmarkIgnoreEos = true`、`profile = true`、模型 `qwen3.6-35b-a3b`、backend `metal`；连续执行两次并比较 tokenIds。再以 `" a" * 3000`、32 输出验证长 prefill；随后重跑短基线与省略输出 budget 的 Chat 请求。请求字段有意固定 EOS 行为，不能把固定输出测试当作自然停止测试。

[公开摘要与产物散列](../site/dashboard-acceptance-summary.json) 保存容量、计时、token ID 散列和检查数量。完整原始日志仅保留在本地忽略目录 `artifacts/chat/dashboard-*`，未伪装为公开下载。真实终端 ANSI/文本记录在 `dashboard-real-tui-final`，受控 PTY 在 `tui-session.ansi`。

下一步仍为 BF16 双机精度、128K 正式性能验收与 Windows SCM 托管；本轮没有修改既定性能目标。
