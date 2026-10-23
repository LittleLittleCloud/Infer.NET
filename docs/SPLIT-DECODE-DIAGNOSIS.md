# Split decode 冷启动诊断与部署方案

2026-10-22（America/Los_Angeles）。关联 #3、#4、PR #8。本记录是诊断与设计，未通过正式性能验收，也没有安装常驻服务。

## 实测

两次均为 Qwen3.6-35B-A3B affine8/group64，固定 revision 见性能目标；FP32 KV/GEMV/递归状态，19-token 输入、256-token 固定输出。提示词相同、输出 token IDs 全部相同。无 prefix cache hit。

|运行|参考预运行|交接就绪|交接后生成|255 个间隔 / 生成时间|
|---|---|---:|---:|---:|
|split-20261023T023146|无|2.626549 s|27.946580 s|9.124551 t/s|
|split-diagnose-20261023T024150|Mac 完整参考 prefill + 256 输出|1.826793 s|4.390947 s|58.074033 t/s|

首个输出由 prefill 产生，所以分子为 255。此前启动脚本误用 256，显示 9.16 t/s；修正为约 9.12 t/s。这只解释 0.4% 的显示误差。

新诊断使用双端 Native AOT，676 项数值检查通过，256/256 greedy tokens 与参考一致，5 项交接保护通过，两端退出 0。同一输入的全 Mac 参考解码按 logits 文件写入时间估计约 67.72 t/s；它是文件时间估计，不是正式计时指标。

测试插电、High Power。旧 5bit worker 保留，未关闭其他应用。每约 2 秒采样整机 VM；整次加载/参考/双机测试约 94.6 秒，新增 swap-in 6.69 GiB、swap-out 7.84 GiB。约 8 秒、包围双机 decode 的采样窗口中新增 swap-out 为 0、swap-in 约 11.84 MiB。采样窗口通过本机文件时间近似对齐，不能把这些整机计数精确归因到模型或某个 token。原慢运行未采集逐 token 和换页数据。

这不是控制变量充分的 A/B：参考预运行、文件读写、系统驻留状态不同，只有一轮。不能据此声称稳定提速 6.36 倍或宣布内存门槛通过。

## 瓶颈判断

1. **已排除 decode 的逐 token 网络通信。** `SplitInference` 收完全部状态和 End 后，生成循环只调用本地 `MetalSplitSession.Forward`。交接的约 0.6 秒已计入 prefill，不计入此处 27.95 秒。
2. **首要嫌疑是首次 GPU 使用和内存驻留。** 普通 split 只在 Mac 跑后 20 层 prefill；前 20 层虽已加载，但首次完整 decode 才在 Mac 使用。验证模式先让全部 40 层运行，且重复相同输出。此次从 9.12 到 58.07 t/s 支持这一方向，尚不能确定原来的慢耗时是否集中在首步。
3. **FP32 诊断配置有代价，但不能解释全部差距。** 启动器清空 DEEPINFER 配置，关闭矩阵 prefill，使用 32-token 分块、sequential Delta，以及 FP32 KV，未启用 BF16 fused attention。GEMM/Delta 分块开关主要影响 prefill；decode 仍使用原融合专家 GEMV、GPU argmax、每 token 一个 command buffer。19→274 的短上下文不足以仅凭 KV 宽度认定数倍差距。
4. **常驻服务缺失。** 目前每次请求都重启两端、加载权重、首次使用 GPU。模型加载虽排除在 result 计时外，但延迟发生的驻留/首次执行仍可进入 decode。`--verify` 是诊断工具，不能作为正式预热方案。

下一次定位应记录每一步 wall/GPU 时间、提交等待、加载/预热/正式请求各阶段 VM 增量。同一进程、同精度、同输入，先测首次请求再预热并测三次；然后做相同条件的单 Mac/双机对照。BF16 恢复另过数值门槛，不用改变精度掩盖问题。

## 正式部署设计（待实现）

- Mac：轻量 coordinator/API + 独立 Metal worker。只有 worker 加载一份完整 8bit 模型，兼顾后 20 层 prefill 和全部 decode。TUI、CLI、Web 都连接同一 API，不再分别启动模型。
- Windows：常驻 CUDA worker，前 20 层权重加载一次，等待请求。保持现有家庭网络；SSH 只用于安装、升级和诊断，正常推理不依赖 SSH 会话。
- 初版请求并发 1，有界队列；状态机为 Loading → Warming → Ready → Busy → Ready。启动预热完整 40 层和 decode，清空 KV/Delta/卷积后再接流量。模型已分配不等于 Ready。
- 短输入可按实测交叉点显式走 Mac-only，并记录路由；长输入走双机。不能预设短输入双机更快。当前 CUDA 串行 token 前段仍需批量算子优化。
- 请求支持流式输出、自然 EOS、取消、超时、长度校验和明确 finish reason；基准模式单独保留固定输出。当前 2K/256 限制必须在能力接口展示，未验收前不开放 128K。
- Mac 采用 launchd 用户 LaunchAgent，登录后常驻；若需要无人登录启动，单独验证 LaunchDaemon 下的 Metal 环境。Windows 集成 Windows Service 生命周期，以专用低权限账号运行；安装服务是独立的管理员安装步骤。
- 发布两个版本化 Native AOT 包：osx-arm64（可执行文件、Metal dylib、服务模板）、win-x64（可执行文件、SM89 CUDA DLL、所需运行依赖、服务安装脚本）。部署前检查 OS/GPU/驱动兼容性；目标机无需源码、编译器或 SSH 运行请求。
- GitHub Releases 发布二进制、校验值和清单；清单固定源码 SHA、后端 ABI、协议、模型 revision、量化和状态布局。模型单独管理并校验散列，不打进应用包。配置/密钥/日志与版本目录分离。
- 控制 API 与跨机状态传输采用 TLS，worker 使用双向认证；保留现有请求/位置/完整性检查。当前 HMAC 通道没有加密，TLS 属于待实现项。
- 升级先排空队列，停旧 worker，校验新版本并预热，通过后开放流量；失败恢复旧版本。48 GiB Mac 不并行加载新旧两份完整模型，因此初版接受维护窗口。协调器不重放已经输出部分 token 的请求；失败请求丢弃状态、显式报告。
- 运行指标包括启动/预热耗时、队列时长、客户端首 token、handoff 时间、decode 分段速度、GPU 用量、整机 swap 增量、失败原因和版本。单机回归、BF16 数值与 128K 固定目标仍按 `QWEN36-8BIT-TARGETS.md` 验收。

系统服务依据：[Apple launchd](https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html)、[Microsoft Windows Service](https://learn.microsoft.com/en-us/dotnet/core/extensions/windows-service)。

## 证据与复现

源码 checkout：`41c4f2efa059972cf015c538c4d5388b3cf41b4c`；沿用已有 native 产物，修改仅为启动器展示。Mac executable SHA256 `fffb24cb0d26671be3d64b9e128d339a0495dbae67f35cf402236cfe7a0e2f54`；Metal dylib `701e01d6ccf99ddc9bf1fb46872658a960bea011c40da5020f4393bad57315ed`。

结果 SHA256：`f2f9f31d9ea19ab2716e12692d4dc365ee13c33e0b09bab22e831f828414e8fc`；输入 UTF-8 SHA256：`468994db24ccc5c0f5747de6f36855da091f7011e90dd5c3852fc53de8ab269f`。

本地原始证据（未公开）：`artifacts/split/split-diagnose-20261023T024150/`、同级 `split-diagnose-20261023T024150-monitor/`。Windows worker 的对应 run 目录保留 PID、二进制散列、日志和退出状态。

```sh
python3 scripts/split/run-dual.py --native --kv fp32 --verify \
  --prompt "解释一下存算分离推理架构" --max-tokens 256
```
