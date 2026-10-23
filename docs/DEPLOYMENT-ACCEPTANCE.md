# 常驻双机部署候选验收

测量日期：2026-10-22（America/Los_Angeles）。关联 [Issue #9](https://github.com/LittleLittleCloud/Infer.NET/issues/9)、[PR #10](https://github.com/LittleLittleCloud/Infer.NET/pull/10)，实现叠加于尚未合并的 PR #8。

两端构建与联合测试源码均为 `85c68efe50ceced6bb00a72ce8214e98dd168724`，测试时两端 checkout 干净。后续笔记/Pages commit 仅记录结果，不改变本轮安装包。

[部署与操作指南](https://github.com/LittleLittleCloud/Infer.NET/blob/85c68efe50ceced6bb00a72ce8214e98dd168724/docs/DEPLOYMENT.md) · [三平台 CI](https://github.com/LittleLittleCloud/Infer.NET/actions/runs/39895496808)

## 本轮交付

Mac coordinator 与 Metal split-serve 独立常驻；Windows split-agent 保持 CUDA 前 20 层权重，逐请求重置状态并主动连接 Mac。模型仍为固定 Qwen3.6-35B-A3B affine 8bit/group64；模型 revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`。Mac 完整模型及 Windows prefix bundle 启动时校验固定散列。

数据通道采用固定证书的双向 TLS、SplitWire v2 HMAC/请求/布局/位置校验；本地 API token 鉴权、SSE 流式、自然 EOS、取消、deadline、1 个执行器加 2 个排队名额。配置文件、私钥和 token 不进入 Git。提供带文件散列的 Native AOT 版本目录、macOS LaunchAgent 和 Windows SCM 安装/卸载脚本、升级回退说明。

当前能力边界：FP32 KV，输入加输出最多 2048 tokens，输出最多 256 tokens；不是 BF16/128K 正式性能交付。Mac LaunchAgent 已安装运行；Windows 原生前台服务已运行验证，SCM 安装/开机启动/服务账号 GPU 可见性仍需管理员执行后验收。Mac 用户退出登录或休眠也不保证可用性。日志持久化已提供，自动轮转仍未实现。

## 验收证据

原始文件仅本地保存于 `artifacts/deploy/acceptance-6/`，不将本地索引伪装成公开下载。下列为实际运行结果：

| 检查 | 结果 |
|---|---|
| readiness advertises 8bit FP32 2K | PASS |
| API rejects wrong authentication | PASS |
| output limit rejected | PASS |
| context overflow rejected without truncation | PASS |
| resident request 0 | PASS |
| resident request 1 | PASS |
| resident request 2 | PASS |
| three requests reset state deterministically | PASS |
| streamed token IDs and text match final result | PASS |
| stream and nonstream agree | PASS |
| client can cancel after first streamed token | PASS |
| next request correct after cancellation | PASS |
| natural EOS terminates before limit | PASS |
| bounded queue provides explicit backpressure | PASS |
| Mac model PID remains unchanged | PASS |

8 个同步请求的响应为 3 × HTTP 200、5 × HTTP 429。Mac 同一进程连续完成测试，不重新载入模型。另验证重复启动使用相同配置被独占锁拒绝，原 worker 不受影响。

断线/重连复测：停止经过路径校验的 Windows agent 后，Mac `/ready` 和推理均返回 503；重启 agent 并等待两级就绪后返回 200，32 个输出 token 与此前完全相同，Mac PID 不变。原始摘要位于本地 `artifacts/deploy/reconnect-summary.json`。实际 Native AOT CLI 流式调用输出“好”，正常以 EOS 结束。

Windows 私有配置已增加 `log-file`，SCM 启动后记录持久日志；日志轮转仍需运维归档。

单元测试 85 项；Linux/macOS/Windows CI 各通过构建、25 项 RPC 和 25 项 Native AOT RPC；这些 RPC 检查使用模拟后端。上表另外使用两台实际设备与真实模型执行，二者不混算。CUDA reset 后的算子自检通过。

## 短请求时延观察（不属于正式吞吐基准）

提示词：`用一句话介绍你自己。`；17 个输入 token，固定 32 个输出 token；无 prefix cache，每轮清空 KV/Delta/卷积状态。Mac M5 Pro 48 GiB / Windows RTX 4090 24 GiB，经已有家庭路由网络；8bit 权重、FP32 KV。进程启动执行全层预热，但本轮没有完整的供电/换页同步采样，不作为正式性能验收。

| 连续请求 | prefill ms | decode token/s |
|---|---:|---:|
| 1 | 3336.6655 | 3.9375 |
| 2 | 1092.8051 | 68.2377 |
| 3 | 869.9283 | 72.0619 |

第一轮第二个 token 的 wall time 为 7455.8398 ms，后续 token 恢复约 13–14 ms；不能因为后两轮较快就删去首轮。启动预热尚未消除所有首次请求停顿，原因未定位；没有同步换页证据，不能断言是 swap。

三轮输出 token IDs 一致，JSON 数组 SHA256 `64062696d8d9d6be39d78dc192aaeb80837d6ed43d99b1becc012037bc6d9936`。

## 失败与修复记录

- Windows EphemeralKeySet 不适用于当前 Schannel 路径；DefaultKeySet 在 SSH 身份下出现 AccessDenied。改用 Windows MachineKeySet 后真实 TLS 互通通过。Mac PFX 使用 DefaultKeySet。
- Windows SSH 子进程可能随会话结束；尝试 WMI 脱离会话启动返回 AccessDenied，未修改权限。开发验证保留 SSH 会话，正式托管使用管理员安装的 SCM 服务。
- 首次跨机 TLS 状态交接失败，Mac 报解密错误；仅增加 End ACK 仍失败。跳过 End 帧零长度 payload 的 SSL 写入后真实交接成功，保留回归测试。
- EOS 后立即发并发验收撞上 peer 重连窗口；等待就绪后得到 3 成功/5 限流。Busy 状态改由执行器实际占用派生，避免前请求 finally 覆盖后请求状态。
- 首次重连探针只等 worker ready、未等 coordinator 心跳恢复，断言失败；后续实际请求成功且输出一致。重连验收应等待 worker ready 和 coordinator healthy 两项。

## 产物追踪

| 平台文件 | SHA256 |
|---|---|
| Mac deepinfer | `2c42b00cd9973cf9825144feba0ce9fb7dfcd4f2efa41bb07b0b95f565cb51fc` |
| Mac Metal dylib | `701e01d6ccf99ddc9bf1fb46872658a960bea011c40da5020f4393bad57315ed` |
| Windows deepinfer.exe | `bfbc0f6a979fa52647b3b1d49c177f123640b628b08946f6fd24d3ab6149ad87` |
| Windows CUDA dll | `21bed60a06b4f52d1dc155e4d2c1e7d8a70207ff46e80baf78f352ee6b3ae706` |

本地原始 summary.json SHA256：`a6c7a6ceb997af33cff3bb02409b5e91b02692e187c2ece40f08694a765f4303`。两端包清单各 9 个文件校验通过；模型、凭据、二进制与原始日志未上传普通 Git。

## 剩余验收

- 管理员安装 Windows Service，验证启动/停止、退出 SSH 后驻留、故障恢复及重启主机后的 GPU 可见性。
- 首次请求长停顿、日志自动轮转和更长运行稳定性。
- BF16 KV 与 128K 双机数值、资源、正确性、性能验收仍依照 [固定目标](QWEN36-8BIT-TARGETS.md)，本轮不改阈值。

## 用户启动后的复测（2026-10-22，追加）

用户反馈“启动了”后，通过新建、随后关闭的 SSH 检查连接读取 Windows agent 日志；agent build 仍为上述 85c68ef 候选，Mac build 相同且模型 PID 未变。目标包 exe/CUDA DLL 散列与上表一致；当前账号无法读取该运行进程的 exe 路径，因此包散列验证不替代运行进程路径核验。

15 项真实模型验收再次全部通过，原始证据位于本地 `artifacts/deploy/acceptance-scm-1/`。目录名只是运行标识，**不证明 Windows SCM 已通过**：`sc.exe queryex/qc/qfailure/qfailureflag InferNetCuda` 均返回 1060（服务不存在），指定名称的 Get-Service 同样未找到；日志中的 agent PID 位于桌面 Session 2。CIM 查询另因权限不足失败，未尝试改变权限。已向用户确认具体启动方式，服务安装、开机运行及 SCM 恢复仍保持未验收。

本轮连续三次 17 输入/32 固定输出请求：

| 请求 | prefill ms | decode token/s |
|---|---:|---:|
| 1 | 66435.5217 | 1.4843 |
| 2 | 1300.6365 | 47.5921 |
| 3 | 806.8059 | 72.2919 |

首次 prefill 约 66.44 秒，decode 约 20.89 秒，单 token 最大约 20.47 秒；保留慢轮，不将第三轮恢复到约 72 t/s 表述为稳定性能。长停顿期间采样显示 Metal command queue 在提交系统调用中、模型线程在 `waitUntilCompleted`，只能定位等待发生在 Mac Metal 路径，不能单凭该采样确定根因。没有完整换页增量和供电对照，本轮不是正式性能实验，也不能证明 swap。未因验收重启两端 worker。

本轮 summary SHA256：`bc19d603077b28c55388309f581b97260c7ff2884c4f9316e1085f5b31b9a3db`。采样及服务查询输出仅本地保留，公开记录省略账号、路径与网络地址。
