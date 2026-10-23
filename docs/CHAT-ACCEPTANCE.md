# Chat HTTP 与 TUI 真实双机验收

日期：2026-10-22（America/Los_Angeles）。[Issue #11](https://github.com/LittleLittleCloud/Infer.NET/issues/11) · [PR #12](https://github.com/LittleLittleCloud/Infer.NET/pull/12)，叠加未合并 PR #10。

运行代码与两端 Native AOT 包均为 `91479e3ac0bb6897fdeb8c90a4255f31fb9140db`。Mac coordinator、split-serve 与 Windows split-agent 已切到该版本；实际进程路径、build、包清单及散列已核对。后续提交只补验收脚本、文档与 Pages，不重新标记运行二进制。

[API 与 TUI 使用指南](https://github.com/LittleLittleCloud/Infer.NET/blob/91479e3ac0bb6897fdeb8c90a4255f31fb9140db/docs/CHAT-API.md) · [三平台 CI](https://github.com/LittleLittleCloud/Infer.NET/actions/runs/39902470818) · [公开验收摘要](../site/chat-acceptance-summary.json)

## 交付内容

- TUI 默认连接本机 coordinator 5200，无参数启动读取现有仓库部署配置。紧凑无边框布局；Microsoft.Extensions.AI.IChatClient 驱动的真实流式多轮对话，Enter 发送、Stop、新会话、system/output 设置、历史浏览。
- HTTP `/v1/chat/completions`、`/v1/messages`、`/v1/responses` 均支持非流式与 SSE，`/v1/models` 列出可用模型。Bearer、x-api-key 兼容现有集群 token。
- 角色化历史送入 Qwen 模板；不把多轮内容扁平拼成一个 user prompt。取消/失败的 TUI 轮次保留屏幕内容但不加入后续历史。
- Responses 支持完整历史回传，包括本接口生成的 output items；不存储 response，不支持 previous_response_id。
- 三种格式的文本、usage、停止原因与错误语义分别映射；流中断不发送成功完成标记、不自动重放部分输出。

当前模型仍是 Qwen3.6-35B-A3B affine 8bit/group64，revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`。Mac M5 Pro 48 GiB + Windows RTX 4090 24 GiB，经现有家庭 LAN；自写 Metal/CUDA 路径，FP32 KV、输入加输出最多 2048 tokens、输出最多 256 tokens、每轮重算 prefill，无 prefix cache。接口兼容的是文本子集；tools、图像、JSON schema、自定义 stop 和非 greedy 采样明确返回错误。

## 验证结果

| 检查层 | 结果与范围 |
|---|---|
| 单元测试 | 102 项通过 |
| Linux/macOS/Windows CI | 构建、Native AOT；每平台普通与 AOT 各 25 项旧 RPC、32 项新协议检查通过；均为受控测试后端 |
| 本机 Native AOT + 官方 SDK + PTY | 48 项通过，含 OpenAI/Anthropic SDK、Responses output 回传、输入清空、Stop/New chat、多轮历史、无边框；受控文本 worker |
| 真实双机旧 API | 15 项通过；状态 reset、确定性、SSE、EOS、取消恢复、容量、鉴权、队列与模型 PID |
| 真实双机兼容 API | 15 项通过；三格式 full/stream 与 token usage 一致，角色记忆、Responses 历史、增量到达、取消恢复、输出截断、连续短回答 |
| 真实模型 TUI | 5 项通过；无参数/无凭据环境变量连接部署 5200、无边框、IChatClient 生成、多轮 Enter、增量显示 |
| 独立 tokenizer 对照 | 4 条 system/user/assistant/user 消息，C# 与固定模型官方 chat template / enable_thinking=false 的 43 个 token 逐项一致 |

真实 SDK 检查使用“记住 blue，再问颜色”的完整历史；三种格式均回答 `blue`，并核对实际 prompt/output token usage。取消后再次发相同历史仍得到相同回答，连续三次短回答均成功。Responses output 项直接作为下一轮 input 使用也通过。

一个 64 输出 token 的真实流式调用收到 64 个文本增量：首文本 0.775 秒，总耗时 1.647 秒；这里只证明增量先于完成到达，不作为稳定 TTFT 或吞吐基准。实际 TUI 第二轮回答解释 prefill，显示 57 output tokens，首文本约 2116 ms、总耗时 3.35 秒；终端记录为真实模型输出，非效果图。

本轮 8 并发旧 API 检查为 3 个 200、5 个 429；同一 Mac 模型进程完成全部真实测试。原有 17 输入/32 固定输出三次观察如下，供故障追踪，不做性能验收：

| 请求 | prefill ms | decode token/s |
|---|---:|---:|
| 1 | 1245.352 | 50.51 |
| 2 | 938.597 | 71.02 |
| 3 | 769.704 | 68.69 |

本次没有完整供电、换页与热状态采样，也没有长上下文测试；不能据短请求结果宣布首请求停顿已解决或 128K 性能达标。

## 发现并修复的问题

- SSE 首事件后重复写响应头导致流失败；改为只在响应尚未开始时设置，加入三格式终止与截断检查。
- RazorConsole 0.6 输入组件在发送后保留旧缓冲，多轮出现旧文本前缀；按轮重建输入组件并验证第二轮请求正文。使用纯文本组件避免方括号被重复转义。
- Responses SDK 回传本接口 output 时含空 logprobs；允许空/null 元数据，非空仍拒绝，并验证真正 output → input 往返。
- 首轮真实 SDK 验收在短 EOS 后立即调用 Messages stream 得到 503；CUDA 下一条 TLS 连接尚未就绪。最终版本在有限队列内等候最多 5 秒，连接缺失返回 503；不启动 GPU 工作、不重试推理。修复后完整 15 项真实 SDK 测试两轮通过，含连续短请求。失败记录保留于本地 `artifacts/chat/real-chat.log`。

## 复现

仓库根目录运行 TUI：

```sh
dotnet run --project src/DeepInfer.Tui -c Release
```

独立 Python 测试环境安装 openai、anthropic 后，连接已就绪的真实部署：

```sh
python scripts/chat-acceptance.py --token-file /PRIVATE/mac/token.txt --output artifacts/chat/real-UNIQUE
```

原有真实部署测试仍使用 `scripts/deploy/acceptance.py`。受控协议/SDK/终端检查使用 `scripts/chat-integration.py --sdk --tui`，额外需要 pexpect、pyte；这些包仅供测试，不是产品依赖。

## 部署状态与边界

Mac coordinator/serve 使用用户 LaunchAgent；Windows agent 从新版本目录以前台方式运行，由保留 SSH 控制会话维持。**没有创建或宣称 Windows SCM 服务**。旧版本目录、配置及启动定义备份已保留。Windows SCM、退出登录/重启恢复、长时稳定性仍按部署 Issue #9 单独验收。

## 产物与证据

| 文件 | SHA256 |
|---|---|
| macHost | `496652b264853f07dd1116107cfd74ba877888f458a29b3bf420523295ec97e4` |
| metal | `701e01d6ccf99ddc9bf1fb46872658a960bea011c40da5020f4393bad57315ed` |
| windowsHost | `7838fd6e23ac1a7f2e223f225a8fb8c3b70b1e3853051584425034b0f1775a48` |
| cuda | `cb90b1801a7066d72d63c1e677b2baf0a0f96c5a0d3be04b78830087309bc2a9` |

两端版本包各 9 个文件校验通过。原始日志、私有配置与二进制仅保留本地，不公开私网地址、账号或凭据。以下本地文件索引不是公开下载链接；可公开的检查列表与散列汇总在页面摘要中。

- `artifacts/chat/real-legacy-final/summary.json`：`2872b73afd22c9578f892c1ffa609534d7dc704297e540d6cd7cf36d6113a84e`
- `artifacts/chat/real-public-final/summary.json`：`c9f18e796e8b134305368427030fef4beddc16aceae2364f4533850e49171916`
- `artifacts/chat/real-tui-final/summary.json`：`2df000dab1019ddd18b18bda1a4f00f717030c1a4026482976fe560b65ad1a03`
- `artifacts/chat/token-probe/comparison.json`：`586b82dce7c918661d3b64ebf86f40533f5e5ac865f58aab04d9d5b7829f80ba`
- `artifacts/chat/aot-91479e3.log`：`db75550f6f6ed2b32fd23467195c9b23d4f0926d4a783d5428e2bd68072031ba`
- `artifacts/chat/windows-91479e3.log`：`5a89fe47db7164383b7b4339431580d1c5381ab535bcad084a6258517f565c1b`
