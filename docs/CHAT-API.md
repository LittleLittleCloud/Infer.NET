# HTTP chat APIs 与 TUI playground

实现分支 `codex/tui-dashboard`，关联 Issue #13，叠加 chat PR #12。模型仍是 Qwen3.6-35B-A3B 8bit / FP32 KV。候选版本移除原部署的 2048 context / 256 output 人为上限；输入加输出受固定模型原生窗口 262144 tokens、实际内存和请求 deadline 约束。此容量声明不代表 262K 整模或性能验收已通过。

## 入口

统一访问 Mac coordinator，当前部署 `http://127.0.0.1:5200`。TUI 默认端口已从 5100 改成 5200。worker 本机 HTTP 5201、Windows 主动连接 Mac 数据端口 5210，计算职责未变。

| HTTP 路径 | 请求内容 | 非流式 | `stream: true` |
|---|---|---|---|
| `/v1/chat/completions` | `model`, `messages`, `max_tokens` 或 `max_completion_tokens` | `chat.completion` | `chat.completion.chunk`、可选 usage、`[DONE]` |
| `/v1/messages` | `model`, `messages`, 必需 `max_tokens`，可选顶层 `system` | Claude `message` | message_start → content_block_start/delta/stop → message_delta → message_stop |
| `/v1/responses` | `model`, `input`，可选 `instructions`, `max_output_tokens` | `response` | created/in_progress → output_item/content_part/text 事件 → completed 或 incomplete |
| `/v1/models` | GET | 当前健康且支持结构化 chat 的模型 | 不适用 |

兼容的是文本对话接口格式，推理由本地 Qwen 执行，不调用 OpenAI 或 Anthropic 云服务。使用已有集群 token：支持 `Authorization: Bearer TOKEN`、`x-api-key: TOKEN` 或 `X-DeepInfer-Token: TOKEN`；内部注册/RPC 仍要求原有 header。默认 HTTP 仅监听 loopback，跨主机数据通道保持双向 TLS。

支持 system/user/assistant 文本消息、文本内容块，Chat/Responses 的 developer 映射为 system。Messages API 用顶层 system。Responses input 可以是字符串或完整消息历史，也可回传本实现生成的 assistant message 项。当前要求最后一条消息为 user；不支持 assistant 预填。所有历史由 worker 按角色编码为 Qwen 消息模板；用户正文中的特殊 token 字面量不会成为角色分隔符。

只有 `*-chat-v1` worker 可处理结构化历史，旧 worker 会被排除，避免旧版本忽略新增字段。旧 `/v1/infer` 和 `/v1/infer/stream` 继续可用。

## 明确的能力边界

- 当前 greedy decoding：省略 temperature/top_p，或传 `temperature: 0`, `top_p: 1`；Chat 的 `n` 只能是 1。
- 图片/音频、工具调用、JSON schema/JSON mode、自定义 stop、logprobs、非 greedy 采样均未实现，相关参数返回 400，不静默忽略。
- Responses 是无状态接口，不存储 response；可传 `store: false`。`store: true`、previous_response_id、后台响应和对话存储不支持。多轮请求显式重传历史，每轮重新 prefill，无 prefix cache。
- Chat/Responses 省略输出上限时使用剩余模型窗口，遇到 EOS 即停止；Messages 仍要求显式 max_tokens。TUI 默认 Auto，同样不预留整个窗口的 KV，而是在 decode 中按需扩容并保留已有状态。
- 超限返回错误，不自动裁剪历史。达到输出上限时 Chat 为 length，Messages 为 max_tokens，Responses 为 incomplete/max_output_tokens；EOS 分别为 stop、end_turn、completed。
- 流式文本是真实增量。input/output usage 来自 worker token 数，不用字符数估算。下游 429/输入错误在流开始前保留 HTTP 状态；流开始后错误以事件报告，不发送成功完成标记，不重试部分输出。
- 不实现其他 provider 路径，例如文件上传、Response retrieve/delete 或 Messages token-count API。SDK 互通只代表以上文本子集。

## TUI

在仓库根目录运行：

```sh
dotnet run --project src/DeepInfer.Tui -c Release
```

无地址参数时，如果存在本机部署 `artifacts/deploy/private/mac/coordinator.json`，TUI 读取其 listen/token-file 连接当前部署。也可以显式指定地址，凭据通过环境传入：

```sh
export DEEPINFER_TOKEN_FILE=/PRIVATE/mac/token.txt
dotnet run --project src/DeepInfer.Tui -c Release -- http://127.0.0.1:5200
```

TUI 通过 Runtime 中的 `InferChatClient : Microsoft.Extensions.AI.IChatClient` 调用公共 Chat Completions 接口，提供 GetResponseAsync 与 GetStreamingResponseAsync。UI 每轮使用 streaming；system prompt、历史、模型与输出上限从统一适配器传入，不绕过 coordinator。

紧凑界面不使用边框，默认 Dashboard，另一个 Tab 为 Playground。支持鼠标点击 Tab、worker、按钮及输入框，滚轮翻看 worker 或对话。Dashboard 显示心跳、在线/离线、serve 状态、附属 agent 连接、排队数、完成数、量化与窗口、最近完成请求的 prefill/decode 和 Metal/KV 分配；没有上报的数据明确显示未知，不编造 GPU 利用率。Windows agent 是 serve 的附属设备，不冒充独立注册 worker。切换 Tab 保留对话、草稿和正在进行的生成。Playground 包含对话区、输入框、操作行与统计。Enter 发送，Tab 移动焦点，Send/Stop/New chat/Settings/Older/Latest 可用；Settings 调整系统提示词和输出上限；默认 `auto`，也可输入正整数。鼠标依赖终端支持 SGR 事件，Tab/Enter 可继续使用。生成时可输入下一轮草稿，但不会同时提交第二个请求。Stop 保留部分输出供查看，取消或失败的轮次不加入下一轮模型历史；New chat 清空会话。历史超过容量会明确报错，可主动开启新会话。

显示 first text 时间、输出 token 数和请求总耗时，不把包含网络/prefill 的平均速度冒充模型 decode 吞吐。TUI 不加载模型，不启动后台 worker，不内置工具执行权限。

## 复现测试

```sh
dotnet test tests/DeepInfer.Tests -c Release
python3 scripts/chat-integration.py
python3 scripts/chat-integration.py --executable /RELEASE/deepinfer
```

协议测试使用独立端口与受控文本 worker，不占用 GPU、不改现有部署。额外安装 openai、anthropic、pexpect、pyte 到独立测试虚拟环境后，可以运行 `scripts/chat-integration.py --sdk --tui` 做官方 SDK 与真实 PTY 交互检查；这些包不是产品依赖。真实双机模型测试和部署结果单独记录，模拟检查不能代替它们。

TUI 参考 [RazorConsole LLMAgentTUI](https://github.com/RazorConsole/RazorConsole/tree/main/examples/LLMAgentTUI) 的对话与输入布局思路，保留已固定 RazorConsole 0.6.0，未引入它的工具执行框架。协议依据：[OpenAI Chat streaming](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events)、[Responses events](https://developers.openai.com/api/reference/resources/responses/streaming-events)、[Claude streaming](https://platform.claude.com/docs/en/build-with-claude/streaming)、[Microsoft IChatClient](https://learn.microsoft.com/en-us/dotnet/ai/ichatclient)。
