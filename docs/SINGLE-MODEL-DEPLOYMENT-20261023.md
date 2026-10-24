# Qwen3.6 唯一模型与双机更新 — 2026-10-23

关联 [Issue #45](https://github.com/LittleLittleCloud/Infer.NET/issues/45)。Mac coordinator、split-serve 与 Windows split-agent 的实际运行源码统一为 `58bf95018975297ba6f2f965948e2c5f4944cd95`。后续文档提交不改变此构建版本。

## 变更与配置

移除 Qwen2.5 3B/7B、TinyStories 的加载器、专用下载/验证脚本、旧模型元数据及 7B 启动器。CLI/demo/默认启动器只选择 Qwen3.6-35B-A3B，默认 8bit；同模型 5bit 检查点继续支持。`synthetic-v1` 保留为无权重 RPC 测试工具。旧模型实验报告明确标为历史记录。

双机继续使用固定模型 revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`、8bit 权重、FP32 KV、SplitWire v4、20/20 prefill 与 Mac 完整 decode。原有模型、凭据、TLS 与兼容配置复用，不重新下载模型。

更新前 Mac serve 为 `6a0d55d`，coordinator 和 Windows agent 为 `eb492b9`。Windows 当时使用实验配置 `batch-fp32-fast`，正式源码不提供该选项；本次恢复已有正式配置的 `batch-fp32-packed`。两份配置只有 `cuda-prefill` 一项差异，没有改变精度。

## 切换与真实验证

两端使用独立构建目录，通过 GitHub 同步同一完整源码 SHA 后分别构建 Native AOT、校验包清单，保留旧版本用于成对回退。确认活动请求结束且队列连续为空后，停止旧 Windows agent 与 Mac 两个 LaunchAgent，再切换绝对路径启动新包；没有同时加载两份完整模型。

| 检查 | 结果 |
|---|---|
| Mac 单元测试 | 131 通过 |
| Windows 单元测试 | 130 通过，1 项 Metal 平台测试跳过 |
| CUDA 算子数值检查 | 通过 |
| 两端原生包散列、版本、运行路径 | 通过；均为上述源码 SHA |
| 两端 Native AOT RPC / 聊天协议 | 每端 40 / 37 项通过 |
| 源码提交的 Linux / macOS / Windows CI | 全部通过 |
| 真实模型部署验收 | 15 项通过：连续请求、状态 reset、SSE、取消恢复、EOS、超限、错误鉴权、队列背压及 PID 稳定 |
| 17 / 301 输入 token 配对 | 均走真实 CUDA → Metal；各 16 个输出 token 与更新前完全相同 |
| 在线模型列表 | 仅 Qwen3.6-35B-A3B |
| 三个已移除模型的 API 请求 | 均明确返回 HTTP 400 |
| Native AOT CLI 流式请求 | 返回“好”，自然 EOS |
| Windows agent 断连/重启 | 断开时撤销 ready、请求返回 503；恢复两级健康后输出 token 一致，Mac 模型 PID 不变 |

原始日志、请求结果、私有配置/服务定义备份仅保留本地，公开摘要为 [single-model-deployment-summary.json](../site/single-model-deployment-summary.json)。散列用于追踪原始证据，不将本机路径当作公开下载链接。

## 产物散列

| 产物 | SHA256 |
|---|---|
| Mac deepinfer | `ab5d8ab1b389a2c87bf26276c87d9e0829cce443c41adea657c0f03b65451f23` |
| Metal dylib | `1739ff5dd22a4b158f97d84d39268478a75d5683f4a9ad1a3cd0d1a8f79d4aef` |
| Windows deepinfer.exe | `97c9a6ebd3e948775d8de3b49337da6820a5b094133a8165599134c2c488c041` |
| CUDA DLL | `1ee90b3059fdbee349ca841f1b955488feb624af2d211081fcf45d88bbdf57c1` |

## 托管与边界

Mac coordinator / split-serve 由用户 LaunchAgent 托管。Windows 保持原有前台 agent 模式，由保留 SSH 控制会话维持；新建检查连接确认了运行路径与 PID。没有安装 SCM 服务，不宣称脱离控制会话或主机重启后自动运行。Mac 退出登录、两端重启和 Windows SCM 仍未验收。

本轮验证模型清理与部署功能，没有重新测吞吐、HumanEval 或 128K 性能，也不改变 [固定验收规范](QWEN36-8BIT-TARGETS.md) 的门槛。后续性能比较应使用同一配置做配对测量；不能从本轮配置切换推导性能收益。
