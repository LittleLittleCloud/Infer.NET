# TUI 三段式布局与连接信息

2026-10-22，Issue #13 / PR #14。TUI 源码 `0ea14381f1ce98498c41fceb68543eafc3f550ac`，后端部署仍为 `506481445d3670d129ff3d101251f96357efb971`；本轮没有替换或重启模型进程。

## 改动

Header 固定标题、Dashboard / Playground tabs 和连接状态；body 使用剩余终端空间；footer 固定请求统计与三个 API 入口。Playground 的输入框与操作行置于 footer，不再跟随对话内容移动。Dashboard 在至少 100 列时显示 worker 列表与详情双栏，80 列时上下排列。不使用边框，保留鼠标、键盘、滚轮、草稿、多轮历史及 Auto 输出。

连接信息取自当前 coordinator 配置：Chat Completions、Responses 显示完整 URL；Claude Code 显示 base URL 与 `/v1/messages` 路径。界面不显示认证 token。当前仍只有文本 API，未实现工具调用及 Messages token-count，不能据此宣称完整 Claude Code 或编码 agent 兼容。

## 验证

Release 编译零警告、零错误。独立端口、受控文本 worker 的协议 / 官方 SDK / 真实 PTY 检查 **65 项通过**，新增完整 endpoint、凭据不泄露、底部输入区位置以及 80×24、100×30、120×40 缩放检查。原有鼠标点击、滚轮分页、流式输出、取消、多轮历史与 Auto 检查仍通过。

另以真实部署执行 100×30、80×24 的 Dashboard 只读终端检查：连接、prefill/decode 指标和 endpoint 均可见。未向生产模型提交测试推理，未更换 Windows 二进制。原始终端记录仅保留本地 `artifacts/chat/layout-current-deployment/` 和 `layout-three-part-checks.log`。

## 用户延迟观察，非基准

当时的运行遥测显示，一条 1997 输入 / 1685 输出请求 prefill 12.157 秒，decode 56.656 tokens/s，请求 41.897 秒；之后一条 3737 输入 / 241 输出请求 prefill 26.082 秒，请求 122.822 秒。当前 `prefillMs` 计时从发送 Start 到完成 state handoff，包含 Windows 前段、Mac 后段、隐藏向量与 KV / Delta 状态交接，不是纯 GPU 算子时间，也未包含更早的排队与 reset。

日志另有首步 decode 墙钟 92.542 秒、GPU 计时 22.877 毫秒的异常，因此不能把所有等待都归到 prefill。以上只是读取现有用户请求日志，不包含用户输入文本；没有控制缓存、预热、供电或内存压力，不构成正式吞吐基准，也没有证明具体换页原因。首字延迟尚未修复，需要进一步分阶段计时和内存驻留验证。

## 使用

从仓库运行 `dotnet run --project src/DeepInfer.Tui -c Release`。已打开的 TUI 需要退出后重新运行；后台 serve / agent 保持运行。
