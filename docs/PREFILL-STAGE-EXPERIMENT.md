# 双机 prefill 分项计时，2026-10-23

运行候选代码为 [PR #25](https://github.com/LittleLittleCloud/Infer.NET/pull/25) 的 `919259d290855b1ce0401374bf8b3d781ed1ed43`；后续 `1bf5ed3` 更新 TUI 冒烟测试，`2dd2a53` 调整紧凑面板间距，当前 TUI 源码已在根工作区构建。Host/agent 运行二进制仍为 `919259d`。Mac coordinator 与 split-serve 使用该 Native AOT 包，Windows 前台 split-agent 使用同源 Windows 包。模型仍是固定 Qwen3.6-35B-A3B affine8/group64、FP32 GEMM/KV、4090 前20层与 Mac 后20层。没有改内核、权重、状态布局或性能目标。

原始用户请求为 4,642 输入、1,088 输出：Mac 记录 prefill 7.011 秒，后段 5.643 秒，隐藏向量等待 0.812 秒，最终状态等待 0.545 秒，导入 0.011 秒；Windows 日志对应该请求的分块计算累计 5.579 秒，分块帧 socket 写入累计 4.346 秒，从开始到状态发送完成为 6.924 秒。旧版没有单独记录最终状态导出/写入，不能将 4.346 秒称为整个跨机传输或纯网络时延。

新版本在 agent 的认证 End 帧携带单调时钟阶段用时，Mac 将其与自己的用时一起放入 HTTP `timings.prefillStages`、worker `lastRequest.prefillStages` 与 TUI Dashboard。旧 agent 的空 End 仍可接入新 Mac；升级顺序为 Mac 再 Windows。**4090 计算、socket 写入与 Mac suffix 重叠，分项不能相加。Mac 等待远端数据也包含上游计算。**

一轮功能验证使用重复构造的 4,636 输入、8 固定输出；这是分项计时探针，不是正式性能基准。Mac prefill **6.893 秒**；Windows chunk 计算 **5.540 秒**、最终状态导出 **0.013 秒**、socket 写入 **5.624 秒**、前段总墙钟 **6.860 秒**。Mac 等待隐藏数据 **0.639 秒**、suffix **5.704 秒**、等待最终状态 **0.540 秒**、状态导入 **0.010 秒**；线缆载荷 **165,855,232 字节**。客户端请求为 7.067 秒，包含网络/API 开销；8-token decode 吞吐不用于长输出结论。

另两次相同 947 输入、8 输出的真实双机请求 token ID 完全相同，响应及 worker 状态都带分项字段。两端 Native AOT 包完整性校验通过；Mac 125 项单测，Windows 124 通过、1 项平台跳过，Windows CUDA 原生数值自检通过；部署验收 15 项全过，PTY 实际看到 Dashboard 的分项行。Mac 由用户 LaunchAgent 托管，Windows 仍是保留 SSH 会话的前台 agent，SCM/重启持久性未因此得到验收。固定 BF16/131072-token 性能、正确性、内存门槛仍未通过。

公开的精简机器可读证据在 `site/prefill-stage-summary.json`；完整请求和私有配置保留本地，公开文件不含提示词、凭据或内网地址。
