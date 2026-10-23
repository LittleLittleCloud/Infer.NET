# 双机 resident prefill / decode 接管

实现分支 `codex/split-prefill`，关联 Issue #3。沿用当前家庭网络：Mac Wi-Fi → 路由器 → Windows 有线网口。

## 执行与存储职责

- Windows RTX 4090：固定 8bit 模型的 embedding 与前 20 层常驻；CUDA 执行前段。没有专家按需加载，也没有 CPU 推理回退。
- Mac M5 Pro：完整 40 层常驻；Metal 执行后 20 层，完成交接后独立运行全部 40 层 decode。
- Mac 协调器通过认证 TCP 派发 token IDs。Windows 每算完 32 tokens，发送 FP32 hidden；发送后继续计算下一块，Mac 同时执行上一块后段。TCP 背压限制在途数据。
- 全部 hidden 之后，Windows 发送前段 5 个 full-attention 层的 BF16 K/V、15 个线性层的 FP32 DeltaNet 状态与卷积历史。
- Mac 对每个 tensor 验证身份、位置、大小、序号和 HMAC，再导入共享 Metal buffer。40 个状态 tensor 全部导入并收到结束帧后才能接管；native graph 同样拒绝缺状态的 decode。
- SSH 仅启动/管理 Windows 进程和读取日志；模型数据使用独立 TCP。Windows 主动连接 Mac，无需新增 Windows 入站防火墙规则。

这是分阶段计算和状态交接，并不把两台机器的内存变成一个统一显存池。

## 当前边界

首个实现是**正确性优先的实验执行器**：CUDA 使用逐 token 的自写 affine8 GEMV、路由专家、DeltaNet 和 attention，FP32 乘法/累加，BF16 KV。
Mac 后段每块最多 32 tokens。CUDA attention 暂限 2048；CLI 同时限制输入加输出不超过 2048。逐层验证最多 128 输入。

这与最终性能规范要求的 **BF16 批量 GEMM、128K 输入**存在明确差距，不能用这版的时延验收 [固定性能目标](QWEN36-8BIT-TARGETS.md)。
短请求的数值检查也不能证明长上下文质量或稳定性。
目前入口是 `split-infer` / `split-worker`；尚未接入原有 TUI、调度器多请求队列及租约。
每个 worker/session 仅服务一个请求，完成或失败后释放；不复用污染的 KV，也不自动重试半个请求。

## 协议 v1

双方握手使用 32 字节随机 nonce、请求 UUID 和固定模型 revision 派生 HMAC-SHA256 会话密钥，并分别证明 client/server 身份。
帧为 92 字节 header + payload + 32 字节 HMAC。header 包含 magic、版本、请求 UUID、单向递增序号、类型、层、tensor、token 位置、token 数、字节数与 40 字节 revision。

v1 固定模型 `mlx-community/Qwen3.6-35B-A3B-8bit` revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`、affine8/group64、前段 `[0,20)` 与后段 `[20,40)`；不接受其他 geometry。
所有整数和 tensor 均 little-endian，layout 由版本及 tensor descriptor 唯一确定：

|消息|dtype / shape / layout|
|---|---|
|Start|I32 `[tokens]`；含是否启用数值 trace|
|Hidden|F32 `[chunk,2048]`，层 20 输入|
|Trace（仅诊断）|F32 `[chunk,20,2048]`，前段每层残差输出|
|State：full attention tensor 0/1|BF16 K/V `[position,2,256]`|
|State：linear tensor 0|F32 Delta `[32,128,128]`，value head/value dim/key dim|
|State：linear tensor 1|F32 卷积 `[3,8192]`，按时间先后保存 raw projection|
|End|无 payload，完整 token 位置|

认证和完整性保护不提供加密，限受信任局域网。随机 token 至少 32 字符；启动脚本通过 SSH stdin 传入秘密，子进程使用环境变量，不写公开日志/命令参数。
错误版本、request、revision、重放、篡改、错误 dtype 对应大小、断帧、乱序/重复状态均拒绝。失联不会宣布 decode ready。

## 构建与运行

1. 两端通过 GitHub 检出同一完整 SHA，确认工作树干净。
2. Mac：`dotnet build src/DeepInfer.Host -c Release`。
3. Windows：`scripts/windows/build-prefix.ps1` 构建 SM89 DLL，然后 `dotnet build src/DeepInfer.Host -c Release`。使用已配置的 `environment.ps1` 与 NuGet v2 配置。
4. Mac 导出权重：`python3 scripts/split/export-prefix.py models/qwen3.6-35b-a3b-8bit-mlx artifacts/split/prefix.bin`。
5. 将 **权重产物** `prefix.bin` 放入 Windows checkout 的 `artifacts/split/`；源码依然只通过 GitHub。bundle 为 18,415,842,388 bytes，固定 SHA256 `ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77`。Windows 每次加载前验证完整散列。
6. Windows 算子检查：`python scripts/split/test-cuda.py`。
7. Mac：`python3 scripts/split/run-dual.py --ssh YOUR_CONFIGURED_ALIAS --verify --prompt '用一句话介绍你自己。' --max-tokens 8`。

统一脚本先跑 Mac 完整模型参考，再启动两端执行。`--verify` 包含逐层输出、所有前段状态、每步完整 logits 与 greedy token 比较；失败写 `result.json` 并返回非零。
诊断 trace 的传输与比较计入记录时间，因此这不是性能基准。移除 `--verify` 仅运行请求，不会把未经数值检查的请求标为 Verified。

原始证据放在两端 `artifacts/split/<run-id>/`，每次独立目录；保存 source SHA、产物散列、PID、日志和退出码。大日志与权重不纳入 Git。
结束时间由 Mac 同一单调时钟记录：`HiddenReadyMs` 仅表示后段已计算，`DecodeReadyMs` 才包含完整状态导入，`ElapsedMs` 直到生成结束。

## 验证记录

截至首轮模型验收之前：两端 managed build 均为 0 warnings / 0 errors；Mac 76 项单元测试通过。
4090 独立 CUDA GEMV max abs `3.8082e-8`，DeltaNet 输出/状态 max abs `1.0201e-8`，通过独立 scalar double 累加对照。
完整模型数值与跨机接管结果见后续记录；算子成功本身不代表端到端成功。
