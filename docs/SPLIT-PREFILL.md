# 双机 resident prefill / decode 接管

2026-10-22，实验实现见 [PR #8](https://github.com/LittleLittleCloud/Infer.NET/pull/8)，关联 [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3)。**真实双机短请求已跑通，但通过的是显式 FP32 KV 诊断模式；BF16 目标路径、128K 与性能验收仍未通过。**

沿用现有网络：Mac Wi-Fi → 家庭路由器 → Windows 有线千兆网口。产品使用自写 C#/CUDA/Metal，没有调用 MLX、llama.cpp 或 vLLM 执行推理。

## 执行与存储职责

- Windows RTX 4090：固定 8bit 模型的 embedding 与前 20 层常驻，CUDA 执行前段。没有专家按需加载或 CPU 推理回退。
- Mac M5 Pro：完整 40 层常驻；Metal 执行后 20 层，完成交接后独立运行全部 40 层 decode。
- Mac 通过认证 TCP 派发 token IDs。Windows 每算完最多 32 tokens，发送 FP32 hidden 后继续下一块；Mac 同时处理上一块后段，TCP 背压限制在途数据。
- hidden 之后，Windows 发送前段 5 个 full-attention 层的 K/V、15 个线性层的 DeltaNet 状态与卷积历史。
- Mac 验证每个 tensor 的身份、位置、大小、序号与 HMAC，再导入 Metal 共享缓冲。40 个状态 tensor 全部导入并收到 End 后才能接管；native graph 也拒绝缺状态的 decode。
- SSH 仅负责启动、Git 同步和日志；模型数据走独立 TCP。Windows 主动连接 Mac，无需新增 Windows 入站防火墙规则。

这不构成统一显存池。模型权重始终是 affine8/group64；`--kv fp32` 仅提高缓存精度，用于隔离计算差异与状态交接问题。

## 实测结果

两端 managed build 和 Native AOT publish 成功。Mac / Windows 各 **79 项单元测试**通过，覆盖篡改、重放、截断、错误密钥、错误 KV profile、tensor 大小与上下文限制。
CUDA 独立 GEMV max abs `3.8082e-8`，DeltaNet 输出/状态 max abs `1.0201e-8`，通过 scalar double 累加参考。

|模式|输入 / 输出|数值检查|最大 relative L2|最大 absolute|greedy tokens|
|---|---:|---:|---:|---:|---|
|Managed，FP32 KV|17 / 8|388/388|7.2243e-7|8.4639e-6|8/8 一致|
|Managed，FP32 KV，两块 32+29|61 / 32|1292/1292|2.6345e-6|4.1962e-5|32/32 一致|
|两端 Native AOT，FP32 KV|17 / 8|388/388|7.2243e-7|8.4639e-6|8/8 一致|

参考是同一固定 8bit 模型的完整 Mac Metal 路径，FP32 GEMV、FP32 KV、每块最多 32 tokens。比较前 20 层全部 hidden、全部前段状态、每一步完整 logits，门槛保持 relative L2 < 1e-4、max abs < 0.01。
AOT 另实测五项拒绝行为：无状态时 decode、错误 token 位置、错误字节数、重复状态、缺最后一份状态时 decode。全部拒绝后正常完成导入与生成。

Managed 两轮源码为 `b180ba405a5485fbd53f4080592bdd5423973ce8`；AOT 与五项拒绝检查源码为 `c25d20970bce198646d7befe08622d685150b4da`。后续代码排版不改变算法。公开摘要含原始结果 SHA256、两端二进制散列：[split-prefill-summary.json](../site/split-prefill-summary.json)。原始 tensor/日志仅保存在两端 `artifacts/split/<run-id>/`；散列不是公开下载链接。

诊断单次 `DecodeReadyMs`：Managed 61/32 为 1923.245 ms，AOT 17/8 为 1604.024 ms；整次分别 2629.277 / 1911.793 ms。**这些不是性能基准**：包含 trace、数值比较与文件 I/O，没有正式预热加三轮测量。旧 5bit 服务保留，加载完整 8bit 模型时出现明显系统 swap，未通过资源验收。不能据此宣称双机加速或达到 2000 token/s。

## 失败实验与当前限制

- FP32 GEMV + BF16 KV：17/8 的 8 个 token 相同，但 7/388 项严格状态/logits 检查不通过；最大 relative L2 `2.5934e-4`，最大 absolute `0.015625`。
- CUDA 归约顺序实验：首次对照因 Windows 运行目录 DLL 过期而无效，保留记录；修复后实际 ascending 对照为 9 项失败，没有采用。
- 短块强制 BF16 乘法输入 + BF16 KV：376/388 项失败，8 个 token 仍相同，没有采用。该参考本身也已变化，不能把它与固定 FP32 参考混算。诊断显示 FP32 小差异跨越 BF16 舍入边界后会累积；这解释观察方向，尚不能证明所有差异原因都已排除。
- 期间发现并修复：Python 3.9 散列 API 兼容；MSVC `Platform=x64` 泄漏导致 .NET 输出到另一目录；32-token scratch 不足以容纳 64 行 padded BF16 tile（执行前中止该轮）。增加源码散列、运行库散列及 CUDA ABI 检查，拒绝旧产物。

当前 CUDA 是逐 token 的正确性执行器，尚无 tensor-core 批量 GEMM。输入加输出上限 **2048 tokens**，逐层验证最多 128 输入。`--kv bf16` 保留用于复现未通过的目标缓存路径；不会把它标为已验收。
入口为独立实验 CLI `split-infer` / `split-worker`，尚未接入原有 TUI、通用调度器多请求队列或租约。每个进程/session 服务一个请求，失败后释放，不重试半个请求。

[固定性能目标](QWEN36-8BIT-TARGETS.md)要求 BF16 GEMM 输入及 KV、128K 输入、4096 输出和正式资源/性能测量，本轮没有替换或降低这些目标。下一步先处理目标精度的一致性，再实现 CUDA 批量 prefill、扩大上下文及正式双机基准。

## 协议 v2

双方握手用各自 32 字节随机 nonce、请求 UUID、固定 model revision 和 KV element width 派生 HMAC-SHA256 密钥，分别证明 client/server 身份。FP32/BF16 profile 不匹配会在握手失败。
帧为 92 字节 header + payload + 32 字节 HMAC；包含 magic、版本、请求 UUID、递增序号、类型、层、tensor、token 位置、token 数、字节数与 revision。

固定模型为 `mlx-community/Qwen3.6-35B-A3B-8bit` revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`，前段 `[0,20)` / 后段 `[20,40)`。所有数据 little-endian，layout 由版本、握手 profile 和 tensor descriptor 唯一确定：

|消息|dtype / shape / layout|
|---|---|
|Start|I32 `[tokens]`；含是否启用数值 trace|
|Hidden|F32 `[chunk,2048]`，层 20 输入|
|Trace（诊断）|F32 `[chunk,20,2048]`|
|Full attention State tensor 0/1|握手指定的 F32 或 BF16 K/V `[position,2,256]`|
|Linear State tensor 0|F32 Delta `[32,128,128]`，value head/value dim/key dim|
|Linear State tensor 1|F32 卷积 `[3,8192]`，按时间先后保存 raw projection|
|End|无 payload，完整 token 位置|

认证和完整性保护不提供加密，限受信任局域网。随机 secret 至少 32 字符，由启动脚本经 SSH stdin 传入子进程环境，不写命令参数或公开日志。
版本、request、revision、profile、序号、状态位置/大小、重复/缺失状态、篡改与断帧均拒绝。状态未齐不宣布 decode ready。

## 复现

实现仍在 `codex/split-prefill`，PR #8 尚未合并；仅从 main 检出不会包含这些入口。

1. 两端通过 GitHub 检出同一 SHA，确认工作树干净。Mac `dotnet build src/DeepInfer.Host -c Release`。
2. Windows 激活 `scripts/windows/environment.ps1`，执行 `scripts/windows/build-prefix.ps1`，随后 `dotnet build src/DeepInfer.Host -c Release`；需要恢复包时用仓库的官方 NuGet v2 配置。
3. Mac 导出：`python3 scripts/split/export-prefix.py models/qwen3.6-35b-a3b-8bit-mlx artifacts/split/prefix.bin`。将权重产物放入 Windows checkout 的 `artifacts/split/`；源码只经 GitHub。
4. bundle 为 18,415,842,388 bytes，SHA256 `ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77`。Windows 每次加载前验证完整散列。
5. Windows 算子检查：`python scripts/split/test-cuda.py`。
6. Mac：`python3 scripts/split/run-dual.py --verify --kv fp32 --max-tokens 8`。默认使用已有 SSH alias `infer-windows`；其他配置使用 `--ssh YOUR_CONFIGURED_ALIAS`。

AOT：分别 publish 到 Mac `artifacts/split/native-mac-final`（osx-arm64）和 Windows `artifacts/split/native-windows-final`（win-x64），再给统一脚本加 `--native`。Windows publish 使用 `-p:RestoreConfigFile=scripts/windows/NuGet.v2.config`。

脚本记录 source、产物散列、PID、stdout/stderr、退出码；每轮使用唯一目录。`--verify` 失败写结果并非零退出，移除此标志的请求不会被标为 Verified。
`HiddenReadyMs` 仅表示后段已计算；`DecodeReadyMs` 包含完整状态导入；`ElapsedMs` 直到生成结束。全部由 Mac 同一单调时钟记录。
