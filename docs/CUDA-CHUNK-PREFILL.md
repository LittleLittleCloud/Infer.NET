# CUDA 批量 prefill：8K 双机 A/B

2026-10-23，关联 Issue #3。固定 Qwen3.6 35B A3B affine8/group64，revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`。

**8K 输入、128 固定输出：worker prefill 从 52.503 秒降至 18.642 秒，2.82 倍；客户端整次请求从 54.782 秒降至 20.926 秒。** 这是 FP32 KV / FP32 乘法的部署诊断路径，尚未通过 BF16、128K 或 4096 输出正式门槛。

## 改动与数值验证

CUDA 原路径按 token 执行全部 20 层。新路径在每个 32-token 块内按层执行批量投影、卷积、attention、路由与专家计算；DeltaNet 把状态行留在寄存器中完成整块递推，减少重复读写与 kernel 启动。投影/专家仍保留原 FP32 GEMV 累加顺序。`cuda-prefill=serial|batch-fp32` 可同二进制切换，默认 batch-fp32。

另有自行实现的 SM89 affine8 → BF16 Tensor Core GEMM 原语，**不在本轮部署性能数据中**。独立 BF16 操作数对照：32/128/512-token 大矩阵约 3.0/10.1/14.4 倍；Delta 原语约 7–8 倍。这些算子比值不能当作完整模型收益。

完整固定前段原生测试使用 131 个合成有效 token ID、1/7/17/32 非整齐分块，比较所有 20 层 trace、hidden、40 个 KV/Delta/卷积状态张量。FP32 与 BF16 KV 两档均有限且逐字节相同。512-token 前段 warmup 1 + 正式 3 次：FP32 KV 2.628705 → 0.764376 秒（3.44 倍）；BF16 KV 2.617131 → 0.765083 秒（3.42 倍）。这是同 CUDA 两条执行路径对照，**不解决 CUDA 与 Metal 的 BF16 跨主机误差门槛**。

## 双机测量

两端同一 source `cc19a65056f351e201f65901d285490ee24b11b1`、Native AOT / CUDA ABI 5，Mac M5 Pro 20 核 / 48 GiB + Windows RTX 4090。Mac AC、`pmset powermode=2`；既有 Mac Wi-Fi / Windows 有线 LAN，TLS 数据通道，无 SSH tunnel。只切 Windows 执行策略，Mac serve PID 和包不变。32-token wire chunk，无 prefix cache，每次 reset；输出忽略 EOS，greedy。

固定 8192-token 输入 SHA256：`defa12f5be86a88aed8a25aa479ab1b17860a661ff364865708e121cac17206d`。

|模式|预热 prefill 秒|正式 prefill 秒（三次）|中位 decode token/s|客户端整次中位秒|客户端首 token 中位秒|
|---|---:|---|---:|---:|---:|
|serial|52.711|52.573 / 52.482 / 52.503|55.97|54.782|52.513|
|batch-fp32|18.305|18.592 / 18.642 / 18.871|55.83|20.926|18.650|

8 次请求全部 128 个输出 token ID 一致。每次新增整机 swap-out 为 0；Metal allocation 为 37,995,495,424 字节（35.39 GiB）。Windows GPU 以 500ms 采样，峰值见公开 JSON；采样不代表逐分配事件的精确峰值。未在此组采集进程 footprint，不能声明正式资源验收完整通过。

串行先测、批量后测，没有随机顺序；两组均完整预热。一次切换因普通账户 WMI 权限失败，误标为 batch 的尝试已中止并排除，失败记录保留。没有挑选最快一轮。

## 剩余瓶颈

正式三次阶段中位约为：

|阶段|serial|batch-fp32|
|---|---:|---:|
|等待前段 hidden|31.77 s|0.43 s|
|Mac 后 20 层计算|17.77 s|15.01 s|
|最终状态接收等待|2.91 s|3.22 s|
|状态导入|0.019 s|0.016 s|

接收端等待是流水线时间，不等于 CUDA 纯计算；各阶段中位相加也不替代整次中位。两组 Mac 计算相差约 2.8 秒仍需进一步分解，不能全部归因于 CUDA kernel 本身。前段加速后，Mac 后段和 200,704,000 字节最终状态交接成为当前可见瓶颈。

本轮报告 worker prefill 阶段与客户端 TTFT；规范所要求的 coordinator t2−t1 尚需完整打点。当前约 439 token/s 是 8K FP32 路径，不是 128K 目标成绩，也不能与历史 Mac BF16 2363 token/s 直接当作同配置对照。

## 复现与证据

- `scripts/windows/benchmark-prefill-ops.ps1`：独立原语；不加载完整模型。
- `scripts/split/benchmark-prefix-chunk.py`：停止已核对的 CUDA agent 后，执行完整前段数值与时间对照；同 GPU 不可加载第二份模型。
- 双机使用同一发布包，分别传入 `--cuda-prefill serial` / `--cuda-prefill batch-fp32`，实际 ready 日志确认策略后，对固定输入发送 `/v1/infer/stream`，`maxTokens=128, benchmarkIgnoreEos=true, profile=true`。每档 warmup 1 + 正式 3 次，记录 SSE 时间、token IDs、vm_stat 与 GPU 采样。
- 公开逐次摘要：`site/cuda-chunk-summary.json`。
- 本地原始证据（未公开）：`artifacts/dual-host/chunk-e2e-serial-20261023T075202Z`、`chunk-e2e-batch-fp32-20261023T075943Z`、`cuda-prefill-chunk-387511b`；排除尝试 `chunk-e2e-batch-fp32-20261023T075701Z/INVALID.json`。
- 107 项 Mac 单测、31 项部署前集成检查通过；随后增加 503/504 转发回归，34 项集成检查通过。原生前段及独立算子检查如上。两端 AOT 包已验证并部署 cc19a65；后续 HTTP 错误映射修复尚未部署到该包。

固定目标仍是 131072 输入 prefill ≥1600 再到2000 token/s，4096 输出 decode ≥37 token/s，整次请求 ≤200 再到180秒。下一步优化 Mac 后段块利用率、状态交接重叠，并完成 BF16 跨主机数值验证；不降量化位宽、不放宽数值门槛。
