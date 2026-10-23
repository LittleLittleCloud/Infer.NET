# 4090 前 20 层同精度 FP32 对照，2026-10-23

问题：先前 CUDA 前段与 Mac Metal 对照，在第 3 层（第一个全注意力层）开始出现 `4.086e-4` 相对 L2 差值。那一轮 CUDA mode 5 虽用 FP32 GEMM，却仍用 BF16 KV；Mac 用 FP32 GEMM/KV，不能将差值归因于设备实现。旧观察及更正保留在[此前实验](https://github.com/LittleLittleCloud/Infer.NET/blob/b023318f821b94e28c1cd1a1551b6e6f37c95d91/docs/BF16-STAGE-PROBE-20261023.md)。本轮将两端的 GEMM 和 KV 都设为 FP32，隔离这个变量。

两端使用完整源码 `1f42e6fab4d58a683c66a3d009a8465521ad4403`、Qwen3.6-35B-A3B 固定 affine8/group64 权重、同一组 128 个合成 token ID（SHA-256 `8894963a38fa5ac5efc5991b5eb193536d712234d434a83cf8dc3cc47072f0e7`）。Windows RTX 4090 采集 CUDA mode 5 的 FP32 GEMM/FP32 KV 和 mode 6 的 BF16 GEMM/FP32 KV；两个模式的路由及共享专家投影均为诊断 FP32。Mac M5 Pro 采集 FP32 GEMM/FP32 KV，128-token 外层块、关闭 matrix 路径、顺序 DeltaNet。CUDA 前段包 SHA-256 为 `ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77`，Windows CUDA DLL SHA-256 为 `4128dd0a48fa25481733f7cbe8851264e006765da1e7af7c233dd398661f3edd`。

| CUDA 与 Metal FP32 对照 | 通过层输出 | 通过状态 | 最大层 relative L2 | 最大状态 relative L2 |
| --- | ---: | ---: | ---: | ---: |
| FP32 GEMM / FP32 KV | 20/20 | 40/40 | `8.53546e-7` | `3.40833e-6` |
| BF16 GEMM / FP32 KV | 0/20 | 0/40 | `0.0733243` | `0.103492` |

同精度 FP32 控制的最大状态绝对差为 `5.43594e-5`；全部低于本项目诊断阈值 relative L2 `<1e-4`、max absolute `<0.01`。与旧轮相比，同精度控制消除了第 3 层差值，支持旧差值来自 KV dtype 混用。保持 KV 为 FP32 时，BF16 GEMM 候选仍在第 0 层失败：relative L2 `0.009913728`，max absolute `0.00564497`。因此 BF16 GEMM 的首层差值不能仅用 KV 格式解释。此结论只针对隔离的 128-token 前段采样；Metal FP32 是设备对照，不是固定的独立整模 FP32 logits 参考。

Windows 结果 JSON SHA-256 `c1a5ef1e419ab2ac52ed76ce96d08841c9ebb07cd367984ea4f8bb4d6a289cac`；Mac 结果 `a20eadb1555ca920922827a84c6c9829af53111323bba09f2e6c1c624a525883`；离线比较 `3c67f4113e8622324f53675885e86c787a352c2e7039d49122c67a994068a08e`。大型原始轨迹与状态仅留开发机，不是公开下载。原有 `919259d` Mac serve 与 Windows agent 在隔离检查后均恢复，协调器显示 Ready、agent 已连接。Mac Release 构建通过；更早的 125 项 Mac 与 124 项 Windows 测试属于上一源码检查点，不能当成本轮新测试。

复现时让原有 GPU 服务空闲并按部署 SOP 临时暂停，采集后恢复；每轮使用新输出目录：

```text
Windows: powershell -File scripts/windows/check-bf16-prefix-profile.ps1 -Config PRIVATE_WORKER_JSON -Output NEW_CUDA_DIRECTORY -Kv fp32 -Router fp32 -Shared fp32 -RouteProbe 999:999
Mac: DEEPINFER_METAL_KV=fp32 DEEPINFER_METAL_PREFILL_GEMM=fp32 DEEPINFER_METAL_PREFILL_CHUNK=128 DEEPINFER_METAL_PREFILL_MATRIX=0 DEEPINFER_METAL_PREFILL_DELTA=sequential deepinfer split-check-fp32-prefix --weights MODEL_DIR --output NEW_METAL_DIRECTORY
Compare: python3 scripts/split/compare-bf16-prefix.py --metal NEW_METAL_DIRECTORY --cuda NEW_CUDA_DIRECTORY --metal-fp32 --output NEW_REPORT.json
```

探针读回改变耗时，不能由这组秒数推断实际 prefill 吞吐。本轮没有跨机传输、后 20 层、logits、greedy 输出、128K、内存或正式性能验收。下一步是用固定独立整模 FP32 参考对 BF16 层输出、状态和 logits，定位 MoE 路由舍入，并在数值通过后测真实 RPC 和 128K 门槛。关联 [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3) 与未合并的 [PR #31](https://github.com/LittleLittleCloud/Infer.NET/pull/31)。
