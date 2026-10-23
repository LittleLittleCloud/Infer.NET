# BF16 前段跨后端精度对齐：路由与共享专家诊断

2026-10-23；关联 [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3)、[PR #29](https://github.com/LittleLittleCloud/Infer.NET/pull/29)。问题：CUDA 模式 6 与 Metal 的 128-token 前段 trace 在第 0 层已有明显差异，是否由路由及共享专家投影的操作数精度不一致造成？本实验只比较两个单机诊断产物，**不是**产品跨机传输、完整模型或吞吐验收。

固定模型为 Qwen3.6-35B-A3B affine8/group64，revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`；CUDA 前 20 层 bundle SHA-256 `ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77`。输入为 128 个合成有效 token ID，SHA-256 `8894963a38fa5ac5efc5991b5eb193536d712234d434a83cf8dc3cc47072f0e7`。RTX 4090 SM89 上 CUDA 每块 32 token；M5 Pro Metal 一块 128 token。两端 KV 为 BF16，权重保持原 affine8；CUDA 先跑线上 FP32 模式 5，再跑 BF16 诊断模式 6 两次，检查后者有限且确定。每次采集 20 层 trace 和 40 份 KV/Delta/卷积状态。两端原线上服务在独占 GPU 试验期间停用，结束后恢复；本次没有做真实 RPC、性能预热/正式轮次、供电或链路吞吐测量。

| 跨后端 128-token 配置 | 首个失败：第 0 层 relative L2 | 第 0 层 max abs | 通过项 / 60 | 最差状态 relative L2 |
|---|---:|---:|---:|---:|
| 原 BF16 路由与共享投影 | 0.00483155 | 0.00481211 | 2 | 0.08883945 |
| 两端 FP32 路由，其余原样 | 0.00043722 | 0.00046915 | 2 | 0.06211780 |
| 两端 FP32 路由，CUDA 共享 gate/up/route 改用 FP32 诊断 | 0.00016054 | 0.00005180 | 2 | 0.07115700 |

路由精度对齐后，之前 token 49/91 的第八专家分别在两端选择同一 ID `137`/`241`；Metal 与 CUDA 的概率仅差约 `1e-6`。进一步对齐共享专家投影后，第 0 层 relative L2 较路由单独试验再降约 63%。但第 1 层 token 93 的相对误差约 `0.0377`，第 3 层 token 15 约 `0.0537`；错误继续传播，最终 58 项不符合固定 `relative L2 < 1e-4` 与 `max absolute < 0.01` 门槛。第 0 层即使消除所知路由分歧，也仍超过相对误差门槛。共享 FP32 改动没有使最差状态更好，不能把局部改善写成整体正确性通过。

Metal 参考源码 `522fa9b87e1848761c8461085f131eaa25d7d342`，Metal dylib SHA-256 `a02b1eb1f59d4bb4b6cd769e1aa029f9908edaa015d57b8abafb749c1dd09132`；CUDA 共享投影试验源码 `a0e3daf18cd80191ef0a775f61377d619bbcc6d6`，CUDA DLL SHA-256 `176da3441475fda9d67b542f5a7781fbad4918ce342afac9ca7d22399d299bd6`。最后一次 Metal/CUDA trace SHA-256 为 `7e820c3bf90f866c2efe879f1c548072ec5748ef282ac61d690c46d1a12749ec` / `fc52acaa68f9205277124f8871bfec417277e5425961755ce327c34198800582`。完整 60 项比较 JSON 仅本机留存，SHA-256 `59d3f284df3773c26b8e49d0646f2a814a7c56a98e016e5e2713fa95cd91a661`；原始 trace/状态也仅本机可用，未公开为下载链接。

复现：在分别停止已核对的线上 Mac serve 与 Windows agent 后，Mac 使用 `DEEPINFER_METAL_ROUTER_FP32=1 DEEPINFER_METAL_KV=bf16 DEEPINFER_METAL_PREFILL_GEMM=bf16-nax DEEPINFER_METAL_PREFILL_CHUNK=128 DEEPINFER_METAL_PREFILL_MATRIX=1 DEEPINFER_METAL_PREFILL_DELTA=sequential` 运行 `deepinfer split-check-bf16-prefix --weights MODEL_DIR --output NEW_DIRECTORY`。Windows 独立 checkout 在相同源码提交上执行 `scripts/windows/build-prefix.ps1`，然后运行 `scripts/windows/check-bf16-prefix-profile.ps1 -Config PRIVATE_WORKER_JSON -Output NEW_DIRECTORY -Router fp32 -Shared fp32`。最后用 `scripts/split/compare-bf16-prefix.py --metal METAL_DIRECTORY --cuda CUDA_DIRECTORY --output NEW_JSON` 比较；非零退出表示固定门槛未过。下一步需定位 token 93 在第 1 层的首个路由或算术分歧，并将第 0 层基础误差压到门槛内，再验证完整 logits、greedy 和跨机状态交接。线上 8bit/FP32 KV 路径保持原版本，BF16 候选未部署。
