# CUDA 长 prefill 分块：1024 / 2048 与双机交接

日期：2026-10-23。关联 [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3)；实现分支 `codex/cuda-1024-prefill`，有线对照源码 `45a51a1669d43b401c806342cebf3f1647d80080`。本报告是 8192 输入 / 8 输出的诊断，不是 `QWEN36-8BIT-TARGETS.md` 的 128K 验收。

## 问题与改动

在固定的 affine 8bit 权重、FP32 KV、20/20 拆分下，扩大 Windows CUDA 的**计算**块是否能减少前段耗时？原生执行器现在接受 32、1024、2048 token；默认部署 `split-agent` 选择 1024，串行模式仍选 32。原生 attention 的 query 子块仍为 32，限制 score 工作区；1024/2048 是整个前段按层执行的外层块。每个外层块作为一个有界发送队列项，内部按 128-token hidden/KV 帧交给 Mac 的 128-token suffix。协议升至 split wire v5、CUDA ABI 7，两端必须同版。

## 数值差值先验分析

并行的 BF16 目标数值诊断见 [PR #29](https://github.com/LittleLittleCloud/Infer.NET/pull/29) 的 `docs/BF16-ROUTER-SHARED-20261023.md`。在 token 93、layer 1，Mac 与 CUDA 的路由 logits 相对 L2 只有 `6.67e-5`；第 8 名附近的 expert 167 与 54 顺序翻转（Mac 差 `+0.00172424`，CUDA 差 `−0.00018668`），layer 1 hidden 差值放大到相对 L2 `0.0376859`。因此异常大差值是近似并列专家的离散选择放大，而非长分块产生的随机结果。更早的算术差异尚未唯一定位；BF16 正确性门槛仍不通过，长分块不能替代该验收。

原生 Windows 独占测试使用固定 synthetic token IDs、相同 8bit 前段、20 层、40 份状态和完整 KV 范围。1024 token 的 32/1024 最终 hidden 与 40 份状态逐字节一致；正式三轮中位数分别 `1.107263 / 0.987767 s`，长块快 `10.8%`。2048 token 的 32/2048 亦逐字节一致，正式中位数 `2.256862 / 2.002092 s`，长块快 `11.3%`。各配置预热一次、正式三次。原始结果仅本地保存：`artifacts/dual-host/cuda-long-chunk-44d955b/1024.json` SHA-256 `59d5ac8cffa693d058a53330d0c68026972492eaad56ace04399ebfacefeddc4`；`2048.json` SHA-256 `ffa71e24d47700c2699eb0c2439206c1d42f459487fecb7401dd416136179d5a`。

## 完整双机 A/B

固定输入文件 SHA-256 `defa12f5be86a88aed8a25aa479ab1b17860a661ff364865708e121cac17206d`，实际模板后 8192 token；固定 8 个输出、忽略 EOS、greedy、0% prefix hit；每次重置状态，预热 1 次、正式 3 次。Mac 接 AC / High Power；Windows RTX 4090，Mac M5 Pro 48 GiB。两端固定同一源码，Windows 只切换 `--cuda-prefill-chunk`。Mac serve PID 在有线 32/1024/2048 三档期间不变。用户响应中的 8 个 token ID 在全部轮次一致；其列表 JSON 的 SHA-256 前缀为 `86ba289e0b2e1415`。实际 payload 每次 `267,812,864` bytes。Mac allocation 约 `37,467,340,800` bytes；Windows 运行中总设备显存快照约 `19,987 MiB`，峰值和 swap-out 未采集，不能据此判定资源门槛通过。

|链路/配置|prefill 三次，秒|中位 prefill，秒|中位 CUDA compute，秒|中位 Mac hidden wait，秒|结论|
|---|---|---:|---:|---:|---|
|Wi-Fi，经路由器，32，候选同版|12.225 / 13.073 / 11.567|12.225|9.734|1.031|对照|
|Wi-Fi，经路由器，1024，最终队列|12.820 / 12.594 / 12.620|12.620|9.091|1.584|CUDA 快，整段慢|
|Wi-Fi，经路由器，2048，最终队列|13.705 / 13.734 / 13.818|13.734|9.112|未单列|更慢|
|Mac 1GbE ↔ Windows 有线，32|11.296 / 11.761 / 11.526|11.526|9.728|0.612|有线对照|
|Mac 1GbE ↔ Windows 有线，1024|12.152 / 12.407 / 12.255|12.255|9.088|1.470|CUDA 快约 6.6%；整段慢约 6.3%|
|Mac 1GbE ↔ Windows 有线，2048|13.548 / 13.298 / 13.392|13.392|9.080|2.483|整段慢约 16.2%|

有线测试前确认 Mac `en7` 为 `1000baseT full-duplex`，数据监听与已建立的 agent TCP 连接都使用此接口地址；Mac 到 Windows 的路由亦走 `en7`。此前虽然插上网线，私有部署配置仍指向 Mac Wi-Fi 地址，所以早先各轮必须保留为 Wi-Fi 结果。两端配置已留存受限备份后改到有线地址；认证、证书与权重路径未变。现有数据无法把 `cudaSendMs` 全部归于链路：它包括对端计算引发的 socket 反压。1024/2048 的 Mac 等待随大计算块增大，表明流水线粒度也限制端到端收益。选择 1024 默认是按用户指定的 CUDA 计算目标；不宣称当前 1GbE 下端到端加速。

有线正式原始 JSON 仅本地：`e2e-wired32.json` SHA-256 `665b51270173871ffdce5de4d3d56ec850c4a8f643506c70a8725ef67c406373`，`e2e-wired1024.json` `7eba84783132c6838b0c44b0375a52e85897aaf84c24e425d567f1d4f391bdca`，`e2e-wired2048.json` `878e79b04633db2fd3f1cda06bd30a7bb01166e6689ae67731207d7281689dfb`。路径位于忽略的 `artifacts/dual-host/cuda-long-chunk-927a254/`，不是公开下载链接。

两端 managed build 均为 `45a51a1`；Mac `deepinfer.dll` SHA-256 `256dd325ca4eba12c94cbe019b4de2017a986f8d0670ed9b9309c411c5da86e4`、Metal dylib `1739ff5dd22a4b158f97d84d39268478a75d5683f4a9ad1a3cd0d1a8f79d4aef`；Windows `deepinfer.dll` `8fc82e57c246f66478e9e78cb308668a9dcb3a24ba261c6b166b55b844d7488a`、CUDA DLL `f0b938ac60d671c950217061d195e3d2f002742b87e8a21032b6ce492c51023c`。Mac 129 项测试通过；Windows 128 通过、1 项 Metal 专属测试跳过。

下一步是保留 32 对照，在更长输入与连续输出下验证容量、峰值内存、swap-out 与持续 decode；BF16 路由差异要继续从更早的算术点定位。128K 门槛、完整 logits 与长状态检查目前均未通过或未执行。

## 256/512 计算块的原生候选

有线 1024 块缩短了 CUDA 计算，却增加了 Mac 等待，因此提交 `f36e71bf1f49986e1efea949b4875dcb8fda9288` 增加可选的 256/512 块，默认 1024 不变。它们继续按 128-token hidden/KV 帧发送，权重、FP32 算术及状态布局不变。以下是 Windows SM89 上、8bit 权重/FP32 KV、mode 5 的**独占原生前段诊断**，每档与 32 对照交替运行，预热 1 次、正式 3 次；不包含网络、Mac 后段或客户端时延。

|合成输入|32 块正式中位|候选正式中位|前段改善|hidden 与 40 份最终状态|
|---|---:|---:|---:|---|
|256 tokens|0.271698 s|0.246584 s|约 9.24%|逐字节一致|
|512 tokens|0.547551 s|0.491736 s|约 10.19%|逐字节一致|

原始结果仅 Windows 本机保存于 `artifacts/dual-host/cuda-mid-{256,512}-f36e71b/result.json`；SHA-256 分别为 `d6b2e8bba5266003c55a840a0a5823700caf920ea1a8bf9b75163e664e67001b` 和 `e1d816efa56f42d4bc37c56948d1f15915c02f7950b4425b6e3b86df8e35901b`。Mac 129 项测试通过；Windows 128 项通过、1 项平台跳过，CUDA DLL 在 Windows 同源编译成功。

原生检查结束后先恢复原 `919259d` agent；Mac serve/coordinator 未换包。原 8192 输入/8 输出经真实 `cuda-metal-split` 请求返回 HTTP 200，8 个 token 的 SHA-256 为 `86ba289e0b2e1415b354ed6baaace9efb1f6b58e1083d3c6c46f605640623544`，与有线对照相同；serve 与 coordinator 重新报告 agent 已连接及 worker 健康。后续同版双机 A/B 见下节。

## 同版有线完整请求：中间块未改善端到端

两端从同一提交 `47fe588160fc748a8a3f0125b46308c3f03d52d3` 构建 Native AOT；打包清单的 split protocol 为 v5，权重 affine8，GEMM/KV 为 FP32。Mac serve 在五档测试期间保持同一 PID，Windows agent 每档重启并明确记录块长。链路为 Mac `en7` 1000baseT 全双工直达既有 LAN，Windows agent 实际连接该接口地址；Mac 到 Windows 的路由也走 `en7`。固定 8192 输入/8 输出、0% prefix hit、greedy、忽略 EOS，每次重置状态；每档一次完整预热、三次正式请求。顺序 256→512→32→1024→256 确认，Mac 在开始前确认 AC。正式试验未逐次记录电源、swap-out 或 GPU 峰值，不能据此通过固定资源门槛。

|CUDA 计算块|prefill 三次，秒|中位 prefill|中位 CUDA 计算|中位 Mac hidden 等待|中位 Mac 后段|
|---|---|---:|---:|---:|---:|
|32|10.972 / 10.936 / 11.071|10.972|9.764|0.411|10.290|
|256|11.275 / 11.281 / 11.266|11.275|9.130|0.703|10.289|
|512|11.551 / 11.480 / 11.474|11.480|9.077|0.913|10.292|
|1024|12.096 / 12.089 / 12.061|12.089|9.164|1.505|10.309|
|256 确认|11.303 / 11.290 / 11.266|11.290|9.192|0.719|10.291|

所有预热与正式轮次均返回同一 8-token ID 序列，SHA-256 `86ba289e0b2e1415b354ed6baaace9efb1f6b58e1083d3c6c46f605640623544`。各档正式 decode 中位约 55–56 token/s，但只有 8 个输出，不能代替长 decode 验收。32 是本轮最快的完整 prefill；256 比它慢约 2.8%，512 慢约 4.6%，1024 慢约 10.2%。更大块减少 CUDA 纯计算，却增加 Mac 等首批 hidden 的时间；Mac 后段约 10.3 秒几乎不变。CUDA `sendMs` 包含 socket 反压，不等于纯网络传输时间。三项阶段时长重叠，不能直接求和。

本机原始响应仅保存在 `artifacts/dual-host/midchunk-e2e-20261023T2030Z/{32,256,512,1024,256-confirm}.json`，对应 SHA-256 依次为 `e05a1ec5e78bd2b078aa83331689e982367e0d43d21c624bffa3eec138de08b1`、`1fb20715933291d49eb96a38c56513232dd9af500e7bcc6b7970e947d36a3dd5`、`686b41c4d9ccead4d53cefecf749009a441730fb3eed56131a317da2091daf89`、`a0ada66ed09d069cd99309857a3b33b462c678a6b1d6d592443620714a36bd58`、`2743a03e9e77a6bfbe98a4973c3a4dfa2105a6da9a99d3ad025b472cf8241765`。Windows 原生输入/状态文件仅本机可用，不提供为公开下载。试验结束后原 `919259d` Mac LaunchAgents 与 Windows agent 已恢复；coordinator 报告 worker healthy/Ready、agent connected，真实 8192/8 跨机请求 HTTP 200 且输出 ID 哈希保持一致。

此轮不选择 256/512 为默认；默认 1024 是先前用户指定的 CUDA 计算块，并非当前有线 8K 端到端最优。需要继续改流水线发送粒度或 Mac 后段，再在 32K/128K、BF16 与长输出上重测，不能把这个短请求的速度外推为固定目标通过。

## 待测：短首块 + 长后续块

现有 1024 块要等前 1024 token 的 CUDA 前 20 层全部完成，Mac 才能拿到第一帧 hidden。为检验流水线启动等待是否是主要损失，新增可选 `cuda-prefill-first-chunk=128`：先计算并发送 128 token，此后仍用配置的 1024/2048 计算块。默认 `0` 保留原行为，验证模式仍强制 32-token 块。它不改变模型、FP32 算术、原生 ABI 或 split wire 格式；发送的 hidden/KV 帧仍不超过 128 token。测试前不把更早的首帧外推为完整请求加速。

`scripts/split/check-cuda-long-chunk.py --chunk 1024 --tokens 8192 --first-chunk 128` 可独占 GPU 对照 32、1024 和短首块三种原生执行序列，逐字节比较最终 hidden 与 40 份状态。通过后再做同源双机 A/B，保留完整响应、Mac 等待和 CUDA 计算时间及换页/显存证据；未通过前不改默认部署。
