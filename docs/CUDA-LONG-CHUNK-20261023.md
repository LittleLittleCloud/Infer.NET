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
