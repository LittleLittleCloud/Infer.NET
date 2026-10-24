# CUDA Long Prefill Chunking: 1024 / 2048 and Dual-Node Handover

Date: 2026-10-23. Related to [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3); implementation branch `codex/cuda-1024-prefill`, wired comparison source code `45a51a1669d43b401c806342cebf3f1647d80080`. This report is a diagnosis of 8192 input / 8 output, not a `QWEN36-8BIT-TARGETS.md` 128K acceptance test.

## Issues and Changes

Under fixed affine 8bit weights, FP32 KV, and 20/20 splitting, can enlarging the **compute** block in Windows CUDA reduce front-end latency? The native executor now accepts 32, 1024, 2048 tokens; default deployment selects `split-agent` choosing 1024, while serial mode still selects 32. The query sub-blocks of native attention remain 32, limiting the score workspace; 1024/2048 is the outer block executing layer-by-layer across the entire front-end. Each outer block serves as a bounded send queue item, internally passing 128-token hidden/KV frames to the Mac's 128-token suffix. The protocol upgrades to split wire v5 and CUDA ABI 7; both ends must use the same version.

## Numerical Difference Prior Analysis

Parallel BF16 target numerical diagnostics are documented in the [BF16 routing log of PR #29](https://github.com/LittleLittleCloud/Infer.NET/blob/codex/bf16-target/docs/BF16-ROUTER-SHARED-20261023.md). At token 93, layer 1, the relative L2 difference between Mac and CUDA routing logits is only `6.67e-5`; the expert 167 and 54 order near the 8 rank is flipped (Mac diff `+0.00172424`, CUDA diff `−0.00018668`), and layer 1 hidden differences amplify to relative L2 `0.0376859`. Therefore, the abnormally large difference is an amplification of discrete selection among approximately tied experts, not a random result from long chunking. Earlier arithmetic discrepancies have not yet been uniquely located; BF16 correctness threshold is still not met, so long chunking cannot replace this acceptance test.

Native Windows exclusive testing uses fixed synthetic token IDs, the same 8bit front-end, 20 layers, 40 state partitions, and full KV range. The 32/1024 final hidden states for 1024 tokens and 40 state partitions are byte-for-byte identical; median values in the formal three rounds are `1.107263 / 0.987767 s` respectively, with long blocks being `10.8%` faster. The 32/2048 for 2048 tokens are also byte-for-byte identical, with formal medians `2.256862 / 2.002092 s`, and long blocks `11.3%` faster. Each configuration was warmed up once and run formally three times. Raw results are saved locally only: `artifacts/dual-host/cuda-long-chunk-44d955b/1024.json` SHA-256 `59d5ac8cffa693d058a53330d0c68026972492eaad56ace04399ebfacefeddc4`; `2048.json` SHA-256 `ffa71e24d47700c2699eb0c2439206c1d42f459487fecb7401dd416136179d5a`.

## Complete Dual-Node A/B

Fixed input file SHA-256 `defa12f5be86a88aed8a25aa479ab1b17860a661ff364865708e121cac17206d`, actual template post-8192 tokens; fixed 8 outputs, ignoring EOS, greedy, 0% prefix hit; reset state each time, warm up 1 times, run formally 3 times. Mac connected to AC / High Power; Windows RTX 4090, Mac M5 Pro 48 GiB. Both ends fixed to the same source code, Windows only switches `--cuda-prefill-chunk`. Mac serve PID remains unchanged during the wired 32/1024/2048 three-tier period. The 8 token IDs in user responses are consistent across all rounds; the SHA-256 prefix of their list JSON is `86ba289e0b2e1415`. Actual payload is `267,812,864` bytes each time. Mac allocation is approximately `37,467,340,800` bytes; Windows total device VRAM snapshot during runtime is approximately `19,987 MiB`, with peak and swap-out not collected, so resource threshold pass cannot be determined based on this alone.

|Link/Config|prefill 3 times, s|Median prefill, s|Median CUDA compute, s|Median Mac hidden wait, s|Conclusion|
|---|---|---:|---:|---:|---|
|Wi-Fi, via router, 32, candidate same version|12.225 / 13.073 / 11.567|12.225|9.734|1.031|Control|
|Wi-Fi, via router, 1024, final queue|12.820 / 12.594 / 12.620|12.620|9.091|1.584|CUDA fast, whole segment slow|
|Wi-Fi, via router, 2048, final queue|13.705 / 13.734 / 13.818|13.734|9.112|Not listed separately|Slower|
|Mac 1GbE ↔ Windows wired, 32|11.296 / 11.761 / 11.526|11.526|9.728|0.612|Wired control|
|Mac 1GbE ↔ Windows wired, 1024|12.152 / 12.407 / 12.255|12.255|9.088|1.470|CUDA fast by approx. 6.6%; whole segment slow by approx. 6.3%|
|Mac 1GbE ↔ Windows wired, 2048|13.548 / 13.298 / 13.392|13.392|9.080|2.483|Whole segment slow by approx. 16.2%|

Before the wired test, it was confirmed that Mac `en7` is `1000baseT full-duplex`; both data listening and the established agent TCP connection use this interface address; the route from Mac to Windows also goes through `en7`. Although the network cable was plugged in earlier, the private deployment configuration still pointed to the Mac Wi-Fi address, so all previous rounds had to remain as Wi-Fi results. After saving restricted backups of the configurations on both ends, they were changed to the wired address; authentication, certificates, and weight paths remained unchanged. The current data cannot attribute all of `cudaSendMs` to the link: it includes socket backpressure caused by computation on the peer side. The Mac wait time for 1024/2048 increases with larger computation blocks, indicating that pipeline granularity also limits end-to-end gains. The choice of 1024 is based on the user-specified CUDA computation target; it does not claim end-to-end acceleration for the current 1GbE.

The official raw JSON for the wired test is only saved locally: `e2e-wired32.json` SHA-256 `665b51270173871ffdce5de4d3d56ec850c4a8f643506c70a8725ef67c406373`, `e2e-wired1024.json` `7eba84783132c6838b0c44b0375a52e85897aaf84c24e425d567f1d4f391bdca`, `e2e-wired2048.json` `878e79b04633db2fd3f1cda06bd30a7bb01166e6689ae67731207d7281689dfb`. The path is located in the ignored `artifacts/dual-host/cuda-long-chunk-927a254/`, not a public download link.

Both ends' managed builds are `45a51a1`; Mac `deepinfer.dll` SHA-256 `256dd325ca4eba12c94cbe019b4de2017a986f8d0670ed9b9309c411c5da86e4`, Metal dylib `1739ff5dd22a4b158f97d84d39268478a75d5683f4a9ad1a3cd0d1a8f79d4aef`; Windows `deepinfer.dll` `8fc82e57c246f66478e9e78cb308668a9dcb3a24ba261c6b166b55b844d7488a`, CUDA DLL `f0b938ac60d671c950217061d195e3d2f002742b87e8a21032b6ce492c51023c`. Mac 129 items passed tests; Windows 128 passed, 1 items (Metal-specific tests) were skipped.

The next step is to retain the 32 control and verify capacity, peak memory, swap-out, and continuous decode under longer inputs and continuous outputs; BF16 routing differences need to be located from earlier arithmetic points. The 128K threshold, complete logits, and long state checks have currently not passed or have not been executed.

## 256/512 Native Candidates for Computation Blocks

The wired 1024 block shortened CUDA computation but increased Mac wait time, so `f36e71bf1f49986e1efea949b4875dcb8fda9288` was submitted to add optional 256/512 blocks, with the default 1024 unchanged. They continue to send hidden/KV frames according to 128-token, with weights, FP32 arithmetic, and state layout unchanged. Below is the **exclusive native front-end diagnosis** on Windows SM89, with 8bit weights /FP32 KV, mode 5, alternating with 32 for comparison, preheated 1 times, and officially run 3 times; it does not include network, Mac back-end, or client latency.

|Synthetic Input|32 Block Official Median|Candidate Official Median|Front-end Improvement|Final State of Hidden and 40 Shares|
|---|---:|---:|---|---|
|256 tokens|0.271698 s|0.246584 s|Approx. 9.24%|Byte-by-byte consistent|
|512 tokens|0.547551 s|0.491736 s|Approx. 10.19%|Byte-by-byte consistent|

Raw results are only saved locally on Windows in `artifacts/dual-host/cuda-mid-{256,512}-f36e71b/result.json`; SHA-256 are `d6b2e8bba5266003c55a840a0a5823700caf920ea1a8bf9b75163e664e67001b` and `e1d816efa56f42d4bc37c56948d1f15915c02f7950b4425b6e3b86df8e35901b` respectively. Mac 129 items passed tests; Windows 128 items passed, 1 items skipped by platform, CUDA DLL compiled successfully on Windows with the same source.

After the native check, the original `919259d` agent was restored; the Mac serve/coordinator package was not changed. The original 8192 input and /8 output were returned via a real `cuda-metal-split` request with HTTP 200, the 8 token SHA-256 was `86ba289e0b2e1415b354ed6baaace9efb1f6b58e1083d3c6c46f605640623544`, consistent with the wired baseline; serve and coordinator re-reported that the agent was connected and the worker was healthy. See the next section for subsequent dual-machine A/B testing with the same version.

## Complete wired request with the same version: intermediate blocks did not improve end-to-end

Both ends were built from the same commit `47fe588160fc748a8a3f0125b46308c3f03d52d3` for Native AOT; the split protocol in the package manifest is v5, weight affine8, GEMM/KV is FP32. The Mac serve maintained the same PID throughout the five test tiers, while the Windows agent restarted for each tier and explicitly logged the block length. The link is Mac `en7` 1000baseT full-duplex direct to the existing LAN, and the Windows agent actually connects to this interface address; routing from Mac to Windows also goes through `en7`. Fixed 8192 input, /8 output, 0% prefix hit, greedy, ignoring EOS, resetting state each time; one full warm-up and three formal requests per tier. Sequential 256→512→32→1024→256 confirmed, with Mac confirming AC before starting. The formal test did not record power, swap-out, or GPU peaks per iteration, so it cannot pass the fixed resource threshold based on this.

|CUDA Compute Block|prefill three times, seconds|Median prefill|Median CUDA Compute|Median Mac Hidden Wait|Median Mac Post-segment|
|---|---|---:|---:|---:|---:|
|32|10.972 / 10.936 / 11.071|10.972|9.764|0.411|10.290|
|256|11.275 / 11.281 / 11.266|11.275|9.130|0.703|10.289|
|512|11.551 / 11.480 / 11.474|11.480|9.077|0.913|10.292|
|1024|12.096 / 12.089 / 12.061|12.089|9.164|1.505|10.309|
|256 Confirmed|11.303 / 11.290 / 11.266|11.290|9.192|0.719|10.291|

All warm-up and formal rounds returned the same 8-token ID sequence, SHA-256 `86ba289e0b2e1415b354ed6baaace9efb1f6b58e1083d3c6c46f605640623544`. The median decode speed for each tier's formal run was approximately 55–56 token/s, but with only 8 outputs, it cannot replace long decode acceptance. 32 is the fastest complete prefill in this round; 256 is slower by about 2.8%, 512 slower by about 4.6%, 1024 slower by about 10.2%. Larger blocks reduce pure CUDA compute but increase the time for the first batch of hidden states on Mac; the Mac post-segment is almost unchanged at about 10.3 seconds. CUDA `sendMs` includes socket backpressure and does not equal pure network transmission time. The durations of the three stages overlap and cannot be directly summed.

The original response on this machine is only saved in `artifacts/dual-host/midchunk-e2e-20261023T2030Z/{32,256,512,1024,256-confirm}.json`, corresponding to SHA-256 in sequence as `e05a1ec5e78bd2b078aa83331689e982367e0d43d21c624bffa3eec138de08b1`, `1fb20715933291d49eb96a38c56513232dd9af500e7bcc6b7970e947d36a3dd5`, `686b41c4d9ccead4d53cefecf749009a441730fb3eed56131a317da2091daf89`, `a0ada66ed09d069cd99309857a3b33b462c678a6b1d6d592443620714a36bd58`, `2743a03e9e77a6bfbe98a4973c3a4dfa2105a6da9a99d3ad025b472cf8241765`. Windows native input/status files are only available locally and are not provided for public download. After the test, the original `919259d` Mac LaunchAgents and Windows agent were restored; the coordinator reported worker healthy/Ready, agent connected, real 8192/8 cross-machine request HTTP 200, and output ID hashes remained consistent.

256/512 is not selected as the default in this round; the default 1024 is the CUDA compute block previously specified by the user, not the current wired 8K end-to-end optimum. Further adjustments to the pipeline send granularity or the Mac post-segment are needed, followed by re-testing on 32K/128K, BF16, and long outputs. The speed of this short request cannot be extrapolated as a fixed target pass.
