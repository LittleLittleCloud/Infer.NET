# KV / recurrent-state synchronization baseline

Measured on **2026-10-22 (America/Los_Angeles)**, source commit `75afbb2ed240ca55fd8aedd00fdb2420dfefd991`, clean on both hosts. Run ID: `kv-sync-20261022-run1`. **40/40 transfers passed** (10 warmups, 30 formal trials). Reusable tooling is in [PR #7](https://github.com/LittleLittleCloud/Infer.NET/pull/7), stacked on the Windows setup PR #6.

## Measured GPU state synchronization

Mac: M5 Pro, AC / High Power, 5 GHz Wi-Fi (802.11ax, 80 MHz); Windows: RTX 4090, driver 610.88, physical 1 Gbps Ethernet. Mac Wi-Fi reported 432 Mbps during a snapshot, RSSI -58 dBm / noise -94 dBm; this reported rate is not a sustained bidirectional throughput guarantee. No model inference was started by this benchmark; existing idle Mac inference/TUI services were preserved.

All GPU transfers use 4 MiB chunks. Timings include receiver GPU validation and final acknowledgment. MB/s uses decimal bytes; MiB/GiB use powers of two.

| State payload | Direction | Three formal times (s) | Median (s) | Median MB/s | CUDA copy median (s) |
| --- | --- | --- | ---: | ---: | ---: |
| 128K first-20 state, 1.281 GiB | cuda-to-metal | 24.659 / 21.489 / 25.933 | 24.659 | 55.77 | 0.5086 |
| 128K first-20 state, 1.281 GiB | metal-to-cuda | 38.317 / 41.203 / 41.995 | 41.203 | 33.37 | 0.5254 |
| 90% hit suffix + state, 159.414 MiB | cuda-to-metal | 2.512 / 2.469 / 2.430 | 2.469 | 67.70 | 0.0616 |
| 90% hit suffix + state, 159.414 MiB | metal-to-cuda | 4.379 / 4.673 / 4.501 | 4.501 | 37.14 | 0.0636 |

For complete state, receiver verification took a median **43.2 ms** on Metal and **48.4 ms** on CUDA. The tiny reported Metal “staging” duration is Python/native call overhead: shared buffers require no separate GPU upload/download copy. CUDA copy durations include actual device↔pinned-memory transfers and API/driver overhead.

The complete forward transfer's median is **24.659 s**, versus about **0.509 s** for CUDA readback. Reverse transfer is **41.203 s**, versus about **0.525 s** for CUDA upload. Thus GPU copy optimization alone has limited benefit for this particular serial LAN path. Different GPUs, loaded-model memory pressure, overlapping compute, encryption or a faster network require fresh measurements.

## Host-only controls and variability

128 MiB host buffers, one warmup plus three trials per configuration; no GPU copies.

| Chunk | Direction | Median MB/s | Formal elapsed range (s) |
| --- | --- | ---: | --- |
| 256 KiB | cuda-to-metal | 40.48 | 3.307–4.704 |
| 256 KiB | metal-to-cuda | 28.48 | 4.156–5.012 |
| 4096 KiB | cuda-to-metal | 61.45 | 2.070–2.384 |
| 4096 KiB | metal-to-cuda | 24.30 | 3.829–9.761 |
| 16384 KiB | cuda-to-metal | 61.20 | 2.179–2.234 |
| 16384 KiB | metal-to-cuda | 34.13 | 3.736–4.073 |

The sequential sweep was visibly variable, including a 9.761 s reverse host transfer. All samples are retained. It does not establish a causal 4 MiB/16 MiB optimization gain; simultaneous radio conditions were not controlled. The 4 MiB state results above are the observed baseline, not a network upper bound.

## Consequence for the pipeline estimate

The earlier **110 MB/s** figure was an assumption for an effective wired-gigabit path. The measured complete-state forward median here is **55.77 MB/s**. At that same rate, the separate 1 GiB hidden stream plus 1.281 GiB state would occupy about **43.91 s** of link service, compared with the earlier assumed 22.26 s. **43.91 s is a byte/rate projection, not a measured simultaneous pipeline or prefill time.** Hidden/state traffic can overlap computation and contend with each other; the final tail still matters.

KV alone is 1.25 GiB; scaling at the complete-state median rate gives approximately **24.07 s**, while the measured full handoff including Delta/conv state is **24.66 s**. The pure-KV number is calculated, not a separately measured trial.

A useful next controlled comparison is Mac wired Ethernet with the same harness; no adapters were installed and no 10GbE performance is claimed. Functional model state export/import and continuation remain Issue #3 work.

## Method and scope

This is a transport microbenchmark for the existing Qwen3.6-35B-A3B 8bit plan, not a CUDA model implementation. The 8bit weight precision does not change the BF16 KV / FP32 recurrent-state byte counts. Payloads are synthetic reproducible bit patterns, not numerically meaningful model states.

Geometry is taken from the pinned model configuration and current Metal state layout. For the first 20 layers, 5 full-attention layers use 2 KV heads and head dimension 256. Each token occupies 5 × 2 (K/V) × 2 heads × 256 × 2 bytes = 10,240 bytes. At 131,072 tokens this is 1,342,177,280 bytes (1.25 GiB). The other 15 layers each have 32 × 128 × 128 FP32 Delta state plus 3 × 8192 FP32 convolution history, totaling 32,931,840 bytes (31.40625 MiB). Full state is 1,375,109,120 bytes.

The 90% prefix-hit surrogate transfers 13,108 suffix tokens, rounded up, plus all final recurrent state: 167,157,760 bytes. This assumes the exact reusable prefix and appropriate recurrent snapshots are already present on both devices. It does not implement prefix caching or arbitrary recurrent rollback.

The measured path is CUDA allocation → pinned host staging → direct TCP → shared Metal MTLBuffer (and the reverse). Metal shared memory matches current model buffers and needs no separate CPU-to-GPU upload; a receiver GPU kernel reads and verifies every uint32 word before completion. CUDA uses synchronous staging copies, with no pipelining/double buffering. No model weights are allocated by the probe, no KV layout conversion or model-session import is performed.

Each direction/configuration has one warmup and three formal trials. Whole-run timing uses the Mac monotonic clock from GO to receiver verification and final acknowledgment. Preparation/allocation/pattern generation are excluded. Receiver validation is included and separately reported. Source and destination phase clocks are not subtracted across machines. Per-copy CUDA wall time is accumulated separately; it includes API/driver overhead.

Full GPU verification covers every word. Both GPU implementations also pass a CPU-reference check, a partial final tile, and an intentionally corrupted byte that must be rejected. Network host-only controls use CRC32 and no GPU copies. Payloads are not compressed. HMAC challenge authentication controls the direct LAN socket, bound to the Mac's specific interface; SSH starts the peer but carries no payload. The benchmark channel is for synthetic data and does not provide TLS confidentiality.

Wi-Fi rate and contention vary. Chunk sizes were tested sequentially; differences cannot establish a chunk-size speedup independently of radio changes. Report median and all three trials, not p95 or the best trial alone. This run measures transfer readiness, not full model continuation correctness, steady-state inference, or the final dual-host performance target.

## Reproduction and evidence

Build and launch instructions are in the [benchmark README](https://github.com/LittleLittleCloud/Infer.NET/blob/codex/kv-sync-benchmark/native/StateTransfer/README.md). After removing a machine-specific launcher path and adding smoke mode, eight bidirectional small GPU transfers passed at `f702499e4cf81ef312e253372f99df37f1c1737d`; those smoke timings are not included above. Native library sources did not change.

The [public JSON summary](../site/kv-sync-summary.json) includes every formal trial, warmup durations, per-stage timing, source versions and library hashes. Raw console logs, local connection/environment snapshots and remote trial logs are retained on the Mac and Windows but not published because they can contain private paths. No raw log URL is implied.

| Evidence | SHA256 |
| --- | --- |
| manifest.json | `65405cbf76bebd814758042b3fd02dbca0b5b672edbd04c74ad094049f38abc2` |
| trials.jsonl | `84a63ec16e946160d371c928e967cf66cf009f228a2b3cc875549e4d648e1f71` |
| windows.log | `b72c766f4b527e20cb05d6b219a33be3868a91a216676f29402850d5b08057be` |
| Metal library | `a5be9b93e4929b8d18a01cb2ace5aebd7084afc8f3741842d9d6c474e334310d` |
| CUDA library | `d738f4e495e7bcd3179e9bd1302f4dd99b76f405eecf046c3647cfb77a0e798e` |

During the host-test phase and after completion, the Mac was on AC power; power mode was High Power. System swap-out counters did not increase between that mid-run snapshot and completion. No pre-run swap baseline was captured, so this does not prove zero swap for the whole experiment. The mid-run system free-memory percentage was 57%; Windows started at 1262 MiB used, 0% GPU utilization and 28 °C. These snapshots do not establish contention-free conditions throughout.
