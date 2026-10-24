# Experiment 54: observed BF16/2048 HumanEval retake

The original [BF16/2048 comparison](EXPERIMENT-54-BF16-2048-20261024.md) measured worker prefill and decode but did not capture client first-token timestamps. Its missing TTFT remains **unmeasured**. This is a separate 20-task run at the same candidate worker commit, with a streaming observer around the client-facing endpoint; it supplies an observed TTFT point without rewriting the original result. The candidate remains **not adopted**.

| Measurement | Observed BF16/2048 retake |
|---|---:|
| Pi HumanEval tasks completed | 20/20 |
| Model requests with client first-token and worker timing | 121/121 |
| Median client TTFT | **7,998.56 ms** |
| Worker prefill | **413.43 input tok/s** |
| Worker decode | **66.42 output tok/s** |
| Worker input / output tokens | 291,205 / 21,660 |
| Pi total wall | 1,037.75 s |

The official HumanEval assertions were not run. “Completed” means Pi produced a syntax-valid solution for each task; it is not pass@1. Worker speed aggregates token counts over the matching worker intervals. TTFT is the median of all 121 observed requests from the observer proxy's outbound request start to the first valid text or tool-call output delta. Headers and role-only events do not count. Pi-to-proxy dispatch is outside this boundary. The observer added two local loopback streaming hops, so the TTFT point has its own configuration and cannot be treated as the missing timestamp of the original run. Request count, generated workload and wall time also differ from that run; these data do not establish a controlled performance gain or regression.

The fixed cohort used the 20 official HumanEval task IDs in the original run manifest, seed 0, Pi 1.4.0, context 32,768, output cap 4,096, and pinned model revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`. Mac ran the full-model suffix/decode with affine 8-bit group-64 weights and FP32 KV; Windows RTX 4090 held the first 20 layers with BF16 Tensor Core dense projections and 2,048-token CUDA outer chunks. All 121 requests reported `cuda-metal-split`; no short-local fallback, cache hit or concurrent GPU benchmark was included. Mac was on AC and Wi-Fi; Windows used its existing LAN connection. Link bandwidth and machine-wide peak memory were not measured. This is a Wi-Fi/FP32-KV diagnostic, not Q8-KV or 128K acceptance evidence.

The Mac serve and Windows agent both ran candidate source `93414ebcb2190447e8f6fa09ed24dd7b7a5570e4`. The coordinator remained at `cd6f7da401fadf044dac13e87b415d27bb5eadfb`, and the Pi harness came from `90d37dd967a3a04cf1260a85511a70914da1ed78`. Mac executable SHA-256: `9aa19566d9c69136e8f37936404911822dbe352c18a88888c51a6bf3a3933107`; Metal library: `1739ff5dd22a4b158f97d84d39268478a75d5683f4a9ad1a3cd0d1a8f79d4aef`. Windows executable: `de13860acd86c7bee208705493e058675ad6aea4e6dab3a50c4fbd94534e23bb`; CUDA library: `a5464e1934177816e7cfe1f3d0ae9b18f961a40c78e4b363ee6477d7b38d98cb`.

An independent short split request warmed the candidate before a separate five-task pilot and this formal 20-task cohort. The pilot had 25/25 observed requests; its timings are not mixed into the table. The formal run had 20/20 completed tasks, no timeouts or request failures, and 121/121 matched worker and TTFT observations. Pi's token totals equal the observer's 291,205 input and 21,660 output tokens. The local ignored evidence directory is `artifacts/dual-host/issue54-ttft-retake-20261024T165739Z/`; raw logs are **not** a public download. Key SHA-256 values:

| Local evidence file | SHA-256 |
|---|---|
| `cohort-20/manifest.json` | `bfae8f9e4da32d6368463d17d4d08ab1936b9e04c605a8dcc03c2489e13f8d3e` |
| `cohort-20/tasks.jsonl` | `cda0bbbe7c924592e72de1fc1004f16487b6e8a1717cd890f0ed06cc368f6d07` |
| `cohort-20/summary.json` | `2822c732d2d07c2d6e46a0de3e2347a0caba921efc539c8460dbe70450ebf2ec` |
| `cohort-observer/requests.jsonl` | `e844f053aa177937e79c71f06e71018798e0b052467cb4d244b40e0ef8ae1316` |
| `preflight.json` | `41f882dd0ccc49711d31d65a90e17395fffc1063fabc5c642a08088ac1a99398` |

After the retake, the candidate processes were stopped by their verified PIDs and the previous Mac LaunchAgent and Windows agent were restored. This run adds a real TTFT measurement to the public trend series while preserving the original missing-data point and its negative adoption decision.
