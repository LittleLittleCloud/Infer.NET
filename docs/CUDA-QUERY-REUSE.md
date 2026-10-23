# Rejected CUDA attention query reuse experiment

2026-10-23, Issue #3. Source `d5699c7d773773df819c8157901514290e37e4b0` on `codex/cuda-query-reuse`. This is an operator-only experiment on the Windows RTX 4090; it is **not connected to the prefix executor or deployed inference service**.

The prior partitioned-key attention kernel changed expert routing on a long prompt. This candidate instead kept the retained key order and per-query FP32 accumulation order. It split the original attention kernel into the same softmax work and a value-reduction kernel that loaded each KV value once for up to four neighboring queries. The hypothesis was that fewer KV reads would help at long context without changing results.

All 40 cases passed a bitwise comparison against the retained kernel: 17, 257, 4096 and 32768-key lengths with tail query counts, plus 131072 keys / 32 queries; both FP32 and BF16 KV. Output guards remained intact. The standalone check did not load the model. Source was transferred through GitHub to an isolated Windows checkout; the resident deployed agent remained running. The GPU reported 19558 MiB already used and P8 before the check. These are three CUDA-event samples after one excluded warmup per path; no full-model, network, or Mac suffix timing was collected.

| Keys / queries | KV | Retained median | Reuse median | Outcome |
|---:|---|---:|---:|---|
| 4096 / 32 | FP32 | 0.742 ms | 0.891 ms | slower |
| 4096 / 32 | BF16 | 0.641 ms | 0.818 ms | slower |
| 32768 / 32 | FP32 | 6.004 ms | 9.272 ms | slower |
| 32768 / 32 | BF16 | 4.840 ms | 8.095 ms | slower |
| 131072 / 32 | FP32 | 83.603 ms | 95.932 ms | slower |
| 131072 / 32 | BF16 | 23.510 ms | 37.661 ms | slower |

Fewer independent CTAs and an extra kernel launch may explain the regression, but these measurements do not isolate either cause.

An isolated follow-up at source `0cef04d3bd2c5d2d22aa10068437248d8bc6d8a7` kept two queries in one 512-thread CTA and staged 32-key KV tiles in shared memory. It also passed all 40 bitwise cases, but was slower in each same-run comparison:

| Keys / queries | KV | Retained median | Shared-two median |
|---:|---|---:|---:|
| 4096 / 32 | FP32 | 0.734 ms | 0.930 ms |
| 4096 / 32 | BF16 | 0.636 ms | 0.908 ms |
| 32768 / 32 | FP32 | 5.957 ms | 7.627 ms |
| 32768 / 32 | BF16 | 4.796 ms | 7.036 ms |
| 131072 / 32 | FP32 | 84.804 ms | 88.777 ms |
| 131072 / 32 | BF16 | 23.549 ms | 33.427 ms |

Shared memory does not make this approach beneficial under the measured configuration. Both candidates are rejected for deployment. In particular, bitwise operator parity does not prove whole-model route parity, and the slower operators do not advance the fixed 128K prefill or decode gates. The deployed attention path is unchanged.

Reproduce from the [experimental branch](https://github.com/LittleLittleCloud/Infer.NET/tree/codex/cuda-query-reuse) in an isolated Windows checkout with `scripts/windows/check-prefill-query-reuse.ps1 -OutputDirectory artifacts\dual-host\query-reuse-NEW-ID`. The original JSONL, binary hash and device snapshot are retained only in the local Windows artifact directories `artifacts\dual-host\query-reuse-d5699c7` and `query-reuse-0cef04d`. First-run JSONL SHA-256: `F8F1B33DD27FE1498F3BDB2C1D20BEC37DB7C59B815FCB69F84F08B1B4B8659E`; checker executable SHA-256: `AE05D18580E0AC7BE444BA8F90ED6D6C6CA0013918BA659EF5CF082FE2EB9791`. Follow-up JSONL SHA-256: `C443B8613C6E9225ACF2477BD8F02FBF5F762486E32E0DF2A2691DCAB13F2ECF`; executable SHA-256: `9457DF9B0C31321942D2D27BB2113165EF1482360CBA45199E85BEA2BDD3832C`. The compact public numbers are in `site/cuda-query-reuse-summary.json`.
