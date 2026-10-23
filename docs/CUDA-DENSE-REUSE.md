# Exact FP32 CUDA weight reuse — 2026-10-23

CUDA prefill was repeatedly reading and dequantizing the same dense weights for each token. The new opt-in path shares decoded weights across four/eight tokens, uses aligned eight-byte loads with one scale/bias pair, and applies packed loads to routed experts. Per-lane accumulation and warp reduction order remain unchanged. No lower-precision operand conversion, model-sized workspace, or wire change is introduced.

`--cuda-prefill batch-fp32-packed` selects the candidate; `batch-fp32` remains the retained baseline. The default is unchanged. Both hosts use source `8b5e3bd51cf9d3465e4302f43b4fe00575b95f36`, affine8/group64 weights, FP32 GEMM/KV, CUDA32/Metal128, incremental KV, send overlap on, receive overlap off, fused FP32 decode, no cache. Fixed prompt hashes are in the public JSON. Every cohort has one warmup and three measured requests; output length is fixed, ignoring EOS for throughput.

## Deployed same-package A/B

| Input/output | Baseline prefill s | Packed prefill s | Baseline client s | Packed client s |
|---|---:|---:|---:|---:|
| 8192/128 | 14.319 | 11.348 | 16.426 | 13.452 |
| 32768/128 | 66.929 | 52.908 | 69.776 | 55.733 |
| 32768/4096 | 66.483 | 53.382 | 156.713 | 143.482 |

All output IDs match the retained baseline, including each4096-token result. Before changing modes, both workers were reloaded to match the baseline8K allocation/capacity, then each mode ran8K,32K,32K-long in the same order. Power was checked at trial boundaries. Whole Windows GPU memory was sampled every500ms; Metal allocations are response samples, not continuous high-water measurements. One packed long-output measured trial added267.719MiB of whole-system swap-out; all other measured requests added zero. That trial is retained and remains below the fixed512MiB limit. The counter cannot be attributed to a single process. Every measured swap delta and memory sample is reported in the JSON; no formal128K resource claim is made.

## Native proof and limits

35dense operator cases and8expert cases passed bitwise retained-FP32 comparisons, independent double-oracle checks for small cases, and tail guards. Three separate native-prefix experiments measured progressively: token reuse13.172→11.483s; vector weight loads13.253→11.018s; adding packed expert reads13.288→9.706s. Each used fixed8192token IDs, one warmup plus three measured passes per mode, and matched all hidden output/40final states byte-for-byte. These native timings exclude network/suffix/decode and are not summed with overlapping pipeline intervals.

The complete84input/8output cross-host verification passed1748tensor/logit checks (max relativeL2≈1.4991e-5, maxAbs≈0.0001234) with matching greedy IDs. Mac124tests passed; Windows123passed with1platform skip; both Native AOT packages,39HTTP checks and32chat protocol checks passed. Three-platform CI at8b5e3bd passed. Live prefill cancellation recovered in1.936s with unchanged Mac/Windows model PIDs, matching next8tokens and deterministic local requests; all15deployment checks passed. Measured Metal response allocations matched between modes (context capacity36864 in the long case); their different warmup allocation samples are retained separately. The earlier CLI integration failure was an outdated expected error string after adding the new option; it was preserved and fixed before packaging.

Reproduce operator validation with `scripts/windows/check-prefill-reuse.ps1`; native prefix A/B with `scripts/split/profile-prefix.py --compare-reuse --packed-experts` and the pinned model/token artifact paths. Run service A/B with the two explicit `--cuda-prefill` modes and the fixed inputs in `docs/QWEN36-8BIT-TARGETS.md`. Raw logs, model/binary hashes, SSE events and allocation samples remain local under the run IDs in `site/dense-reuse-summary.json`.

The formal BF16 GEMM/KV and131072input targets remain unpassed. Worker prefill and client TTFT are reported here, not the still-uninstrumented formal coordinator t2−t1. The rejected BF16 reduction experiment and unintegrated attention-operator figures remain separately documented. [PR23](https://github.com/LittleLittleCloud/Infer.NET/pull/23) is unmerged; no automatic PR merge is performed.
