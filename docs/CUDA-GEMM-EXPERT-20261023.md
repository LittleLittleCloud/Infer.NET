# CUDA affine8 GEMM and expert candidate, 2026-10-23

Issue [#38](https://github.com/LittleLittleCloud/Infer.NET/issues/38) tracks this opt-in SM89 experiment. The deployed `919259d` Mac serve and Windows agent were left unchanged. All numbers below are **single-operator CUDA-event measurements**, not full-prefix, cross-host or 128K throughput. A resident but idle Windows agent occupied about 21 GiB of the 4090's 24 GiB; the microbenchmarks used a separate source checkout and small allocations. The operator scripts warmed up once and reported three trials of five launches each. Their raw results remain local on Windows under the indicated `artifacts/dual-host` directories; they are not public downloads.

## Dense BF16 GEMM: vectorized affine8 tile loading

The existing 32 × 64 output tile used the same BF16 Tensor Core MMA before and after this change. The candidate loads eight adjacent affine8 weights together and loads their shared group64 scale and bias once, then converts each dequantized value to BF16. It preserves BF16 input operands and FP32 accumulation. The candidate at `4c0aed36064914f0229414363d49d00c59052012` was compared with unchanged `main` at `e017d84e805c016d5912b5051e4c69f758056363` on the same RTX 4090, using `scripts/windows/benchmark-prefill-ops.ps1`. The independent double-BF16 oracle covered small/tail shapes; all large shapes had relative L2 below `7.3e-7` and max absolute error below `4.8e-7` against the SIMT BF16 operator. Both runs reported `passed=true`.

| Tokens × rows × cols | Old Tensor Core median | New Tensor Core median | Kernel-only speedup |
|---|---:|---:|---:|
| 32 × 8192 × 2048 | 0.2133 ms | 0.0692 ms | 3.08× |
| 128 × 8192 × 2048 | 0.2517 ms | 0.1071 ms | 2.35× |
| 512 × 8192 × 2048 | 0.7641 ms | 0.3463 ms | 2.21× |

The raw baseline/candidate directories are `gemm-experts-e017d84-ops` and `gemm-experts-4c0aed3-ops` under each checkout's ignored `artifacts/dual-host` directory. These figures do not establish a deployed speedup: `batch-fp32` does not use the BF16 Tensor Core kernel, and the BF16 model path has outstanding cross-backend quality checks documented in [the stage probe](BF16-STAGE-PROBE-20261023.md).

A same-binary check at `25b0153` retained the original scalar tile loader alongside the candidate: all six shapes, including non-full rows/tokens, produced **bitwise-identical Tensor Core output**. The 32 × 8192 × 2048 scalar/packed medians were 0.2134/0.0690 ms. The checker reported `passed=true`; its raw JSONL SHA-256 is `b6bb0536ff827d74ef0c24a4c70f751e947aff4dcb8f0c7aa38089a8ba4da2c5` in `gemm-experts-25b0153-ops`. The GPU was in P8 at 31°C and 7.34 W before warmup, with 21107 MiB allocated by the resident agent; CUDA events cover warmed kernel execution, not model loading or transport.

## FP32 expert and dense candidate

The new `batch-fp32-fast` option combines three FP32 changes while retaining the existing 32-token chunk and FP32 KV:

- Dense projections reuse packed affine8 weights over a 4- or 8-token tile. The earlier operator checker verifies bitwise equality with the original GEMV across tail, large QKV and output-projection shapes; the 32 × 8192 × 2048 case measured about 0.43 ms versus 0.64 ms for original GEMV in the latest isolated checker.
- Routed experts use the existing aligned packed-weight kernel. At `e3dce22`, the 32 × 4096 × 2048 gate/up-shaped case measured median **2.38 → 1.73 ms** and the 32 × 2048 × 4096 down-shaped case **2.37 → 1.69 ms**. Routed and non-routed outputs matched original FP32 GEMV bitwise. This is an activation of an existing validated primitive in the new mode, not a new Tensor Core expert GEMM.
- The shared expert's gate/up projections share one kernel launch and activation load stream. At `2a98284`, 32 tokens × 512 rows × 2048 columns with one shared expert measured median **0.0836 → 0.0327 ms**, with outputs bitwise equal to the two production SIMT kernels.

The first attempt to fuse the **large routed** gate/up matrices was a negative result: for 32 tokens × 8 routes × 4096 rows × 2048 columns, separate packed kernels took about **3.48 ms**, while dual-accumulator fusion took **4.83 ms**. A single-launch dispatch variant also took about **4.82 ms**. Both were bitwise correct. The final candidate keeps large routed gate/up separate and fuses only the small shared expert. The failed measurements remain in `gemm-experts-de4e1bc-reuse`, `gemm-experts-9760a33-reuse`, and `gemm-experts-25863e2-reuse` on Windows.

The routed expert checker at `e3dce22` passed with JSONL SHA-256 `94acc1d99a50b3dd6e69cc7c5ac624d055147ccf1bcd72ebf028cbe012ad07a8`. The shared-expert checker at `2a98284` passed with JSONL SHA-256 `bef9dfa6dc3a95fa9d25c8fbb326ed115a31d7559bef15658e148a7496efdb85`. At `cbff46a`, the native SM89 DLL and managed Host built on Windows, Windows tests passed 127 with one Metal-only skip, and Mac tests passed 128; the Host build also passed on Mac. These establish build and operator correctness for the candidate, not full-model equivalence.

## Isolated full-prefix A/B and restoration

With the user's permission, the verified old Windows agent was briefly stopped to load the pinned 8bit prefix into a separate candidate process. The old `919259d` executable and config were restored after each run. The final restoration is a foreground SSH session, PID `6276` at the time of measurement; the 4090 again held about 20011 MiB, Mac `/ready` returned HTTP 200, the coordinator reported a healthy connected agent, and a real 16-token request returned HTTP 200. The first attempted detached `Start-Process` restoration did **not** survive SSH logout; it was immediately replaced by foreground execution. The automation now explicitly uses the foreground mode. These are recovery facts for this run, not a claim of Windows Service installation.

The candidate DLL SHA-256 was `ce04673d942c0a4eef298ad1a38ea1cae41ad5d8e00ba4e54e3378896ca81bbe`; its native/managed source did not change between `25b0153` and the script-only runs below. The benchmark loaded the same pinned prefix once per run, used FP32 KV and 32-token native chunks, verified 131 synthetic token IDs including irregular tails, then alternated one warmup and three measured 512-token passes per mode. All 20 layer traces, final hidden vectors, all 40 KV/Delta state tensors, incremental KV versus final KV, and the 512-token hidden hashes were bitwise equal. The comparison does **not** include Mac suffix, logits, network, decode, real prompt tokenization or 128K behavior.

| Same-binary FP32 comparison | Baseline median | Candidate mode 6 median | Native prefix-only change |
|---|---:|---:|---:|
| Mode 1 → mode 6, source `8c0e76c` | 0.7679 s, 667 tok/s | 0.5432 s, 943 tok/s | 1.41× versus older unpacked mode |
| **Deployed packed mode 5 → mode 6**, source `4e00cae` | **0.5493 s, 932 tok/s** | **0.5408 s, 947 tok/s** | **1.016×, 1.55% less time** |

The current deployed agent advertises `batch-fp32-packed`, which maps to mode 5. Therefore the second row is the relevant incremental comparison. Mode 5 already uses packed dense and routed expert kernels; mode 6's remaining change is principally the small shared-expert fusion. The large operator-only speedups cannot be projected onto the present service. Each run's raw `result.json`, 40 state files, traces and log are local-only under Windows `artifacts/dual-host/gemm-experts-full-ab-8c0e76c` and `gemm-experts-packed-ab-4e00cae`. Their result JSON SHA-256 values are `80f47afff7516ee092acb6031a13a027cfbd3b6c8bda26b76032a2021dca4c2a` and `cfe8492ad3a3dced50af4ebaf7bcbca678788dcbcfed3887b55ae611798d2b7e`, respectively. The 512-token timing is a diagnostic, not a target acceptance run.

`batch-fp32-fast` remains opt-in. Before selecting it as default, run a real split RPC A/B with compatible builds on both hosts, verify suffix logits and complete generation, then satisfy the correctness, memory and 128K cold-prefill/long-decode gates in [the fixed target](QWEN36-8BIT-TARGETS.md). The modest mode 5 → 6 gain means the next performance work should focus on larger routed expert computation and overall chunk execution, with correctness checked at each arithmetic change.
