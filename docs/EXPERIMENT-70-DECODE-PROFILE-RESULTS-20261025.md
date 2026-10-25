# Experiment 70: representative Metal decode stage measurements

Date: 2026-10-25 America/Los_Angeles. [Experiment #70](https://github.com/LittleLittleCloud/Infer.NET/issues/70), [candidate code PR #90](https://github.com/LittleLittleCloud/Infer.NET/pull/90). **Collection in progress: partial data below are explicitly labeled.** Diagnostic implementation remains unadopted. No optimization or formal 128K acceptance is claimed.

## Scope and authorization

The user requested measuring decode stage shares after reviewing Ignis and Splash. This resumed previously approved diagnostic #70 after its earlier session closeout; the earlier research and closeout remain historical records. The resumed scope and authorization are recorded in the Issue before implementation.

Main baseline: `4f51f241e87999c8cc3f79d733cd95ade9c711cd`. Its difference from the original #70 baseline `4ca29eed1a2acb8a870d33e980dce37dfbc1b7e3` contains no runtime/native changes. Final encoder-mode candidate: `43c81faff69a5e503a0f04be9ca664c73d89cce8`. Earlier full-model candidate: `bd3daa6db80d3230e8ccec9d974989a536dabec9`. Earlier `de7c332` built the initial instrumentation; no full-model candidate cohort ran on that version. The later revision clears diagnostic state on reset and prefill, without changing arithmetic.

M5 Pro 20 GPU cores /48 GiB, AC/High Power; pinned Qwen3.6-35B-A3B affine8/group64 model revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`. Entire model runs locally on Metal. FP32 KV, recurrence and existing arithmetic, explicit fused-fp32 decode attention, scratch128, managed local-prefill chunk32, native scratch128 (matrix/chunked-Delta/grouped-expert capabilities reported separately in the runtime profile), concurrency1, request state cleared, no prefix cache or speculation. Windows takes no inference part and its original agent is left untouched. Per-platform artifacts/runtime builds are retained separately from the source checkout.

## Measurement boundaries

The fixed shapes are 21/128, 8192/128 and 32768/128 input/output tokens. Phase A measured uninstrumented main and bd3daa6 default-off with one warm-up and three retained trials in the same shape order. Its attempted counter-on mode had unavailable dispatch counters: short/8K completed as host-only measurements;32K warmup was interrupted. Phase B measures43c81fa encoder-on then default-off, each with one warm-up and three retained trials per shape. The independent two-encoder check passed output equality, valid timestamps and exact dispatch coverage before full-model use. Decode uses 127 first-to-last output intervals per fixed request; prefill includes producing the first output. Throughput ignores EOS. Natural EOS is handled normally in the application pilot.

Native samples cover the first eight classifier forwards after reset and subsequent positions divisible by128. Fixed-request stage shares therefore describe selected decode positions, not all 127 intervals. The first output normally comes from prefill and is outside these native decode samples. Application samples are also position-selected, not a time-weighted whole-cohort cost profile.

M5 reports dispatch-boundary sampling unsupported, encoder/stage-boundary sampling supported, and timestamp-buffer creation successful. Phase A has no valid per-stage shares. Phase B automatically samples coarse encoder regions. Splitting encoders changes scheduling; dispatch-mode counters on supporting hardware instead insert barriers. Region times include the selected dispatches and, for some regions, trailing buffer synchronization; fused attention includes both of its dispatches. Encoder-mode MoE includes routing, shared/routed experts, activation, merge and next normalization; expert gate/up/down are not separately sampled in this mode. Zero-valued unused fine-grained tags do not mean their operations cost zero. Dense projections include attention and DeltaNet input/output projections; the Delta recurrence percentage is not the cost of the entire DeltaNet block. Report measured-region shares and the sum relative to complete command GPU duration. The remainder is unattributed, not a measured barrier or scheduling cost. CPU encode, commit, wait, and GPU execution overlap: wait includes GPU work and cannot be added to it. Native readiness stops before diagnostic logging and managed emission; it does not measure client TTFT. The observer measures valid text/tool-call arrival separately.

One final CPU compilation overlapped part of main 32768 trial3; the controller was briefly held between arms until candidate packaging completed. That trial is retained and this is a diagnostic comparison, not clean formal adoption evidence. No GPU benchmark overlapped another GPU benchmark.

## Validation and failures retained

- Initial native build failed because a source insertion matched a Q8-branch-specific line absent on main; corrected and subsequent AOT builds succeeded.
- First managed test invocation used `--no-restore` in a new checkout and failed for missing assets. Restored invocation passed131 checks.
- Initial instrumentation native graph check passed60 hardware cases and existing KV/profile guards. Existing legacy/fused decode attention checks at lengths17,2048,8192,32768 passed their independent numerical checks. Candidate counter-on/off full-model output checks provide the later reset/prefill instrumentation check.
- Python compilation/help and whitespace checks passed. CI status is recorded in the PR.

## Preliminary sampled stage shares

These are shares of measured GPU regions, not physical ALU/Tensor utilization or clean throughput. Short and8K have all three formal trials;32K currently has one formal trial. Final on/off perturbation and application-pilot results are pending.

| Coarse region | Short prompt21 |8192 input|32768 input (partial)|
|---|---:|---:|---:|
|MoE block|40.0%|33.6%|25.0%|
|Dense input/output projections|37.9%|31.8%|23.9%|
|LM head|13.6%|11.3%|8.5%|
|Attention preparation/core|3.5%|19.1%|39.6%|
|Delta recurrence|2.1%|1.8%|1.3%|
|Delta preparation/output gate|1.4%|1.2%|0.9%|
|Other normalization/embedding/argmax|1.5%|1.3%|1.0%|

The provisional priority is quantized linear projections/MoE and long-context attention. This does not identify the limiting hardware resource within those regions, establish achievable speedup, or demonstrate that native INT8 is faster than a BF16/tiled alternative. Delta input/output projections remain in the dense category.

## 128K and resource limits

FP32 capacity preflight rejects128K in this round. Retaining observed non-KV allocation and replacing KV with135168-token FP32 capacity projects42,584,391,680bytes (39.660GiB), above38.5GiB before further attention/workspace growth. This is an allocation projection, not a measured peak or a proof of minimum memory. No128K request was issued and no lower precision was substituted.

Completed Metal allocations, per-request whole-machine swap-out, selected process-footprint snapshots, system memory pressure and one-second RSS/VM samples are retained. RSS does not replace process footprint or GPU residency. Loading/warmup and formal swap are distinct. These diagnostics do not establish continuous transient allocation, per-submission physical residency or formal128K resource acceptance.

## Evidence and adoption

Raw evidence is local-only in the ignored `artifacts/dual-host/issue70-decode-profile-20261025/` (phase A) and `artifacts/dual-host/issue70-encoder-profile-20261025/` (phase B) under the candidate worktree; public summaries retain hashes and sanitized fields, not prompts, solutions, credentials or private endpoints. The separate results PR does not adopt candidate code or publish Pages until explicitly merged. Fixed20 remains required before adopting the profiler or claiming a formal main performance comparison. This round is diagnostic only.
