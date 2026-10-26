# Experiment 92: Metal routed expert small-row specialization

Status: evaluation in progress; candidate unadopted. [Issue #92](https://github.com/LittleLittleCloud/Infer.NET/issues/92), [code PR #93](https://github.com/LittleLittleCloud/Infer.NET/pull/93). Independent results branch from main `4f51f241e87999c8cc3f79d733cd95ade9c711cd`.

The user approved plan revision 1 on 2026-10-25. This experiment tests whether 4-, 8-, or 16-lane groups per output improve single-token affine8 routed experts. The main kernel assigns a complete SIMD32 group per output. Default selection remains unchanged.

## Implementation and reference

Splash's [small-live-row column tile](https://github.com/incoai/splash/blob/1482f6bbc2abf12375332725302019594150b6cb/runtime/metal/kernels/common/gguf_column_tile.h) motivated specializing sparse expert tiles instead of padding one live row to an MPP tile. Its per-lane output layout is designed for Splash's tiled weight image and BF16/half operands. Our candidate preserves row-major checkpoint weights, BF16 scale/bias storage, FP32 operands and accumulation, and existing gate/up fusion. This is a transfer of the specialization principle, not a faithful port of Splash's kernel or evidence about its performance.

The candidate changes only routed gate/up and down. Shared experts, routing, prefill, dense projections, LM head, attention, KV, recurrence, selected expert count, and command-buffer scheduling retain their original algorithms. `DEEPINFER_METAL_EXPERT_LANES=0|4|8|16` is read natively; invalid selections and conflicts with generic kernels are rejected. The native log/getter exposes the actual selection. Unsupported formats explicitly retain the legacy path. No native INT8 matrix computation or lower-precision activation path is introduced.

## Measurement scope

Kernel source commits, complete raw timing rows and scalar comparisons are in [kernel.json](evaluations/issue92/kernel.json). Initial source and strengthened-oracle confirmation source use different coefficient/input fixtures and reverse variant order; their trials are not pooled. The source comparison is an in-binary original-kernel control, not a separately loaded full-model main benchmark.

Each timing cell records two excluded warmup batches and seven formal batches, each containing 64 dispatches with buffer barriers. Gate/up uses K=2048,N=512; down uses K=512,N=2048; eight selected experts. Warm-cache uses fixed IDs; rotating uses all 256 experts twice per batch. GPU command time divided by 64 is batch-amortized time including dispatch/barrier overhead, not an individual hardware-counter interval.

Independent affine8 scalar checks cover signs, zeros, cancellation, non-power-of-two inputs, tails, selected expert boundaries/reordering, distinct per-slot down inputs, finite outputs, and output sentinels. The confirmation's 320 reported comparisons include direct and graph-helper comparisons; they are not 320 independently generated fixtures. FP32 limits were fixed before evaluation: relative L2 <1e-4 and maximum absolute <0.01. Existing graph hardware checks cover 60 cases plus handoff/scratch guards.

Full-model measurements, application evaluation, resource reconciliation, restoration, and the adoption decision will be completed before finalizing this record. No provisional microbenchmark number establishes a HumanEval gain or formal 128K acceptance.
