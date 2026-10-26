# Experimental target block verifier

This candidate is tracked in experiment #95 and is disabled in normal inference. It exposes a target-only verification transaction through `MetalSplitSession`; it does not load or run a draft model. The original scalar `Forward` remains available. The verifier currently requires local prefill; imported split-prefix sessions are rejected.

Use the pinned Qwen3.6 affine8 checkpoint and FP32 KV/operands/state. Construct the session with FP32 fused decode attention and prepared scratch, reserve the required context, and call `PrepareTargetBlock(8)` after reset and before prefill. Preparation is lazy and rejects incompatible profiles and a forecast state-buffer allocation above the experiment's 38.5 GiB limit. Callers must also observe the resource limits when growing context. Eight prefix slots add 519127072 bytes; this is separate from normal KV and prefill scratch.

`VerifyTargetBlock(tokens, logits, trace)` accepts 1/4/8 consecutive input token IDs. It returns one greedy classifier result per input row, optionally writing `[rows,248320]` FP32 logits and `[rows,40,2048]` post-layer residual traces in row-major order. It does not advance the committed graph position. Recurrence and convolution history remain unchanged; attention writes tentative KV rows behind the current valid-prefix boundary.

There can be only one pending block. Scalar forward, another verify, prefill, context reserve/grow and state import are rejected while it is pending. `CommitTargetBlock(accepted)` selects an accepted prefix from 0 through the block length. Zero discards the block. A positive prefix copies the selected recurrent state, updates convolution history from the captured projected inputs, and advances the valid KV position. It performs no accepted-prefix forward replay. Reset discards a pending block; an execution failure faults the graph until reset. Sessions are single-request objects and do not support concurrent callers.

`TargetState` is a read-only diagnostic API for full-model live state and pending recurrent prefixes. It is distinct from the product's 20-layer state-transfer protocol. `ScalarTargetTransaction` is a diagnostic control that executes every input using the original scalar forward, captures the selected prefix once and restores it after evaluating the rejected suffix. Its copy costs belong in the control's transaction time.

Run hardware checks only in an exclusive GPU window, after preserving the original service and its recovery path:

```sh
bash scripts/check-metal-target-block.sh
DEEPINFER_METAL_KV=fp32 DEEPINFER_METAL_PREFILL_CHUNK=32 \
DEEPINFER_METAL_PREFILL_GEMM=fp32 DEEPINFER_METAL_PREFILL_MATRIX=0 \
DEEPINFER_METAL_PREFILL_DELTA=sequential \
dotnet run -c Release --project tools/TargetBlockBench -- \
  gate "$MODEL_DIRECTORY" artifacts/target-block-gate
```

The `long-gate` and `bench` modes use scratch 2048 and should be run with FP32 GEMM, prefill chunk 2048, matrix 1, chunked DeltaNet and fused-all FP32 prefill attention. The benchmark clears request state and rebuilds the prefix for every warmup/formal trial. It records target transaction wall/GPU times, commit wall time, compute dispatches, submissions and resource snapshots. These are candidate input rows and transaction milliseconds, not emitted-token autoregressive throughput. Complete DFlash generation and HumanEval application throughput must be evaluated separately.

No GPU check or service maintenance is triggered by building this project. See the independent experiment report for retained failures, measured commits, numerical/resource coverage and adoption disposition.

The complete-state numerical harness allocates CPU reference arrays and writes state fixtures. These diagnostics are separate from product transaction timing. For bounded-memory validation use `DOTNET_GCHeapHardLimit=0x40000000` (1 GiB) in the diagnostic child process; it does not change Metal allocations, model precision or production configuration. Keep earlier unconstrained diagnostic resource outcomes in the experiment record.
