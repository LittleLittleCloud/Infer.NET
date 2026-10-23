# Native Metal profile and scratch capacity

The deployment reads the effective native graph settings before warmup and readiness. It logs `METAL_NATIVE_PROFILE` with KV element width, GEMM precision, matrix/expert/DeltaNet modes and scratch geometry. FP32 deployment rejects native BF16 GEMM or KV and a scratch capacity that differs from the requested value.

This is deliberately a native readback: .NET 10 on Unix keeps Environment.SetEnvironmentVariable changes in a managed dictionary, while Metal code reads libc getenv. The earlier managed profile assignments did not prove the native math configuration. This change preserves existing native defaults and makes them observable; it does not force a different matrix or recurrence implementation. Profile readback is an additive C ABI and requires packaging the matching native library.

`split-serve --metal-scratch-tokens 128` is an opt-in allocation experiment. Default remains512. Supported capacities128/512 must cover suffix-chunk-tokens32/128/512. Short local requests are already capped at128tokens. Request chunking, quantization and arithmetic remain unchanged; only preallocated temporary capacity changes. Smaller scratch is not a larger context window or a throughput claim.

Validation: native getter rejects null pointers and wrong result sizes, guards output bounds, reports changed precision/mode fields, and checks scratch32/128/512 allocation. CLI integration rejects invalid or insufficient scratch geometry. Real request parity, allocation and timing evidence will be appended after the experiment.
