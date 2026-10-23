# Bounded suffix receive overlap experiment

The 3fee488 FP32 32768-input baseline spent about 53.2 seconds in Metal suffix execution and 12.5 seconds waiting for hidden/KV frames, out of 66.4 seconds of worker prefill. The receive loop previously stopped while the GPU executed a suffix batch. This experiment lets the next batches arrive while that GPU work runs.

`--suffix-receive-overlap off|on` defaults to `off` until measured acceptance. Both modes use the same validated batch assembly and native arithmetic. CUDA still sends 32-token bundles; the Mac coalesces the configured 32/128/512-token suffix batches. Wire version4 and CUDA ABI6 are unchanged. Incremental and final-state-only KV transport are supported.

The reader is the sole owner of ReceiveAsync during hidden/KV batches. A channel holds at most two completed batches, plus one being read and one being consumed. At suffix128, these four batch payloads total at most14MiB, plus a transient authenticated frame and runtime overhead. State import remains on the GPU consumer after its corresponding suffix completes. The reader stops at the exact hidden-stream boundary and is joined before final recurrent-state reception and acknowledgement. Cancellation or GPU/receive failure cancels and joins the reader before its peer is disposed. No replay, reordering, compression, or GPU arithmetic change occurs.

`hiddenWaitMs` measures the consumer's exposed waiting time, including batch validation/coalescing and scheduling. It does not measure total concurrent network-reader activity. Do not sum overlapping sender, receiver and GPU durations as if they were serial.

Validation pending: unit tests for boundedness, cancellation, failure ownership and exact irregular-tail batches; both-host AOT/integration; same-package off/on8K and32K tests, byte/token equality, cancellation/reconnect, memory and swap monitoring. This document describes the experiment, not a measured speedup.
