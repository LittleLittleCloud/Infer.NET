# Prefill stage timings

`timings.prefillMs` and the Dashboard's Prefill value use the Mac worker's monotonic clock. The clock starts after prompt tokenization and Mac KV reset, immediately before the split Start frame; it stops after all hidden chunks, state frames, imports, End frame and acknowledgement. It includes the first output-token computation in the final Mac suffix, but excludes subsequent autoregressive decode, tokenization, reset, coordinator queueing and response delivery.

The optional `timings.prefillStages` and worker `lastRequest.prefillStages` fields itemize the completed request:

| Field | Measurement |
|---|---|
| `cudaComputeMs` | Windows elapsed time inside chunk production, including the CUDA run/readback and incremental KV export. |
| `cudaStateExportMs` | Windows final state extraction/readback after all hidden chunks. |
| `cudaSendMs` | Windows time spent writing hidden, KV and state frames to the authenticated socket, excluding the final timing frame. It can include socket backpressure. |
| `cudaElapsedMs` | Windows stage wall time from after prefix reserve to just before the final End frame. |
| `wirePayloadBytes` | Hidden, incremental KV and final-state payload bytes sent, excluding framing. |
| `macHiddenWaitMs` | Time the Mac worker waits for the next hidden batch. It includes time spent waiting for upstream computation and network delivery. |
| `macSuffixMs` | Mac suffix compute and first-token selection. |
| `macStateWaitMs` | Mac wait for final state frames. It includes upstream state export and network delivery. |
| `macStateImportMs` | Mac state/KV import. |
| `macOtherMs` | Mac prefill wall time outside the measured waits, suffix and import, including Start/End messaging and scheduling. |

`cudaComputeMs + cudaStateExportMs` is the 4090 front-stage work shown in the Dashboard. The two host clocks are independent. CUDA production, socket writes and Mac suffix compute can overlap, so **do not sum these durations to obtain prefill wall time**. In particular, `macHiddenWaitMs + macStateWaitMs` is a receiver-wait measurement, not isolated network latency. A transfer-only latency measurement would require a separate controlled transport benchmark.

An older agent sends an empty End frame. The Mac service accepts it and leaves `prefillStages` absent instead of reporting zeros as measurements. Deploy the Mac service before the upgraded Windows agent; an older Mac service rejects the new End payload. Keep both binaries from the same source revision for normal operation.
