# CUDA / Metal state transport probe

This standalone microbenchmark measures synthetic, GPU-resident state over authenticated direct LAN TCP. It is not a model backend, production state protocol or prefix-cache implementation. No model weights are loaded by the probe. See `docs/KV-SYNC-BENCHMARK.md` for geometry, timing and measured results.

Prerequisites: the Windows user-tool/MSVC/CUDA setup from PR #6, SSH alias `infer-windows` with a verified host key, and independent clean checkouts of the same commit. By default the Windows checkout is under the SSH user's `repos/Infer.NET` directory. Source synchronization uses GitHub.

Build on Mac:

```sh
mkdir -p artifacts/state-transfer
clang++ -std=c++17 -O2 -fobjc-arc -dynamiclib native/StateTransfer/transfer_metal.mm -framework Foundation -framework Metal -o artifacts/state-transfer/transfer_metal.dylib
```

Build on Windows from its repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\windows\build-transfer.ps1
```

Run on Mac, with a new output directory for every attempt:

```sh
python3 scripts/benchmark-state-transfer.py server --smoke --output artifacts/dual-host/unique-smoke
python3 scripts/benchmark-state-transfer.py server --output artifacts/dual-host/unique-full-run
```

The Mac derives its bind address from the route to the configured SSH peer and opens an ephemeral port on that interface only. Windows connects outbound. No firewall changes or SSH payload tunnels are made. A fresh HMAC challenge secret reaches the peer only via SSH stdin; it is not written to logs. Payloads are synthetic and transmitted without compression or TLS. Do not use this probe as a private model-state service.

Both GPUs first check the generated pattern against CPU values, verify a tail, and reject an intentionally corrupted byte. Full runs contain 10 configurations (five payload/chunk combinations, both directions), each with one warmup and three trials. Every receiving GPU checks every transferred word. Host-only controls use CRC32. Mac elapsed time includes GO, transfer, receiver validation and acknowledgment; allocation and initialization are excluded. CUDA staging time is measured separately. Metal buffers use shared storage, as in the current model's KV implementation.

The benchmark is deliberately serial: CUDA copies do not overlap socket I/O, and no inference competes with the probe's work. Any pre-existing services remain untouched. This establishes a transport baseline, not the performance of a future overlapped inference pipeline.

Mac output contains `manifest.json`, `trials.jsonl` and the remote console log, updated incrementally. Windows retains its own trials and completion metadata under `artifacts/state-transfer/<run-id>/`. A failure preserves partial results; do not reuse its run ID or silently count it as passed.
