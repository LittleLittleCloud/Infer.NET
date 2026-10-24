# Dual-machine resident prefill / decode takeover

2026-10-22, experimental implementation see [PR #8](https://github.com/LittleLittleCloud/Infer.NET/pull/8), associated [Issue #3](https://github.com/LittleLittleCloud/Infer.NET/issues/3). **Real dual-machine short requests have been run through, but via explicit FP32 KV diagnostic mode; BF16 target path, 128K, and performance acceptance have not yet passed.**

Reuse existing network: Mac Wi-Fi → home router → Windows wired gigabit port. Product uses custom C#/CUDA/Metal, without calling MLX, llama.cpp, or vLLM for inference.

## Execution and Storage Responsibilities

- Windows RTX 4090: Fixed 8bit model embedding and the preceding 20 layers remain resident, executing the CUDA frontend. There is no on-demand expert loading or CPU inference fallback.
- Mac M5 Pro: The complete 40 layers remain resident; Metal executes the subsequent 20 layers, and after handover, independently runs all 40 layers for decoding.
- Mac dispatches token IDs via authenticated TCP. Windows processes up to 32 tokens per chunk, sends the FP32 hidden states, and then proceeds to the next chunk; Mac simultaneously processes the latter half of the previous chunk, with TCP backpressure limiting in-flight data.
- After the hidden states, Windows sends the K/V for the preceding 5 full-attention layers, and the DeltaNet states and convolution history for the 15 linear layers.
- Mac verifies the identity, position, size, sequence number, and HMAC of each tensor before importing them into Metal shared buffers. Only after all 40 state tensors are imported and an End signal is received can it take over; the native graph also rejects decoding with missing states.
- SSH is only responsible for startup, Git synchronization, and logging; model data travels over an independent TCP connection. Windows initiates the connection to Mac, requiring no new inbound firewall rules on Windows.

This does not constitute a unified VRAM pool. Model weights remain fixed at affine8/group64; `--kv fp32` only improves cache precision, used to isolate computational differences and state handover issues.

## Actual Test Results

Both ends successfully completed managed builds and Native AOT publishing. Mac and Windows each passed **79 unit tests**, covering tampering, replay, truncation, incorrect keys, incorrect KV profiles, tensor sizes, and context limits.
CUDA independent GEMV max abs `3.8082e-8`, DeltaNet output/state max abs `1.0201e-8`, verified against scalar double accumulation reference.

| Mode | Input / Output | Numerical Check | Max Relative L2 | Max Absolute | Greedy Tokens |
|---|---:|---:|---:|---:|---|
| Managed, FP32 KV | 17 / 8 | 388/388 | 7.2243e-7 | 8.4639e-6 | Consistent with 8/8 |
| Managed, FP32 KV, two chunks 32+29 | 61 / 32 | 1292/1292 | 2.6345e-6 | 4.1962e-5 | Consistent with 32/32 |
| Both ends Native AOT, FP32 KV | 17 / 8 | 388/388 | 7.2243e-7 | 8.4639e-6 | Consistent with 8/8 |

The reference is the complete Mac Metal path for the same fixed 8bit model, with FP32 GEMV, FP32 KV, and up to 32 tokens per chunk. Comparing all hidden states of the preceding 20 layers, all frontend states, and complete logits at every step, the threshold remains relative L2 < 1e-4, max abs < 0.01.
AOT additionally tested five rejection behaviors: decoding with no state, incorrect token positions, incorrect byte counts, duplicate states, and decoding when the last state is missing. All were rejected, followed by successful import and generation.

Managed two rounds of source code amount to `b180ba405a5485fbd53f4080592bdd5423973ce8`; AOT and the five rejection check source code amount to `c25d20970bce198646d7befe08622d685150b4da`. Subsequent code formatting does not alter the algorithm. The public summary includes original results SHA256 and binary hashes for both ends: [split-prefill-summary.json](../site/split-prefill-summary.json). Original tensors/logs are only stored on both ends `artifacts/split/<run-id>/`; the hash is not a public download link.

Diagnosing a single `DecodeReadyMs`: Managed 61/32 took 1923.245 ms, AOT 17/8 took 1604.024 ms; the entire process took 2629.277 / 1911.793 ms respectively. **These are not performance benchmarks**: they include tracing, numerical comparison, and file I/O, with no formal warmup or three-round measurement. The old 5bit service is retained, but loading the complete 8bit model causes noticeable system swap, failing resource acceptance. One cannot claim dual-machine acceleration or achieving 2000 token/s based on this.

## Failed Experiments and Current Limitations

- FP32 GEMV + BF16 KV: 17/8 of 8 tokens were identical, but 7/388 strict state/logits checks failed; max relative L2 `2.5934e-4`, max absolute `0.015625`.
- CUDA reduction order experiment: The initial comparison was invalid due to outdated DLLs in the Windows runtime directory; records retained; after fixing, the actual ascending comparison had 9 failures, so it was not adopted.
- Short chunks forced BF16 multiplication input + BF16 KV: 376/388 failures, 8 tokens remained identical, so it was not adopted. This reference itself has also changed and cannot be mixed with the fixed FP32 reference for calculation. Diagnostics show that small FP32 differences crossing BF16 rounding boundaries accumulate; this explains the observed direction, but cannot yet prove that all causes of differences have been excluded.
- Discovered and fixed during the process: Python 3.9 hash API compatibility; MSVC `Platform=x64` leak causing .NET output to another directory; 32-token scratch insufficient to hold 64 row-padded BF16 tile (aborted that round before execution). Added source code hash, runtime library hash, and CUDA ABI checks to reject old artifacts.

Current CUDA is a token-by-token correctness executor, with no tensor-core batch GEMM. Input plus output limit is **2048 tokens**, verifying up to 128 inputs per layer. `--kv bf16` is retained to reproduce target cache paths that failed; it will not be marked as accepted.
The entry point is an independent experimental CLI `split-infer` / `split-worker`, not yet integrated with the original TUI, general scheduler multi-request queue, or leases. Each process/session serves one request, releasing upon failure, without retrying half a request.

[Fixed performance targets](QWEN36-8BIT-TARGETS.md) require BF16 GEMM inputs and KV, 128K inputs, 4096 outputs, and formal resource/performance measurements; no replacements or reductions to these targets are made in this round. The next step is to address consistency of target precision, then implement CUDA batched prefill, expand context, and establish formal dual-machine benchmarks.

## Protocol v2

Both sides handshake using their respective 32-byte random nonces, request UUIDs, fixed model revisions, and KV element widths to derive HMAC-SHA256 keys, proving client/server identity respectively. A FP32/BF16 profile mismatch will cause handshake failure.
Frames consist of a 92-byte header + payload + 32-byte HMAC; containing magic, version, request UUID, incrementing sequence number, type, layer, tensor, token position, token count, byte count, and revision.

The fixed model is `mlx-community/Qwen3.6-35B-A3B-8bit` revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, with a front segment `[0,20)` / back segment `[20,40)`. All data is little-endian, and the layout is uniquely determined by version, handshake profile, and tensor descriptor:

|Message|dtype / shape / layout|
|---|---|
|Start|I32 `[tokens]`; includes whether numerical tracing is enabled|
|Hidden|F32 `[chunk,2048]`, layer 20 input|
|Trace (diagnostic)|F32 `[chunk,20,2048]`|
|Full attention State tensor 0/1|F32 or BF16 K/V `[position,2,256]` as specified by handshake|
|Linear State tensor 0|F32 Delta `[32,128,128]`, value head/value dim/key dim|
|Linear State tensor 1|F32 convolution `[3,8192]`, saving raw projections in chronological order|
|End|No payload, complete token position|

Authentication and integrity protection do not provide encryption, limited to trusted LANs. The random secret is at least 32 characters, passed to the subprocess environment via SSH stdin by the startup script, and not written to command arguments or public logs.
Versions, requests, revisions, profiles, sequence numbers, state positions/sizes, duplicate/missing states, tampering, and broken frames are all rejected. Decode ready is not announced until all states are complete.

## Reproduction

Implementation is still in `codex/split-prefill`; PR #8 has not yet been merged; checking out from main will not include these entries.

1. both ends check out the same SHA from GitHub, confirming a clean working tree. Mac `dotnet build src/DeepInfer.Host -c Release`.
2. Windows activates `scripts/windows/environment.ps1`, executes `scripts/windows/build-prefix.ps1`, then `dotnet build src/DeepInfer.Host -c Release`; when a recovery package is needed, configure using the repository's official NuGet v2.
3. Mac export: `python3 scripts/split/export-prefix.py models/qwen3.6-35b-a3b-8bit-mlx artifacts/split/prefix.bin`. Place the weight artifacts into the Windows checkout's `artifacts/split/`; source code only via GitHub.
4. bundle is 18,415,842,388 bytes, SHA256 `ff5ba3ac11d26d77ea930709ff2a0489df3a00bbdaa2b5e8978a16845a50fe77`. Windows verifies the complete hash before each load.
5. Windows operator check: `python scripts/split/test-cuda.py`.
6. Mac: `python3 scripts/split/run-dual.py --verify --kv fp32 --max-tokens 8`. Uses existing SSH alias `infer-windows` by default; other configurations use `--ssh YOUR_CONFIGURED_ALIAS`.

AOT: Publish separately to Mac `artifacts/split/native-mac-final` (osx-arm64) and Windows `artifacts/split/native-windows-final` (win-x64), then add `--native` to the unified script. Windows publish uses `-p:RestoreConfigFile=scripts/windows/NuGet.v2.config`.

The script records source, artifact hash, PID, stdout/stderr, and exit code; each round uses a unique directory. `--verify` failure writes results with a non-zero exit code; requests with this flag removed will not be marked as Verified.
`HiddenReadyMs` indicates only that the back segment has been computed; `DecodeReadyMs` includes complete state import; `ElapsedMs` until generation ends. All recorded by the same monotonic clock on the Mac.
