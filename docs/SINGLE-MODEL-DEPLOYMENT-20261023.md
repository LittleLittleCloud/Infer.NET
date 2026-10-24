# Qwen3.6 Single Model and Dual-Machine Update — 2026-10-23

Related to [Issue #45](https://github.com/LittleLittleCloud/Infer.NET/issues/45). The actual running source code for the Mac coordinator, split-serve, and Windows split-agent has been unified to `58bf95018975297ba6f2f965948e2c5f4944cd95`. Subsequent documentation commits will not change this build version.

## Changes and Configuration

Removed Qwen2.5 3B/7B, TinyStories loader, dedicated download/verification scripts, old model metadata, and 7B launcher. The CLI/demo/default launcher now selects only Qwen3.6-35B-A3B, with default 8bit; checkpoints for the same model 5bit continue to be supported. `synthetic-v1` is retained as a weightless RPC testing tool. Old model experiment reports are explicitly marked as historical records.

Dual-machine operation continues to use fixed model revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, 8bit weights, FP32 KV, SplitWire v4, 20/20 prefill, and full Mac decode. Existing models, credentials, TLS, and compatibility configurations are reused without re-downloading models.

Before the update, the Mac serve was `6a0d55d`, and the coordinator and Windows agent were `eb492b9`. Windows was then using experimental configuration `batch-fp32-fast`, which is not provided in the official source code; this update restores the existing official configuration `batch-fp32-packed`. The two configurations differ only in `cuda-prefill`, with no change in precision.

## Switching and Real Verification

Both ends use independent build directories. After synchronizing the same complete source code SHA via GitHub, they build Native AOT and verify package manifests separately, retaining the old version for paired rollback. After confirming that active requests have ended and the queue is continuously empty, the old Windows agent and Mac LaunchAgents are stopped, and the new package is launched via absolute path; no two complete models are loaded simultaneously.

| Check | Result |
|---|---|
| Mac unit tests | 131 passed |
| Windows unit tests | 130 passed; 1 Metal platform test skipped |
| CUDA operator numerical checks | Passed |
| Native package hashes, versions, and running paths on both hosts | Passed; all match the source SHA above |
| Native AOT RPC / chat protocols on both hosts | 40 / 37 checks passed per host |
| Linux / macOS / Windows CI for the source commit | All passed |
| Real-model deployment acceptance | 15 checks passed: consecutive requests, state reset, SSE, cancellation recovery, EOS, limits, invalid authentication, queue backpressure, and PID stability |
| Paired 17 / 301 input-token requests | Both used actual CUDA → Metal execution; all 16 output tokens matched the previous deployment |
| Online model list | Only Qwen3.6-35B-A3B |
| API requests for the three removed models | All explicitly returned HTTP 400 |
| Native AOT CLI streaming request | Returned the literal `好` (English: “Good”), with natural EOS |
| Windows agent disconnect/restart | Readiness withdrawn and requests returned 503 while disconnected; after both health levels recovered, output tokens matched and the Mac model PID was unchanged |

Original logs, request results, and private configuration/service definition backups are retained locally only; the public summary is at [single-model-deployment-summary.json](../site/single-model-deployment-summary.json). Hashes are used to trace original evidence; local machine paths are not treated as public download links.

## Artifact Hashes

| Artifact | SHA256 |
|---|---|
| Mac deepinfer | `ab5d8ab1b389a2c87bf26276c87d9e0829cce443c41adea657c0f03b65451f23` |
| Metal dylib | `1739ff5dd22a4b158f97d84d39268478a75d5683f4a9ad1a3cd0d1a8f79d4aef` |
| Windows deepinfer.exe | `97c9a6ebd3e948775d8de3b49337da6820a5b094133a8165599134c2c488c041` |
| CUDA DLL | `1ee90b3059fdbee349ca841f1b955488feb624af2d211081fcf45d88bbdf57c1` |

## Hosting and Boundaries

The Mac coordinator / split-serve is hosted by the user's LaunchAgent. Windows retains the original foreground agent mode, maintained by a retained SSH control session; the new connection check confirmed the execution path and PID. No SCM service is installed, and it does not claim to operate independently of the control session or auto-run after a host reboot. Mac logout, dual-end reboots, and Windows SCM have not yet been accepted.

This round of verification tested model cleanup and deployment functions; throughput, HumanEval, or 128K performance were not re-measured, nor were the thresholds in the [fixed acceptance criteria](QWEN36-8BIT-TARGETS.md) changed. Subsequent performance comparisons should use paired measurements with the same configuration; performance gains cannot be inferred from switching configurations in this round.
