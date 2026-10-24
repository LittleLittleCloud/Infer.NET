# Permanent Dual-Machine Deployment Candidate Acceptance

Measurement Date: 2026-10-22 (America/Los_Angeles). Associated with [Issue #9](https://github.com/LittleLittleCloud/Infer.NET/issues/9), [PR #10](https://github.com/LittleLittleCloud/Infer.NET/pull/10), implemented on top of the unmerged PR #8.

Both ends' build and joint test source code are `85c68efe50ceced6bb00a72ce8214e98dd168724`, with both ends checked out cleanly during testing. Subsequent notes/Pages commits only record results and do not change the installation packages for this round.

[Deployment and Operations Guide](https://github.com/LittleLittleCloud/Infer.NET/blob/85c68efe50ceced6bb00a72ce8214e98dd168724/docs/DEPLOYMENT.md) · [Three-Platform CI](https://github.com/LittleLittleCloud/Infer.NET/actions/runs/39895496808)

## This Delivery

Mac coordinator and Metal split-serve run as independent persistent processes; Windows split-agent retains CUDA prefix 20 layer weights, resets state per request, and proactively connects to Mac. The model remains a fixed Qwen3.6-35B-A3B affine 8bit/group64; model revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`. The Mac full model and Windows prefix bundle verify fixed hashes at startup.

The data channel uses fixed-certificate mutual TLS, SplitWire v2 HMAC/request/layout/location validation; local API token authentication, SSE streaming, natural EOS, cancellation, deadline, 1 executors, and 2 queue slots. Configuration files, private keys, and tokens are not committed to Git. A Native AOT version directory with file hashes, macOS LaunchAgent, and Windows SCM install/uninstall scripts are provided, along with upgrade and rollback instructions.

Current capability boundaries: FP32 KV, input plus output totaling at most 2048 tokens, with output capped at 256 tokens; this is not a BF16/128K formal performance delivery. The Mac LaunchAgent is installed and running; the Windows native foreground service has been verified running. SCM installation, boot startup, and service account GPU visibility still require administrator execution for acceptance. Mac availability is not guaranteed upon user logout or sleep. Log persistence is provided; automatic log rotation is not yet implemented.

## Acceptance Evidence

Raw files are stored locally only at `artifacts/deploy/acceptance-6/`; local indexes are not disguised as public downloads. The following are actual runtime results:

| Check | Result |
|---|---|
| readiness advertises 8bit FP32 2K | PASS |
| API rejects wrong authentication | PASS |
| output limit rejected | PASS |
| context overflow rejected without truncation | PASS |
| resident request 0 | PASS |
| resident request 1 | PASS |
| resident request 2 | PASS |
| three requests reset state deterministically | PASS |
| streamed token IDs and text match final result | PASS |
| stream and nonstream agree | PASS |
| client can cancel after first streamed token | PASS |
| next request correct after cancellation | PASS |
| natural EOS terminates before limit | PASS |
| bounded queue provides explicit backpressure | PASS |
| Mac model PID remains unchanged | PASS |

The response for 8 synchronous requests is 3 × HTTP 200, 5 × HTTP 429. The Mac completes tests continuously within the same process without reloading the model. Additionally, repeated startups using the same configuration are rejected by an exclusive lock, and the original worker is unaffected.

Disconnect/reconnect retest: After stopping the Windows agent that passed path validation, Mac `/ready` and inference both return 503; after restarting the agent and waiting for two-level readiness, it returns 200, with 32 output tokens identical to before, and the Mac PID unchanged. The original summary is located locally at `artifacts/deploy/reconnect-summary.json`. The actual Native AOT CLI streaming call emitted the literal `好` (English: “Good”) and ended normally with EOS.

Windows private configuration has added `log-file`; SCM startup records persistent logs; log rotation still requires operations archiving.

Unit tests: 85 items; Linux/macOS/Windows CI each passed build, 25 RPC items, and 25 Native AOT RPC items; these RPC checks use simulated backends. The table above additionally uses two actual devices with real models for execution, which are not mixed in calculations. Operator self-checks after CUDA reset passed.

## Short Request Latency Observations (Not Formal Throughput Benchmarks)

Prompt: `用一句话介绍你自己。`; 17 input tokens, fixed 32 output tokens; no prefix cache, KV/Delta/convolution state cleared each round. Mac M5 Pro 48 GiB / Windows RTX 4090 24 GiB, via existing home router network; 8bit weights, FP32 KV. Process startup executes full-layer warmup, but this delivery lacks complete power/page-sync sampling and does not serve as formal performance acceptance.

| Consecutive Requests | prefill ms | decode token/s |
|---|---:|---:|
| 1 | 3336.6655 | 3.9375 |
| 2 | 1092.8051 | 68.2377 |
| 3 | 869.9283 | 72.0619 |

The wall time for the second token of the first round is 7455.8398 ms; subsequent tokens recover to approximately 13–14 ms; the first round cannot be deleted simply because the latter two rounds are faster. Startup warmup has not eliminated all first-request stalls; the cause is unidentified; without evidence of page swapping, it cannot be asserted as swap.

The token IDs from three rounds of output are consistent, JSON array SHA256 `64062696d8d9d6be39d78dc192aaeb80837d6ed43d99b1becc012037bc6d9936`.

## Failure and Repair Records

- Windows EphemeralKeySet is not suitable for the current Schannel path; DefaultKeySet encountered AccessDenied under SSH identity. After switching to Windows MachineKeySet, real TLS interoperability was achieved. Mac PFX uses DefaultKeySet.
- Windows SSH subprocesses may terminate with the session; attempting to start outside the session via WMI returned AccessDenied, and permissions were not modified. Development verification retained the SSH session, while production hosting uses the SCM service installed by an administrator.
- The first cross-machine TLS state handover failed, with Mac reporting a decryption error; adding only an End ACK still failed. Skipping the SSL write for the zero-length payload of the End frame resulted in a successful real handover, with regression tests retained.
- Sending concurrent acceptance tests immediately after EOS collided with the peer's reconnection window; after waiting for readiness, 3 success/5 rate limiting was observed. The Busy state is now derived from the executor's actual occupation to avoid the finally block of the previous request overwriting the state of the subsequent request.
- The first reconnection probe only waited for worker ready and did not wait for the coordinator heartbeat to recover, causing an assertion failure; subsequent actual requests succeeded and outputs were consistent. Reconnection acceptance should wait for both worker ready and coordinator healthy conditions.

## Artifact Tracking

| Platform File | SHA256 |
|---|---|
| Mac deepinfer | `2c42b00cd9973cf9825144feba0ce9fb7dfcd4f2efa41bb07b0b95f565cb51fc` |
| Mac Metal dylib | `701e01d6ccf99ddc9bf1fb46872658a960bea011c40da5020f4393bad57315ed` |
| Windows deepinfer.exe | `bfbc0f6a979fa52647b3b1d49c177f123640b628b08946f6fd24d3ab6149ad87` |
| Windows CUDA dll | `21bed60a06b4f52d1dc155e4d2c1e7d8a70207ff46e80baf78f352ee6b3ae706` |

Local original summary.json SHA256: `a6c7a6ceb997af33cff3bb02409b5e91b02692e187c2ece40f08694a765f4303`. Each side's package manifest had 9 files pass verification; models, credentials, binaries, and raw logs were not uploaded to standard Git.

## Remaining Acceptance

- Install Windows Service as administrator, verify start/stop, residency after exiting SSH, fault recovery, and GPU visibility after host reboot.
- Long pauses in the first request, automatic log rotation, and stability over longer runtimes.
- BF16 KV and 128K dual-machine numerical, resource, correctness, and performance acceptance still follow the [fixed targets](QWEN36-8BIT-TARGETS.md); thresholds are not changed in this round.

## Retesting After User Startup (2026-10-22, Added)

After user feedback of "started," connections were checked by creating and then closing an SSH session to read Windows agent logs; the agent build remains the aforementioned 85c68ef candidate, the Mac build is the same, and the model PID has not changed. The hashes of the target package exe/CUDA DLL match the table above; the current account cannot read the exe path of the running process, so package hash verification does not replace running process path verification.

The real model acceptance for 15 items passed again in full, with original evidence located locally at `artifacts/deploy/acceptance-scm-1/`. The directory name is merely a runtime identifier and **does not prove that Windows SCM has passed**: `sc.exe queryex/qc/qfailure/qfailureflag InferNetCuda` all returned 1060 (service does not exist), and Get-Service for the specified name also failed to find it; the agent PID in the logs is located in Desktop Session 2. CIM queries also failed due to insufficient permissions, and no attempt was made to change permissions. The specific startup method has been confirmed with the user; service installation, boot-time execution, and SCM recovery remain unaccepted.

Three consecutive 17 input/32 fixed output requests in this round:

| Request | prefill ms | decode token/s |
|---|---:|---:|
| 1 | 66435.5217 | 1.4843 |
| 2 | 1300.6365 | 47.5921 |
| 3 | 806.8059 | 72.2919 |

The first prefill took approximately 66.44 seconds, decode approximately 20.89 seconds, and a single token maximum approximately 20.47 seconds; slow rounds are retained, and the third round is not described as stable performance at approximately 72 t/s. During the long pause, sampling showed the Metal command queue was in the submission system call and the model thread was in `waitUntilCompleted`; only the wait occurring in the Mac Metal path could be located, and the root cause cannot be determined solely based on this sampling. Without complete swap increment and power supply comparisons, this round is not a formal performance experiment, nor can it prove swap. Neither side's worker was restarted due to acceptance.

This round's summary SHA256: `bc19d603077b28c55388309f581b97260c7ff2884c4f9316e1085f5b31b9a3db`. Sampling and service query outputs are retained locally only; public records omit accounts, paths, and network addresses.

## Retesting of Front-end Agent Again (2026-10-22, Added)

The user clarified that they previously started the exe directly; the service installer waiting for credentials has stopped, and no SCM service was created. At the start of this round, the old PID had exited, there were no deepinfer processes on Windows, and Mac was not ready. After confirming no conflicts, the same verified candidate binary was restarted, with the front-end process maintained by the development runner retaining the SSH session; the service hosting method was not changed, and the Mac model process was not restarted.

The real-check for this round of 15 items has again passed completely; the queue 3 successfully /5 429; output state reset, SSE, cancellation recovery, and natural EOS are normal. The original results of three consecutive 17 inputs /32 fixed outputs:

| Request | prefill ms | decode token/s |
|---|---:|---:|
| 1 | 68665.6741 | 0.9847 |
| 2 | 1688.1245 | 30.0251 |
| 3 | 1050.1618 | 72.2015 |

At the start of this round, the Mac was on AC Power. The incremental vm_stat for the entire machine covering readiness wait and full acceptance testing: swap-in 24.637 GiB, swap-out 22.562 GiB (16384 byte pages). This interval includes startup wait and represents whole-machine counts; it cannot be entirely attributed to the model, nor is it a swap measurement for a single request. It proves that significant swapping activity exists, but cannot independently prove that slow requests are entirely caused by swap.

The first prefill still took approximately 68.67 seconds, decode approximately 31.48 seconds; the third recovery was about 72.20 t/s. Retaining three rounds, the conclusion is that functionality is available but the latency of the first request is unstable; we do not claim performance compliance. The Windows foreground agent remains running; Windows Service installation continues pending user action.

Original evidence locally at `artifacts/deploy/acceptance-foreground-2/summary.json`, `foreground-2-environment.json`, and Windows runner logs; raw logs are not publicly available.
summary SHA256: `f4447c82fe98043c84492825408e1c88314c30217a9dcae081ce792500270a5d`.
