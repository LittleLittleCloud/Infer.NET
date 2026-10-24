# Infer.NET Dual-Machine Deployment

This entry point is for Qwen3.6-35B-A3B **8bit weights, FP32 KV**; candidate versions remove the artificial upper limits of 2048 context / 256 output. The model window is input plus output 262144 tokens; actual requests are still constrained by memory and deadline. Long-window full-model availability and throughput require independent empirical testing and are not guaranteed by upper-limit declarations. It provides persistent execution and service lifecycle; BF16, 128K, and performance targets still require separate acceptance testing. The code is located on the deployment implementation branch and has not been automatically merged into the upstream PR.

## Model Scope

The project retains only Qwen3.6-35B-A3B; this dual-machine deployment continues to use fixed revision 8bit weights and FP32 KV. The single-machine CLI/demo and launcher also default to selecting the 8bit checkpoint for this model. Qwen2.5 3B/7B, TinyStories loaders, downloaders, and dedicated validation entry points have been removed. Existing Qwen3.6 models, certificates, tokens, and compatible configurations can be reused; weights or credentials are not re-downloaded or regenerated during updates. Related to [Issue #45](https://github.com/LittleLittleCloud/Infer.NET/issues/45). The actual running versions, artifact hashes, and acceptance results for both ends this time are seen in the [Unique Model Deployment Record](SINGLE-MODEL-DEPLOYMENT-20261023.md).

## Processes and Network

- Mac coordinator: loopback HTTP 5200, unified entry for CLI/TUI, does not load models.
- Mac `split-serve`: loopback HTTP 5201 + LAN TCP 5210. Holds the unique complete model; responsible for the back 20 layers' prefill, receiving the front 20 layers' state, and all decode.
- Windows `split-agent`: actively connects to Mac 5210, front 20 layer weights resident. No need to open new Windows inbound ports.
- TCP 5210 enforces bidirectional TLS + fixed certificate SHA256 verification; internally continues to use SplitWire v3 identity/layout/location/sequence/HMAC checks. Client per-token output goes via local HTTP SSE. v3 is incompatible with old v2, and the CUDA native ABI has also upgraded to 3; both ends must switch together during upgrades, with old packages retained for paired rollback.
- The API is open only to the Mac localhost by default; other devices cannot directly treat it as a secure public API. The cross-machine data channel is already encrypted; the HTTP remote TLS gateway is not within the scope of this version's installer.

Only one deployment worker per machine is allowed to hold the exclusive lock for the corresponding configuration. Do not run old demos, single-machine models, or another complete model simultaneously for formal performance acceptance. The Qwen3.6 default weights for historical demos have been changed to 8bit.

## Build, Configuration, Packaging

Mac：

```sh
dotnet publish src/DeepInfer.Host -c Release -r osx-arm64 -o artifacts/deploy/build-mac
python3 scripts/deploy/package.py --publish-dir artifacts/deploy/build-mac --rid osx-arm64 --output artifacts/deploy/releases/mac-VERSION
python3 scripts/deploy/verify-package.py artifacts/deploy/releases/mac-VERSION
```

Windows standard PowerShell:

```powershell
& scripts\windows\build-deployment.ps1
python scripts\deploy\package.py --publish-dir artifacts\deploy\build-windows --rid win-x64 --output artifacts\deploy\releases\windows-VERSION
python scripts\deploy\verify-package.py artifacts\deploy\releases\windows-VERSION
```

Packaging requires clean source code, and the source SHA of `deepinfer version` must match the current checkout. The manifest includes file hashes, platform, model revision, KV precision, and capability upper limits. The program runtime does not require SDK, Python, or nvcc; configuration generation/installation helper scripts use Python/OpenSSL or PowerShell respectively. GPU drivers, system frameworks, and MSVC system runtime dependencies must be pre-checked on the target machine.

Use `scripts/deploy/prepare.py --help` to generate an independent private configuration directory, explicitly passing the Mac LAN address, model locations for both ends, and the Windows configuration target directory. This tool generates two sets of fixed certificates, random cluster tokens, and configurations; certificate validity is 365 days; when replacing certificates, drain requests and update pins in pairs. New directories use 0700, files 0600; the Windows configuration directory should only allow access by dedicated accounts, SYSTEM, and Administrators.

Only securely transmit the generated `windows/` content to the specified Windows configuration directory; do not transmit Mac private keys, and do not put configurations/credentials into GitHub. Source code still synchronizes via GitHub, models are verified separately; Windows prefix bundles and the actual files used by the Mac complete model are verified against fixed hashes at startup.

## Direct Startup and Usage

First start three independent processes (paths use the generated absolute configuration paths):

```sh
deepinfer coordinator --config /PATH/mac/coordinator.json
deepinfer split-serve --config /PATH/mac/worker.json
```

Windows：

```powershell
.\deepinfer.exe split-agent --config C:\PATH\worker.json
```

The standard foreground mode requires retaining the startup terminal; SSH subprocesses may exit with the session. `scripts/windows/run-deployment.ps1` is a development verification tool for retaining SSH sessions and must never replace Windows Services. This standard account lacks the permission to create processes outside a session via WMI, and the development process does not modify this permission.

User request:

```sh
deepinfer infer --endpoint http://127.0.0.1:5200 \
  --token-file /PATH/mac/token.txt --stream \
  --model qwen3.6-35b-a3b --prompt "你好" --max-tokens 128
```

`--stream` prints tokens in real-time, finally displaying the finish reason and decode throughput; without this parameter, the full response is returned. Natural generation ends upon encountering EOS; the test API can explicitly set `benchmarkIgnoreEos=true`. The TUI connects to the same 5200 coordinator by default, displaying multi-turn conversations via streaming using Microsoft.Extensions.AI's IChatClient; when launched without parameters from the repository root, it reads the existing deployment configuration, or credentials can be specified using `DEEPINFER_TOKEN_FILE` or `DEEPINFER_TOKEN`. The TUI itself does not load models. For text interfaces compatible with Chat Completions, Messages, and Responses, see the [HTTP chat API guide](CHAT-API.md).

`GET /health` shows Loading/Warming/Ready/Busy/Faulted status, PID, build, model, and capacity. `GET /ready` requires `X-DeepInfer-Token`; it only returns 200 after the model has completed actual warming and the Windows connection has been authenticated; loading completion does not equal Ready. The first warm-up executes all 40 layers and 32 decode steps, then resets the request state, with no prefix cache hit.

## System Hosting

Mac uses user LaunchAgent:

```sh
python3 /RELEASE/deploy/install-macos.py --executable /RELEASE/deepinfer --config-dir /PATH/mac
```

Starts after login, restarts on abnormal exit; logging out or suspending is not an assurance of unattended availability. To disable, use the same parameters with `--uninstall`, operating only on `com.infernet.coordinator` and `com.infernet.metal`. Logs are in the configuration directory `logs/`; they should be archived according to the retention period, and the current installation script does not perform automatic log rotation.

When updating only compatible coordinators, `--service coordinator` can be passed; this option applies to both old package uninstallation and new package installation, preserving Metal workers and model processes; default is `--service both`. Before switching, requests must still be drained, the old installation path saved, and compatibility between old and new protocols verified.

Windows uses a dedicated standard local account. First, stop the development foreground agent, then execute in **Administrator PowerShell**:

```powershell
& C:\RELEASE\deploy\install-windows.ps1 -Executable C:\RELEASE\deepinfer.exe -Config C:\PATH\worker.json
```

The installation script interactively requests service account credentials on the local machine and does not accept passwords from chat. The account requires the 'Log on as a service' permission and appropriate access rights to model/configuration/program directories; if organizational policies prohibit this, it must be handled by an administrator, and the script does not modify drivers or firewalls. The service name is `InferNetCuda`, set to automatic start, with retries every 10/30/60 seconds on failure. Deletion uses the same command with `-Uninstall`. Windows Services cancel tasks via the .NET SCM lifecycle; GPU visibility and automatic recovery after system installation or reboot must be verified under the actual service account.

## Runtime and Upgrade Boundaries

Single GPU executor, defaulting to a maximum of 2 queued requests; exceeding this returns 429. The default 300 second deadline includes queuing, configurable to 1..1800 seconds; client disconnection cancels the current request. Cancellation is cooperative; submitted single-GPU work cannot be instantly interrupted. Each request resets KV/Delta/convolution state, blocking decode if transmission is missing, duplicated, or misplaced. GPU errors exit the process, handing it over to the system for restart; requests with partially output tokens are not automatically replayed.

Upgrade: Stop incoming traffic, wait for requests to complete, stop the old worker, verify the new package, switch absolute paths, restart, and wait for ready; on failure, revert to the old package and old compatible configuration. The split protocol, model revision, KV precision, and ABI must match on both ends. Mac does not load old and new full models in parallel; this version accepts maintenance windows and does not promise zero-downtime rolling upgrades.

The API returns prefill/decode/per-token profiles, Metal allocation, GPU command time, and submission counts; `--log-file` can write output to a specified persistent log. System swap and power supply are still sampled by experimental monitoring, not mistaking existing total swap for this session's new paging.

Automatic acceptance entry:

```sh
python3 scripts/deploy/acceptance.py --token-file /PATH/mac/token.txt --output artifacts/deploy/acceptance-UNIQUE
```

It verifies real continuous requests, output determinism, consistency between streaming text and tokens, recovery after cancellation, natural EOS, capacity rejection, queue backpressure, and PID stability. Agent disconnection/restart tests, service installation/restart tests, and the full performance threshold for `docs/QWEN36-8BIT-TARGETS.md` must be executed separately.

## Metal Weight Residency Candidates

`split-serve` enables `metal-residency=on` by default, establishing a Metal residency set for loaded weights, with a budget not exceeding the smaller value of the suggested working set and physical memory 80% minus 512 MiB. Weights exceeding the budget will fail explicitly, not silently falling back; KV/workspace not yet pinned for residency. Residency requests are submitted after weight upload completes, before warm-up and Ready. Default is `residency-refresh-ms=1000`; residency is extended when idle, without submitting empty GPU inference; registration is revoked when the last graph/matrix reference for resources is released.

`metal-residency=off` is a control switch for the same candidate version; `residency-refresh-ms=0` can test behavior with only one application. These configurations only affect the application, not the system wired limit. `RESIDENCY` logs budget and registration size, but does not guarantee the OS will satisfy residency requests; real requests and paging after idle must still be tested.

New `PREFILL` logs record reset, hidden wait, Mac suffix, status wait/import, and handover bytes separately; hidden wait includes remote computation and transmission, not referred to as pure CUDA time. `METAL_WAIT` records commit, wait, and GPU time for commands exceeding 100 ms. Logs are for localization and do not change existing API timing definitions.

Resident parameters are passed from C# to Metal via an explicit C ABI; do not rely on `Environment.SetEnvironmentVariable`: the managed environment copy on Unix does not update the native `getenv`. Consequently, the initial launch of 0bcbbb6 did not enable residency, remaining a failed experiment and not constituting residency performance results. Only the native `RESIDENCY` logs and cross-boundary tests proved that this option took effect. Other environment parameters for historical `Profile` also require item-by-item review; do not treat managed assignments as evidence of native activation.

### Small-request routing and CUDA readback

`short-prefill-tokens` defaults to `128` (range `0..262144`). The count includes the complete chat template and history. Small requests run full prefill and decode in the existing resident Mac model; larger requests retain the CUDA-prefix / Metal-suffix path. Set it to `0` to force split execution for comparisons. Set `short-prefill-tokens=262144` to run every accepted request entirely on the resident Mac Metal model, including long structured chat and tool calls, without a CUDA agent. This selects the existing full-model prefill implementation; it does not change weights, KV precision, scratch size or the default routing threshold. The configured model window still bounds input plus output. A worker in this mode can report `ReadyLocal` without an attached agent; every completed request must report `executionPath=metal-local`.

This is an explicit routing policy, not prefix caching; each request still resets its state. RPC results and latest worker metrics expose `executionPath` (`metal-local` or `cuda-metal-split`), also shown in the TUI.

`cuda-readback` defaults to `token`. The `batch` candidate is opt-in: measured 512-token native-prefix A/B was 3.55% slower with FP32 KV and 3.95% slower with BF16 KV, despite exact hidden/state agreement. Batch mode writes each final-layer hidden vector into a 256 KiB device buffer and performs one host readback per chunk (up to 32 tokens). It does not change arithmetic, precision or wire layout, and it is not batched GEMM. Native CUDA ABI is now 5; rebuild both managed/native components. Use `scripts/split/benchmark-readback.py` for exact hidden/state A/B and a synthetic-token prefix-only timing check, with the deployed agent stopped first so there is only one CUDA model.


### Layer-major CUDA prefill

`cuda-prefill=batch-fp32` (default) processes all tokens of each 32-token chunk layer by layer. Projections/experts keep the original FP32 GEMV arithmetic; convolution, attention and routing are batched, and Delta recurrence keeps each state row in registers across the chunk. This is distinct from the experimental Tensor Core BF16 GEMM primitive. `cuda-prefill=serial` retains the original token-major path for A/B. The agent logs its actual policy on readiness. Source fingerprints now cover all CUDA include files as well as the main translation unit.

A standalone pinned-model test compared 131 synthetic token IDs across irregular chunk sizes, all 20 layer traces and all 40 prefix state tensors: byte-exact and finite for FP32 and BF16 KV. The 512-token prefix-only median improved 3.44x (FP32 KV) / 3.42x (BF16 KV). These tests do not pass cross-host BF16 accuracy or 128K end-to-end performance gates. Use `scripts/split/benchmark-prefix-chunk.py` with the deployed agent stopped to avoid loading another model on that GPU.

When the CUDA agent is disconnected, an enabled short-request route remains available as `ReadyLocal`; longer requests return a bounded peer-unavailable error. `short-prefill-tokens=0` retains the original all-split readiness behavior. No long request silently changes route.

`suffix-chunk-tokens=128` is now the selected default (32 and 512 remain available for A/B). CUDA still sends 32-token hidden chunks; Metal combines them before executing the suffix. Native scratch is explicitly configured to 512 tokens rather than relying on managed environment changes reaching libc. Rebuild/package the updated Metal bridge with the managed executable. The selected 128-token policy passed 234/714-token layer/logit checks and repeated 8K end-to-end comparisons; 512 did not improve the complete pipeline. See `SUFFIX-BATCHING.md`. These results do not pass BF16/128K targets.

`cuda-send-overlap=on` is selected after same-package A/B (off remains available). A bounded two-frame queue overlaps one CUDA chunk producer with one authenticated sender. All exits join the producer before reset/disposal; no background retry. Final KV/recurrent state transfer still happens after hidden production. See `PREFIX-SEND-OVERLAP.md` for the 4.27% 8K prefill gain and real mid-prefill cancellation/recovery check.

`kv-transfer=final|incremental` selects final full KV handoff or experimental per-chunk lossless KV transfer. Default remains `final` pending end-to-end A/B. The incremental mode sends ten attention KV slices after each 32-token hidden chunk; Metal coalesces and imports them after each suffix submission. The thirty DeltaNet/convolution states still transfer last. Decode remains blocked until every tensor is complete. The two-item send queue now holds complete bounded chunk bundles, and every failure joins the GPU producer before reset. This change requires **both hosts rebuilt with split protocol v4 and CUDA prefix ABI 6**; v3 peers fail authentication. Deploy the pair together, retain the previous immutable pair for rollback. No precision or model change.

`decode-attention=legacy|fused-fp32` is an experimental independent decode-only selector; default `legacy`. The new kernel computes one query and shares its KV tiles across eight GQA heads, with 128-key partitions and FP32 storage/arithmetic. No BF16 KV or GEMM acceptance is implied. Use only after same-package numeric and whole-model A/B validation; native operator timings are not inference speed. This requires the matching Metal bridge exposing `di_graph_decode_fused_fp32`.
