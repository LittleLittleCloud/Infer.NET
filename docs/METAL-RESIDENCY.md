# Metal Idle Residency A/B

2026-10-22; [PR #15](https://github.com/LittleLittleCloud/Infer.NET/pull/15), associated with #9 / #3. Fixed 8bit model and FP32 KV, both ends' Native AOT source code are `b69f90e601e2e0f434bd153298a4475b3f62774d`. M5 Pro 48 GiB + RTX 4090, existing home LAN; Mac plugged in, High Power not re-verified.

## Results

Same binary, only changing `metal-residency=on/off`, each configuration warmed up once, idle 0/2/10/60 seconds, each formally three times. 24 input, natural EOS 13 output, all tokens consistent with previous diagnostics.

|Idle|Switch|Median First Step Decode (ms)|Median Prefill (ms)|Median Client Total (s)|Max Machine Swap-out|
|---:|---|---:|---:|---:|---:|
|0s|off|21.019|734.673|0.926|0 B|
|0s|on|21.594|800.701|0.991|0 B|
|2s|off|198.924|1385.449|1.753|0 B|
|2s|on|27.486|1181.659|1.376|0 B|
|10s|off|200.196|1358.970|1.732|0 B|
|10s|on|24.358|1173.629|1.372|0 B|
|60s|off|203.298|1432.194|1.803|0 B|
|60s|on|26.296|1513.007|1.710|0 B|

Enabling it eliminated the approximately 0.2 second first-step wait that stably reproduced this round; one cannot directly divide the 16–84 second pauses under previous memory pressure by the low-pressure results of this round to claim thousands of times speedup. The ON→OFF order was not randomized, all measurements retained. Neither group of formal requests had new swap-outs.

Short requests still have a fixed dual-machine state handover cost. Each 24 token request returns 33,423,360 bytes of front-end KV/Delta/convolution state; this stage log includes waiting and transmission, cannot be treated as pure CUDA time. Long context 37 tokens/s, 128K prefill 1600/2000 tokens/s, and total 200/180 second target were not accepted.

## Implementation and Failure Records

- Created a budgeted Metal residency set for original weight buffers, no weight copies, no dummy GPU computation. Registered 36,825,666,560 bytes; budget 39,664,025,600 bytes, calculated by recommended working set/physical memory upper limit minus 512 MiB. Here only weights are managed, not guaranteeing all long context workspaces can reside.
- Graph copy matrix descriptors share lease; deregistration only occurs when the last descriptor is released. Residency renewed every 1000ms requests, thread stopped and released on exit.
- 0bcbbb6 initial deployment invalid: managed `Environment.SetEnvironmentVariable` did not update native `getenv` on Unix. b69f90e changed to explicit C ABI parameters, cross C#/Metal test verified actual registration and output consistency.
- Another unenabled diagnostic appeared during compilation on both ends with about 80 second first-step wait and massive machine swapping, recorded `idle-latency-20261023T064533Z`; background compilation interference existed, not included in A/B.
- Historical documentation on deployment `Profile`'s matrix=0 / sequential Delta inference was inaccurate. Assignment only changed managed copies; without external native override, actual native default matrix=true, chunked Delta, FP32 GEMM/KV, 2048 scratch; C# and network calls still 32 token. Independent old split script via startup environment settings parameters unaffected by this issue.

## Verification and Reproduction

Mac 107 item unit test (including real Metal residency switch, same GEMV output, final release cleanup); Windows 107 item test returned passed, where Mac-specific test directly returned on Windows; CUDA GEMV/Delta/attention checks passed; both ends' Native AOT build and immutable package generation successful. Native lifecycle/budget checks previously seven items passed.

Run `scripts/benchmark-idle-latency.py --help` to view parameters. First switch residency in deployment config and restart same version, then sequentially measure after Ready, keeping only one model instance per device. Probes do not restart services, nor output token file contents.

Public step-by-step results and original file hashes see [residency-summary.json](../site/residency-summary.json). Original events, stage logs, power supply and version records only local `artifacts/dual-host/residency-{on,off}-*`, hash is not a public download link.

References: [Microsoft hosted environment variables documentation](https://learn.microsoft.com/dotnet/api/system.environment.setenvironmentvariable), [Apple residency sets](https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets).
