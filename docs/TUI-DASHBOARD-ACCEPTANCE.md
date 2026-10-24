# Mouse Dashboard, Auto output, and context boundary acceptance

2026-10-22. Issue #13, PR #14 (superimposed on #12, not merged). Run source code `506481445d3670d129ff3d101251f96357efb971`, Native AOT with the same SHA at both ends, M5 Pro 48 GiB / Windows RTX 4090 24 GiB, fixed model revision `e06a74e6236a60c8367e1a3214e83d8b61b637b0`, 8bit weights, FP32 KV. Windows remains the foreground agent, maintained by the existing SSH controller; Mac is a user LaunchAgent. SCM not installed, host reboot acceptance not performed.

## Changes and purposes

- Default Dashboard, independent Playground; no borders used. Real SGR mouse clicks on Tab/worker/button/input box, scroll wheel for pagination and browsing history, retain Tab/Enter keyboard operations. Switching Tabs retains drafts, history, and currently generating requests.
- Dashboard displays heartbeat, status, attached agent connections, queue, completion count, model quantization/window, and prefill, decode, Metal/KV allocation for recently completed requests. Unknown data is explicitly marked as not reported; the Windows agent does not impersonate an independently registered worker.
- TUI defaults to Auto: do not fix 128/256 output until EOS or remaining model window. Chat/Responses omit budget and also use Auto; Messages retain required max_tokens. Users can still set explicit output limits.
- Remove deployed 2K context / 256 output artificial limits. CUDA attention uses capacity-based workspace allocation, frontend KV expands per request; Metal decode copies existing KV when expanding, retaining Delta/convolution state. SplitWire v3 checks shape/length before allocation, CUDA ABI 3, both ends must be upgraded simultaneously.
- The model's native window remains 262144 tokens. Actual memory, HTTP request size, and deadline are still limited; no claim of infinite context or 262K full model availability.

## Actual test evidence

- Mac and Windows: each 106 item unit test, Native AOT build passed. [Final source code CI](https://github.com/LittleLittleCloud/Infer.NET/actions/runs/39906897249) passed.
- Final Mac AOT + official SDK + actual PTY: total 59 items controlled check passed; protocol check using simulated worker does not count as GPU inference.
- CUDA GEMV/Delta numerical check and 17, 2048, 4097, 8193 tokens × FP32/BF16 8 items attention check passed, attention max absolute error 3.27e-8.
- Metal 38 items hardware check passed: including attention to the last position 262143, and byte retention before and after KV expansion. These are operator/buffer checks, not representing 262K whole model acceptance.
- Real dual-machine: old API 15 items, Chat/Claude Messages/Responses 15 items, real TUI 8 items, context/output 7 items passed.
- 1962 input / 300 fixed output: decoding exceeded original 2048 boundary; compared to the second run with pre-allocated 4096 capacity, all 300 token IDs are consistent. Total time for both runs 19.142 / 17.401 seconds.
- 3012 input / 32 output: CUDA prefill, cross-host state import and Metal decode completed, total time 19.189 seconds. Subsequent short request token IDs are consistent with the old deployment baseline. Auto short answer natural EOS.
- After disconnecting the agent, the serve heartbeat is still healthy but the status changes to WaitingForAgent, unable to accept requests, and the model is removed from the available list; after reconnecting, the output token IDs remain unchanged, and the Mac model PID remains unchanged.

These are functional tests, not controlling power supply, warm-up, and repetition count according to formal performance thresholds; not using time consumption as a new speed baseline. The current home network is still Mac Wi-Fi → router → Windows 1GbE. KV still goes through CUDA D2H, host buffer, TLS/TCP, and Metal shared-buffer copy, not GPUDirect.

## Failures and Fixes

Candidate `317e17f` exposed heartbeat sampling issues after completing long requests: the disposed TcpClient.Client was null, causing the heartbeat task to exit and the coordinator lease to expire. `5064814` added safe connection probing and regression tests, rebuilt and deployed both ends, repeated long requests and all formal functional regressions passed. The earliest test obtained 503 during the Warming phase; subsequently, tests were performed only after waiting for both worker and coordinator levels to be ready. Failure logs are retained, not hidden as successful rounds.

## Reproduction

First deploy matching v3 packages for both ends; old packages kept in pairs for rollback. Use `scripts/deploy/acceptance.py`, `scripts/chat-acceptance.py`, and `scripts/chat-integration.py --sdk --tui`. The first two test real deployments, the last one uses an independent port to simulate a worker to check protocol and mouse behavior.

Expansion cases require a newly started serve: send `prompt = " a" * 1950`, `maxTokens = 300`, `benchmarkIgnoreEos = true`, `profile = true` to `/v1/infer`, model `qwen3.6-35b-a3b`, backend `metal`; execute twice consecutively and compare tokenIds. Then verify long prefill with `" a" * 3000`, 32 output; subsequently rerun short baseline and Chat requests with omitted output budget. Request fields intentionally fix EOS behavior, so fixed output tests cannot be treated as natural stop tests.

[Public summary and artifact hashes](../site/dashboard-acceptance-summary.json) save capacity, timing, token ID hashes, and check counts. Complete raw logs are only retained in the local ignore directory `artifacts/chat/dashboard-*`, not disguised as public downloads. Real terminal ANSI/text records are in `dashboard-real-tui-final`, controlled PTY in `tui-session.ansi`.

Next steps remain BF16 dual-machine precision, 128K formal performance acceptance, and Windows SCM hosting; this round did not modify the established performance goals.
