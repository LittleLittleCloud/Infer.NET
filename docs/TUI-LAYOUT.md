# TUI Three-Stage Layout and Connection Information

2026-10-22, Issue #13 / PR #14. TUI source code `0ea14381f1ce98498c41fceb68543eafc3f550ac`, backend deployment still `506481445d3670d129ff3d101251f96357efb971`; this round did not replace or restart model processes.

## Changes

Header fixed title, Dashboard / Playground tabs, and connection status; body uses remaining terminal space; footer fixed request statistics and three API entry points. Playground's input box and action row are placed in the footer, no longer moving with conversation content. Dashboard displays worker list and details in two columns when at least 100 columns, and stacks vertically when 80 columns. No borders used, retaining mouse, keyboard, scroll wheel, drafts, multi-turn history, and Auto output.

Connection information is taken from the current coordinator configuration: Chat Completions, Responses display full URL; Claude Code displays base URL and `/v1/messages` path. The interface does not display authentication tokens. Currently, only text API is available, tool calling and Messages token-count are not implemented, so it cannot be claimed to have complete Claude Code or coding agent compatibility.

## Verification

Release compilation zero warnings, zero errors. Independent port, controlled text worker protocol / official SDK / real PTY checks **65 items passed**, added complete endpoint, credential not leaked, bottom input area position, and 80×24, 100×30, 120×40 scaling checks. Original mouse click, scroll wheel paging, streaming output, cancellation, multi-turn history, and Auto checks still passed.

Additionally, perform a read-only terminal check of the 100×30 and 80×24 Dashboard in a real deployment: connections, prefill/decode metrics, and endpoints are all visible. No test inferences were submitted to the production model, and no Windows binaries were replaced. Original terminal logs retain only local `artifacts/chat/layout-current-deployment/` and `layout-three-part-checks.log`.

## User Latency Observations, Not Benchmarks

Telemetry at the time showed that a 1997 input / 1685 output request took 12.157 seconds for prefill, 56.656 tokens/s for decode, and 41.897 seconds for the request; subsequently, a 3737 input / 241 output request took 26.082 seconds for prefill and 122.822 seconds for the request. Current `prefillMs` timing measures from sending Start to completing state handoff, including the Windows frontend, Mac backend, hidden vectors, and KV/Delta state handoff; it is not pure GPU operator time, nor does it include earlier queuing and reset.

Logs also show an anomaly with the first-step decode wall-clock time of 92.542 seconds and GPU timing of 22.877 milliseconds, so not all waiting time can be attributed to prefill. This merely reads existing user request logs and does not include user input text; it does not control for caching, warming up, power supply, or memory pressure, does not constitute a formal throughput benchmark, and does not prove specific paging causes. First-token latency has not yet been fixed and requires further phased timing and memory residency verification.

## Usage

Run `dotnet run --project src/DeepInfer.Tui -c Release` from the repository. The currently open TUI must be exited and re-run; the background serve/agent remains running.
