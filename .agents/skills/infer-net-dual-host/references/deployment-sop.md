# Infer.NET Deployment, Update, and Rollback SOP

Applies when users request deployment, upgrade, restart, or rollback of Infer.NET dual-machine services. When only asked to write the SOP, maintain only the documentation; do not stop services, install system components, or start new deployments based on this. When the user says they will install it themselves and provide feedback later, wait for the feedback before performing acceptance dependent on that installation.

## Source of Facts and Roles

Before execution, read the `docs/DEPLOYMENT.md`, recent `docs/DEPLOYMENT-ACCEPTANCE.md`, and the actual `scripts/deploy/` installation, packaging, and acceptance scripts from the target checkout. Capabilities, model revisions, precision, capacity, and test results are based on the target version's source code, package manifest, and actual measurements; do not treat historical candidate packages or this SOP as the latest running state.

Retain user-selected command names `split-serve` / `split-agent`:

| Component | Location | Responsibility and Integration |
|---|---|---|
| coordinator | Mac | CLI/TUI request entry point; registration, selection, and forwarding, does not load models |
| split-serve | Mac | Full model resident; prefill for 20 layers, import frontend state, full decode; registers with coordinator |
| split-agent | Windows | Front 20 layer weights resident; actively connects to serve's data port, executes frontend prefill, and sends hidden/KV/Delta/convolutional states |

The currently accepted topology is a pair of serve/agent, with serve having a single executor and limited queuing. The agent does not register directly with the coordinator; what the coordinator sees is a dual-machine inference worker provided by serve. The default configuration is coordinator local HTTP 5200, serve local HTTP 5201, and serve LAN data port 5210; actual addresses are read from private configuration. Windows actively connects to Mac, so there is no need to add Windows inbound listening for this purpose.

One agent connects to one serve at a time; do not interpret multiple TCP connections, registry capacity, or multiple processes as indicating multiple agents coordinating scheduling. Multiple serves require independent IDs, ports, locks, configurations, and model memory, and require additional acceptance testing; the current Mac cannot share the same model instance by adding processes. serve/agent are computing and connection roles, while foreground, LaunchAgent, and Windows Service are process hosting methods.

## Pre-deployment Records

Establish a unique local running directory, for example `artifacts/deploy/<run-id>/`, and record:

- Absolute paths of packages on both old and new ends, source SHA, platform file hashes; actual process paths, PIDs, hosting methods, service accounts, and startup commands.
- Model paths/revisions/hashes, private configuration locations, interface addresses, protocols/native ABIs, precision, and capacity; do not record tokens, passwords, or private key contents.
- Current health, active requests and queues, GPU usage, old version recovery methods, and allowed maintenance windows. Save restricted-permission backups of old configurations and service definitions.

Check for changes in both ends' repositories to avoid overwriting user work. Check existing service/process identities; do not stop processes based on historical PIDs. Background shells or foreground agents with retained SSH are not equivalent to Windows Services.

Continue with ordinary builds, package preparation, and verification within existing permissions as authorized. For SCM installation/removal requiring administrator privileges, first complete the reviewable new/old packages, configurations, and specific operation commands; only proceed to the dependent downtime switch after confirming the administrator steps are executable. Account credentials are entered only on the local Windows machine. Do not use WMI, scheduled tasks, or expanded account privileges to bypass permissions; required permissions are handled by the administrator.

## Prepare Candidate Versions

Code, configuration templates, documentation, and experiment record modifications required for deployment and site publishing must also be submitted to a feature branch and a PR created first. Requesting "update deployment/site" does not authorize direct pushes to `main`; the exception is only when the user explicitly requests a direct push to `main`. When a mainline version is needed, wait for user authorization to merge the PR, then build or publish from the actual merged SHA.

1. clearly defines the target complete SHA, synchronizes source code on both ends via GitHub, and verifies remote, HEAD, and clean status. Unmerged PRs can serve as explicitly marked candidate versions; do not automatically merge PRs or treat main branch page publishing as a running code upgrade.
2. builds Native AOT on both ends separately. Mac uses the publish entry in the deployment documentation; Windows uses `scripts/windows/build-deployment.ps1`, first checking its current output path and environmental prerequisites. Build tools may write to a fixed staging directory: if the running foreground program comes from that directory, use an independent checkout/staging instead, and do not overwrite running exe, DLL, or dylib files.
3. executes corresponding unit tests, RPC/Native AOT checks, and necessary native numerical checks. Performance or model acceptance on the same GPU must not compete with online requests; checks requiring exclusive GPU access should be placed in maintenance windows; do not treat successful compilation as successful inference.
4. uses `scripts/deploy/package.py` to output a brand new version directory, then executes `verify-package.py`. Verify the manifest source against `deepinfer version`, cross-end model/protocol/precision/capacity/ABI compatibility, and platform native libraries. Manifest declarations must also align with target code capabilities.
5. retains recoverable old packages. Commits that only append documentation do not require replacing running binaries; record the actual build SHA of the package, and do not relabel old packages as new source code versions.

Packages, models, and configurations are managed separately. Models with the same revision are reused and verified, eliminating the need to download them every time; restarting the worker still requires reloading the model.

## First Deployment

1. uses `scripts/deploy/prepare.py --help` to verify parameters, then generates a new private configuration directory. Only transmit Windows-side configurations/certificates to Windows; keep Mac private keys on Mac. Restrict configuration ACLs, maintaining token, certificate pinning, and request data channel authentication. Standard updates reuse compatible configurations without regenerating cluster credentials.
2. configures persistent logging. The Mac installer uses `logs/` from the configuration directory; the Windows service sets `log-file` in a restricted directory within `worker.json`, ensuring the service account has write permissions. Check remaining disk space and archival methods; do not claim automatic rotation unless the code actually implements it.
3. stops conflicting instances that have passed authentication. Start/host the Mac coordinator and serve, then start the Windows agent. The Windows configuration's endpoint points to the Mac LAN data address; tasks are executed only after fixed bidirectional certificates and cluster identity verification pass.
4. Mac installs the user LaunchAgent using `install-macos.py` from the deployment documentation. Windows administrators install the SCM service using `install-windows.ps1`; the dedicated standard account must have directory access and 'Log on as a service' permissions. A successful installer return still requires the following real acceptance checks.
5. If it can only run in the foreground, clearly mark it as 'Development foreground run', retaining its session/PID/logs; do not describe it as having completed boot startup or being system-hosted detached from SSH.

## Update: Prepare → Drain → Switch → Accept

The current upgrade is a downtime switch within the maintenance window, not a hot load or rolling upgrade. `git pull` will not change already running programs.

1. prepares new packages for both sides, configuration compatibility checks, old packages, and recovery commands before stopping the old service. Steps involving service account interactions are also implemented in advance.
2. stops clients from submitting new requests, waiting for active and queued requests to finish. There is currently no implemented generic drain API; do not fabricate drain parameters, nor assert the queue is empty based solely on a momentary non-Busy state. Verify using known client completion status, request logs, and health status. If the maintenance period is exceeded, explain the remaining requests, handle them according to existing authorizations, retain partial output, and do not automatically replay.
3. stops the old Windows agent and Mac serve/coordinator. In hosted mode, disable/uninstall via the corresponding manager first to avoid automatic restart of old processes; in foreground mode, only stop the task PID at the verified path. Confirm processes have exited and GPU resources are released before loading new models.
4. updates the startup definition to point to the absolute path of the new package. The current installer does not overwrite existing definitions in place: Mac first runs the installer `--uninstall` with the old path/configuration, then installs the new path; Windows first runs the old installer `-Uninstall` as administrator, confirms `InferNetCuda` is deleted, then installs with the new package installer. Uninstalling services does not delete models or private configurations. Re-read the script before calling; parameters are based on the current version.
5. starts the new Mac coordinator/serve and Windows agent, waiting for loading, warming up, authentication connection, and registration. Do not load both old and new full models simultaneously, nor start incompatible joint requests after updating only one side.
6. Proceed to acceptance as per the next section; if passed, resume user requests; if failed, stop the current switch, retain logs, and handle according to the rollback process. Do not infinitely loop reinstallations or restarts to mask failures.

When installation fails midway, first check which service definitions have been created and which processes are still alive; only clean up the partial installation of this session. When Windows service deletion is marked as incomplete, wait for related handles to be released and check status; do not create services in parallel repeatedly.

## Acceptance and Stop Conditions

Record pass, fail, or not executed item by item according to the actual scope:

- Running version: The actual exe/native library hashes on both sides match the expected package, and process paths are correct; no old model worker competition. Confirm build, precision, model, and capacity; do not rely solely on source code HEAD.
- Ready: serve `/health` is readable, auth `/ready` returns success; coordinator `/v1/workers` corresponding ID is also healthy. There may be a time difference in heartbeats between the two; do not assume the entry has recovered just because the worker is ready.
- Requests: Run `scripts/deploy/acceptance.py`, select a new evidence directory, verify continuous requests, status reset, SSE, natural EOS, cancellation recovery, limit/exceed auth, and rate limiting; additionally perform corresponding entry checks for CLI/TUI. Do not use fixed old test quantities instead of viewing results.
- Connection recovery: Stop the current Windows agent within the maintenance window, confirm serve revokes ready and requests explicitly fail; restart and wait for two-level health recovery, then repeat deterministic input to verify token consistency and unchanged Mac model PID. Retain the first error and retry results on failure.
- Hosted: Check LaunchAgent/SCM status, real processes, and logs. After service installation, close the install/SSH control session, then check PID, readiness, and real requests with a new connection; additionally verify service stop/start and fault recovery. Successful service registration alone does not prove GPU availability in that account's session.
- Host reboot, exiting Mac login, or fault injection will affect other work; execute only within the scope authorized by the user; if not executed, mark as 'Not Tested'. LaunchAgent starts only in the user login session, do not claim it as an unattended boot service.

Do not resume user traffic when health/auth/model compatibility fails, native errors occur, or output does not match the baseline; do not lower precision or disable TLS to 'pass'. Startup warming does not guarantee first-request latency; retain first-round anomalies, and formal performance and long-context thresholds are accepted separately per `docs/QWEN36-8BIT-TARGETS.md`.

## Rollback

Stop new version access and drain; disable new service definitions, confirm processes have exited, restore old version-compatible private configurations and old absolute startup paths, rolling back both sides together. Certificate rotation involves both parties' pins; do not restore configuration on only one side. Reuse old models but verify compatibility, re-wait for two-level health, run real short requests, then resume access.

If the old package is missing, old configuration cannot be restored, or rollback acceptance fails, keep the service stopped and report specific reasons and evidence; do not continue guessing versions or overwriting files. Retain the failed new package and logs for diagnosis.

## Recording and Maintenance

Use existing GitHub Issue/PR to record target and actual running versions, hashes of artifacts on both sides, installation method, acceptance evidence, failures/rollbacks, and incomplete items. Update Pages upon successful switch or important state changes; only modify SOPs, do not fabricate new deployment experiments. Public records exclude private network addresses, credentials, and account private directories.

This SOP is maintained alongside the warehouse skill version; it also synchronizes the installed `infer-net-dual-host` skill entry, UI metadata, and this reference on this machine. When only skill modifications are made, perform skill validation and link checks; do not restart real services for text verification.
