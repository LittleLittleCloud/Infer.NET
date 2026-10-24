---
name: infer-net-dual-host
description: "Use GitHub to synchronize Infer.NET dual-host code, manage Issues, display project progress and experiment notes via GitHub Pages, and conduct development and verification of Mac Metal and Windows RTX 4090 inference via SSH. This is used for dual-host onboarding, implementation, tuning, deployment updates and rollbacks, and progress/experiment record maintenance; it is not for general remote operations."
---

# Infer.NET Dual-Host Development

## Scope and Sources of Truth

Maintain the current Mac session as the controller, operating Windows via user-configured SSH. Standard dual-host development does not require starting Codex separately on Windows, nor does it automatically create new sessions or delegate agents.
The user's current task determines the scope of operation; this skill is not a permanent authorization for all management operations on the remote machine.

The known development configuration is M5 Pro 20-core GPU / 48 GiB and Windows RTX 4090 / 24 GiB. First, verify the current devices; do not treat 72 GiB as a unified VRAM pool.
The preferred repository location is `/Users/littlecloud/repos/Infer.NET`; historical session working directories may still be the migrated `Documents/ChatGPT/Infer.NET`.
Explicitly use valid working directories, read applicable AGENTS.md files, and check the change status on both ends. If a path is invalid, first look for existing repositories; do not create empty repositories as substitutes.

Read within the repository according to tasks:

- `docs/EXPERIMENT-SOP.md`: User-approved experiment proposals, manual approvals, code/result PRs, evaluation, and baseline update processes; each experiment still requires separate approval.

- `docs/QWEN36-8BIT-TARGETS.md`: The sole source for fixed performance, accuracy, timing, correctness, and resource thresholds; must be read before starting tuning or acceptance.
- `docs/QWEN36-8BIT.md`: Existing single-machine evidence, failure records, memory limits, and dual-host estimates.
- `README.md`, `scripts/`, `native/`, `src/`: Current startup, build, and actual implementation status. Plans in documentation do not prove that code has been implemented.

Do not copy performance thresholds or model revisions into the skill. If specifications are missing, look for migrated files; if not found, it indicates a lack of acceptance criteria, and do not invent substitute metrics.

## Deployment, Updates, and Rollbacks

When involving initial deployment, updating running versions, service restarts, rollbacks, or hosting acceptance, read the [Deployment and Update SOP](references/deployment-sop.md) and verify actual capabilities against the target version's deployment documentation, installation scripts, and package manifests.
Retain `split-serve` / `split-agent` command names; the current acceptance topology is a one-to-one mapping between Mac serve and Windows agent, with Windows accessing the coordinator indirectly via serve.
Only write SOPs without triggering deployment; when the user explicitly installs themselves and provides feedback later, wait for the feedback before executing dependency installation verification. Routine updates follow existing authorizations, and administrator steps adhere to the permission boundaries below.

## SSH Access and Environment Discovery

The user is responsible for initial SSH configuration. Only use SSH aliases, usernames, and paths provided by the user or already existing that explicitly correspond to the device; do not guess IPs or scan home networks.
After confirming that local SSH configuration can resolve the target, test with a limited timeout and non-interactively, e.g., `BatchMode=yes` and `ConnectTimeout=10`.
Retain host key verification; first-time trust or host key changes must be resolved through user verification or existing trusted fingerprints, and cannot be bypassed by disabling verification.

If connection information is missing, ask once for the SSH alias (or address and username) and the desired remote working directory.
Do not request passwords, private keys, or login tokens. Do not output SSH private keys, full credential environments, or configurations containing secrets.
If the user is still configuring, authentication fails, or the connection times out, report the specific missing items and continue with independent local implementation; retry after configuration changes, but do not retry infinitely.

The first read-only check after a successful connection must determine:

- Windows user, default shell, working directory, disk space, and repository changes.
- Whether RTX 4090 is visible, actual VRAM, driver version, and existing GPU load; refer to `nvidia-smi`.
- .NET SDK, MSVC/C++ Build Tools, CUDA toolkit, build tools, and actual available compilers; the CUDA version reported by the driver does not equal the installed toolkit.
- Whether the execution environment is native Windows or WSL. Default to checking the user-provided native Windows environment first; do not silently migrate to WSL, nor mix benchmarks from both environments.
- Current LAN type and link status; wired, Wi-Fi, Gigabit, and 10-Gigabit are all based on actual measurements.

A successful connection does not mean CUDA, RPC, models, or dual-host inference are available; record verification results separately.
Standard compilation, synchronization, and testing during development are authorized to continue; installations requiring admin privileges, driver changes, firewall modifications, or system setting changes should specify the exact requirements and not force changes using a standard SSH identity.

## Responsibilities and Execution Boundaries on Both Ends

GitHub stores the shared version history for both ends. The Mac acts as the controller for developing Metal, the coordinator, and unified test drivers; Windows uses an independent Git checkout for development, building, and running CUDA.
CUDA is built and run on Windows; Metal is built and run on Mac. Joint verification is based on the same complete commit SHA, but executables and platform dependencies are generated separately.

Read the expected architecture for this round from repository specifications: 20/20 prefill pipeline, 4090 front-end weights resident, Mac takes over full model decoding.
The product must use self-written .NET/C#/Metal/CUDA paths; MLX, llama.cpp, vLLM, etc., can serve as independent numerical or performance references but cannot impersonate the product's execution backend.
Do not dynamically move expert weights by default, do not automatically downgrade quantization precision, and do not silently fall back to CPU or other backends.

SSH is the development control channel, used for remote Git operations, building, starting, logging, and process management; source code is synchronized via GitHub. Model hidden/KV/Delta data travels through the implemented RPC/data channels.
Moving files in chunks via SSH commands or simulating network latency does not count as implementing the inference pipeline. If an experiment uses an SSH tunnel, explicitly record that topology and its overhead.
Cross-host listening follows the current product's remote access and authentication configuration; do not disable authentication or expose to the internet for measurement convenience.

Windows commands are written according to the actual shell; do not directly send POSIX commands to PowerShell/cmd. For complex operations, prioritize writing scripts included in version control, call them after synchronization via GitHub, to avoid nested interpolation and multi-layer quotes. Do not assume Windows has rsync, bash, or the same path syntax as Mac.

## GitHub Project Management

All project modifications (including code, documentation, skills, experiment records, and site content) must first be committed and pushed on a feature branch, and a PR must be created, waiting for explicit user authorization to merge. Only when the user explicitly requests direct push to `main` is it allowed to bypass the PR; requests such as "commit and push," "update deployment," or "update site" do not constitute authorization for direct push to `main`. This rule applies to modifications and synchronization on both Mac and Windows; `main` must not be updated directly to trigger deployment or Pages publishing.

The current repository is `https://github.com/LittleLittleCloud/Infer.NET` (public), and the progress and experiment notes site is `https://littlelittlecloud.github.io/Infer.NET/`. The actual remote and publishing status still need to be verified.

Use GitHub Issues to track tasks, dependencies, blockers, acceptance criteria, and empirical evidence; chat is for discussion and instructions, not as the sole task record.
First, verify the Git remote, the user-specified repository, and available identities, then search for existing related Issues/PRs, prioritizing continuation of existing tasks to avoid creating duplicates.
When no target repository is available, complete locally independent work and ask for the GitHub repository address or the owner/name and visibility of the new repository; do not guess the target or publicly expose projects on your own.

A work item that can be independently accepted corresponds to one Issue, recording the specific problem, scope, dependencies, acceptance criteria, and the version/link of performance specifications.
Larger dual-machine targets use a parent Issue to link sub-tasks; each sub-task specifies whether it acts on Mac, Windows, or both.
Associate implementation branches with Issues, and reference relevant numbers in commits and PRs. Significant progress, failure reasons, blockers, and empirical results are written back to the Issue; avoid generating comments for every command.
Use existing labels, milestones, and Projects conventions; if no conventions exist, express status in the Issue body or checklist, without creating additional management systems.

PRs record changes, dual-machine verification, accurate commit SHA, benchmark configuration, and evidence location. The corresponding Issue is closed only after the task acceptance is actually passed; merging the code itself does not prove that performance goals are met.
Work that does not meet acceptance criteria remains in an incomplete state with remaining items noted; the overall goal cannot be closed just because a phase ends or CUDA compilation succeeds.
Create/update Issues, commits, and synchronize implementation branches within the scope of the current task authorization; do not automatically merge PRs, force-push shared branches, or change repository visibility based on this.

## GitHub Pages Project Progress and Experiment Notes

Use GitHub Pages as the common entry point for project progress and experiment notes. Issues hold task details and acceptance evidence; Git holds code, specifications, and note sources; Pages summarizes current status and presents experimental methods, results, and evidence links. A GitHub Projects board or chat summary does not replace Pages.
Reuse the repository's existing site directory, framework, and deployment workflow. If none exists, maintain a lightweight static progress site in the same GitHub repository rather than adding a separate backend.
Track site sources and display data in Git, and deploy through the repository's configured GitHub Pages workflow. Do not maintain a separate, unversioned copy of the live content.

The progress page displays the current phase, completed/in-progress/blocked items, next steps, milestones, and a comparison of single-machine/dual-machine empirical results with fixed targets.
Each item's status is associated with an Issue/PR; performance records include actual model accuracy, input/output length, cache conditions, device, measurement date, commit SHA, and evidence links.
Targets, theoretical estimates, and empirical results must be clearly distinguished. Metrics are summarized from repository specifications and verified results; untested items display "Untested" and cannot be filled with zeros or replaced by estimates.
The page displays the content update time and the corresponding source code version; meaningful task status changes, new benchmarks, acceptance completion, or blocker changes are synchronized and updated, not just updating the Issue and letting the homepage expire.

Experiment notes are organized by date and experiment identifier; each meaningful experiment is recorded independently, and links to relevant notes are provided from the progress page. Follow the repository's document structure to avoid duplicating result copies with existing benchmark reports.
Notes must include at least: the problem and hypothesis to be verified, baseline and changes in this round, reproducible configuration/commands, model and input hashes, source code and artifact versions on both ends, raw results, correctness checks, analytical conclusions, and next steps.
Performance experiment records include warm-up and formal rounds, chunking, cache, precision, network, power supply, and memory conditions; A/B comparisons explain which variables changed, and attribution to that change is made only when supported by measurements.
Successful, degraded, no significant gain, interrupted, and failed experiments are all retained; unrun plans are clearly marked as planned, and expected values cannot be filled in to fake results.
Associate Issues/PRs, commit SHAs, run identifiers, and raw log links; raw data that cannot be made public is marked with access restrictions, and local machine paths are not disguised as accessible download links.
When subsequent conclusions change, add re-tests or corrections, retain original observations, and indicate what evidence corrected them; do not overwrite failed experiments to create a record of continuous speedup.

Check display data, build results, and links before publishing; links in the site must be suitable for GitHub Pages, cannot use local absolute paths, and cannot embed credentials or private network connection information.
After deployment, verify workflow results, actual pages, and versions; only report "Published" upon success; if build or deployment fails, retain the last available page and explain that it is not yet online.
When the target repository is undetermined or publishing permissions are missing, continue to complete locally previewable content and explain the missing items; do not fabricate site URLs or automatically change repository visibility to enable Pages.

## Code Synchronization and Artifact Tracking

Source code synchronization strictly uses GitHub: the development end commits and pushes on the branch corresponding to the Issue, and the other end fetches and checks out that complete SHA in an independent checkout/worktree, then builds and verifies.
The receiving end verifies the remote and actual HEAD; identical branch names do not imply identical source code. Joint acceptance requires clean source code on both ends and identical HEADs, with platform artifact hashes recorded separately.
CUDA modifications made on Windows must also be committed and pushed, then fetched and received by Mac. When both ends modify simultaneously, resolve through normal merging first; do not directly overwrite files.

Before syncing, check for uncommitted and untracked files on both ends. The historical repository may have all files untracked; before establishing version history for the first time, check ignore rules, sensitive information, and large files, stage by explicit file scope, and do not upload the entire directory directly.
When there are uncommitted modifications on the remote, first identify ownership and preserve them; use an independent worktree or merge modifications before syncing. Do not use `reset --hard`, `clean`, force push, or deletion-style mirroring to eliminate conflicts.
Do not bypass source code synchronization via scp/rsync/archives; when GitHub is temporarily unavailable, local development can continue, but dual-machine synchronization should be marked as blocking unless the user explicitly authorizes a temporary alternative method.

Content included in Git includes source code, build attributes, project/dependency lock files, native code, test scripts, and necessary documentation.
Model weights, credentials, build artifacts, Python environments, and large raw logs do not enter the standard Git history. Model weights are downloaded at a fixed revision or transferred and hash-verified separately.
Small benchmark summaries and reproduction configurations can be included in the repository; raw logs are retained via appropriate artifact storage, with accessible links recorded in Issues/PRs or locations marked as local-only with hashes.
Public Issues/PRs must not include private network connection information, tokens, or user private inputs; code and evidence follow the repository's established visibility settings.

Record the GitHub repository, Issue/PR, full commit SHA, and hashes of artifacts on both ends for each round; locally save the Mac repository location, SSH alias, and Windows source/build/model paths.
Connection mappings record only non-secret fields to local ignore files, such as `artifacts/dual-host/connection.local.json` within the repository; first confirm the path is ignored by Git.
Do not treat example placeholder values for usernames, addresses, or directories as configured facts.

Experimental artifacts use independent version directories and ports, without overwriting user services binaries, model defaults, or TUI currently in use.
Only stop the PID started in this round after identity verification; do not batch terminate dotnet, Python, or GPU processes by process name.
Remote long-running tasks should have persistent logs, exit codes, and run identifiers; after SSH disconnection, first confirm the status of old tasks and do not restart duplicate copies that occupy GPU resources.

## Status Transfer and Verification Order

Mac taking over full decoding requires the KV from the 4090 frontend, as well as the correct position of DeltaNet/convolutional states; hidden vectors alone are insufficient.
The protocol explicitly specifies model/revision, request ID, layer range, token position, dtype, shape/layout, endianness, chunk sequence number, and version, rejecting mismatched and incomplete states.
First token readiness and complete state decodability are observed separately; prefill completion must not be declared until state import is complete.

Proceed in an order that enables error localization: CUDA operator numerical checks → frontend layer output and state checks → cross-machine transfer and import → complete short input generation → long input and continuous decoding → formal performance acceptance.
References and tolerances for verification follow repository specifications; numerical correctness for short inputs cannot replace tests for long request state continuity, capacity, and stability.
When token divergence occurs, first locate the first point of divergence and the corresponding logits/state; do not pass based solely on text that "looks reasonable."

## Measurement and Completion Criteria

Formal measurements strictly follow the target document's rules for input, quantization, caching, chunking, warm-up, iterations, power supply, memory, and stability.
Competitive benchmarks on the same GPU run serially; overlapping computation at both ends of the pipeline belongs to the system under test and should not be artificially serialized.
When users request battery or other power modes, separate measurements can be taken, but they must be clearly categorized; they cannot be mixed into fixed plugged-in benchmarks.

End-to-end intervals are measured by a monotonic clock on the same control end, with each end recording local phase durations; do not subtract uncalibrated cross-machine timestamps.
Total duration includes actual dispatch, communication, read-back/upload, state handover, and pipeline start/end; list pure GPU operator durations separately.
Internal first-token readiness cannot be treated as user-visible TTFT for non-streaming APIs.

Source logs are saved in the `artifacts/dual-host/` directory with run identifiers, including source/artifact/input hashes, actual topology, and exit status; collect logs from both ends before drawing conclusions.
Retain failed rounds and reasons; do not select only the fastest run, do not write theoretical limits as measured results, do not pass off short outputs as long outputs, and do not claim parity across different precision or caching conditions.
After OOM or disconnection, first locate resource/task status, adjust with traceable reasons, and retest; changing precision, model, or targets must be explicitly stated; do not silently lower thresholds to pass.

Report what was achieved this time, which checks actually passed, the gap between each measured tier and the target, unverified items, and the next bottleneck.
Final completion can only be declared when all repository specification thresholds are actually passed; stage passes, successful CUDA compilation, or SSH connectivity are not end-to-end completion.

## Project Writing Language

Follow the repository root `AGENTS.md`: use English for documentation, reports, experiment records, website content, code comments, commits, Issues, and PRs. Preserve literal benchmark/test inputs and raw evidence, including their original language, hashes, identifiers, and numerical results.
