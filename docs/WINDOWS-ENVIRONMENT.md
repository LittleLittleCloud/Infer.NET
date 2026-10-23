# Windows development environment

Verified on 2026-10-22 (America/Los_Angeles), using Windows-native SSH and a dedicated standard local account. The host fingerprint was verified by the user before it was trusted. Public-key login and remote `hostname` / `whoami` commands succeeded. Addresses, account names, keys and connection mappings remain local and are not published.

## Read-only discovery

| Item | Observed result |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090; 24564 MiB total, 1334 MiB used, 0% GPU utilization at discovery |
| Driver | 610.88 |
| CUDA compiler | `nvcc` 13.4.59 available; toolkit directories 12.1, 12.8, 12.9 and 13.0 through 13.4 also present |
| .NET SDK | 6.0.428, 7.0.410, 8.0.426, 10.0.113 and 10.0.402; required 10.0.100 absent |
| MSVC | `cl` not on PATH; Visual Studio discovery and searches under the usual Visual Studio roots did not locate a C++ toolchain. Installation/readability still needs verification. |
| Git / CMake | Git 2.55.0.windows.5; CMake 4.4.4 |
| Python | Not found on this account's PATH |
| Windows | Native execution; OS version 10.0.26300.0; PowerShell 5.1.26100.9711 |
| Disk | System drive: 2047039500288 bytes total, 640833208320 bytes available at discovery |
| Physical network | Realtek Gaming GbE Family Controller, up at 1000000000 bit/s |
| Virtual network | Hyper-V adapters report 10 Gbps; this is not evidence of a physical 10GbE connection |
| Repository | No checkout found in the three proposed standard repository locations; no repository cloned or synchronized in this check |

Some CIM-based queries returned access denied under the standard account. OS, disk and active network information above was obtained through ordinary .NET APIs without elevation. Reported adapter speed is not measured application throughput.

## Authentication findings

The initial supplied identity could not be resolved by Windows OpenSSH. Removing its prefix reached the authentication protocol but the server subsequently reported an invalid user; a client-side `Permission denied` response alone did not establish that an account existed or that only the public key was missing. Local account enumeration confirmed that the requested local account was absent. The user created a dedicated local account and configured its public key; actual noninteractive login then succeeded.

## Remaining work

- Install the exact .NET SDK 10.0.100 without changing the committed SDK/lock policy.
- Locate or install MSVC C++ build tools and make Python available for the existing checks.
- Select an independent writable checkout, synchronize through GitHub, and verify the complete commit SHA.
- Build and validate CUDA on Windows; `nvcc --version` is not a CUDA compilation or kernel correctness test.
- Measure effective network throughput and state-transfer overhead before updating the dual-host estimates.

No compiler installation, repository synchronization, CUDA build, GPU inference, network throughput benchmark or model performance measurement was performed in this discovery. Issue #2 remains open.

## Follow-up: user tools and managed validation

Later on 2026-10-22, an independent user-owned checkout was cloned through GitHub. Mac and Windows used source commit `1022ea3e0e6503c2c7600f86b314c154dfb8dedd` on `codex/windows-environment`. Reusable installation and validation scripts are in [draft PR #6](https://github.com/LittleLittleCloud/Infer.NET/pull/6).

- Installed .NET SDK **10.0.100** from the official Windows x64 archive after checking Microsoft's SHA512 release metadata. Existing system SDKs were preserved.
- Installed **Python 3.13.17**, including pip **26.2.2**, in the user-local tools directory. The first EXE installation failed with Windows Installer error **1601**. The subsequent official full runtime ZIP passed its release-manifest SHA256 and ran successfully; the failed installation log is retained.
- Default NuGet V3 access failed TLS negotiation with Schannel and Python/OpenSSL. The explicit per-run configuration uses the official V2 package endpoint plus `https://data.nuget.org/v3/index.json` for auditing. Both package download and vulnerability-data cache creation were verified. No certificate-validation, lock-file or vulnerability-check bypass was introduced. The original V3 connectivity issue remains unresolved.
- Audited locked restore and Release build passed, with **0 warnings and 0 errors**; **68/68 unit tests** and **25/25 synthetic RPC checks** passed on Windows.
- The built host's device discovery identified the RTX 4090 as **SM89**, with the driver available and CUDA inference explicitly marked unimplemented.
- Native preflight failed with **MSVC C++ toolchain not found**. The Microsoft-signed Visual Studio 2022 Build Tools bootstrapper is prepared, but administrator installation is pending. No CUDA kernel or Native AOT executable has been validated on this host yet.

Local evidence: run `setup-managed-audited-20261022` (exit 0), `setup-native-preflight-20261022` (exit 1), and the earlier installation/restore failures under `artifacts/windows/` on Windows. Private logs were also collected on the Mac; they are not published. The opt-in NuGet configuration SHA256 is `DE0969C14B59BD38DC75339106D6589F4D74A238A971AA27383EF8E7FDE17715`.

Next: complete administrator Build Tools installation, then run the scripted SM89 numerical check and Native AOT RPC validation. Effective network bandwidth and dual-host inference remain unmeasured. Issue #2 stays open.
