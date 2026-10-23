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
