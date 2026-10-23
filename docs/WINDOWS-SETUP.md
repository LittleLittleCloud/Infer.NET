# Windows development setup

Use a dedicated Windows-native checkout under the SSH user's writable directory. Synchronize source through GitHub and check the complete commit SHA before comparing with Mac results. Do not reuse or overwrite a checkout with uncommitted work.

## User tools

In PowerShell at the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\windows\install-user-tools.ps1
# Open an activated development shell:
powershell.exe -NoProfile -ExecutionPolicy Bypass -NoExit -Command ". .\scripts\windows\environment.ps1"
```

The installer reads the exact SDK from `global.json`, checks the Windows x64 ZIP against Microsoft's release-metadata SHA512, and installs it under `%LOCALAPPDATA%\Infer.NET\dotnet`. Python 3.13.17 uses the official full runtime ZIP, checked against the SHA256 in Python.org's Windows release manifest; `ensurepip` enables pip. This avoids Windows Installer dependencies in SSH sessions and retains venv support. Existing machine-wide SDKs are preserved. No administrator rights are needed for these user tools.

Dot-source `environment.ps1` in each new build session. This places the project SDK before machine-wide SDKs on PATH and enables UTF-8 Python output. The installer also adds tool directories to the user PATH; machine PATH entries can still take precedence without this activation step.

## C++ compiler and CUDA

The RTX 4090 host already has CUDA 13.4 and Windows SDK directories. Verify the actual C++ compiler separately. If Visual Studio Build Tools are missing:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\windows\prepare-build-tools.ps1
```

This downloads the official Visual Studio 2022 Build Tools bootstrapper, validates its Microsoft Authenticode signature, and prints the exact administrator command. Run that command in an administrator PowerShell on Windows. It installs the C++ tools and recommended Windows SDK components without automatically restarting the machine. Check the installer result; exit code 3010 indicates a restart is required. Do not elevate the ordinary SSH account merely to run builds.

## Validation

```powershell
# Can run while an administrator installs C++ tools:
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\windows\verify-environment.ps1 -Stage Managed

# Requires working MSVC, Windows SDK and CUDA:
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\windows\verify-environment.ps1 -Stage Native
```

`Managed` performs locked restore, Release build, unit tests and real-process synthetic RPC integration checks. `Native` imports the Visual Studio x64 compiler environment, compiles and executes an SM89 CUDA numerical check, publishes the host using Native AOT, and runs RPC integration checks against that executable. `-Stage All` runs both stages. A nonstandard compiler installation can be selected with `-VcVarsPath`.

Every invocation writes a transcript, result JSON, source commit and (for successful native validation) executable hashes under a new `artifacts/windows/<run-id>` directory. Failures return nonzero and remain recorded. Installation logs and downloads stay in the user-local installer directory.

This host's `api.nuget.org` TLS handshake failed with both Schannel and Python/OpenSSL. The official V2 package endpoint and separate `data.nuget.org` vulnerability endpoint are reachable. For this host, add `-NuGetConfigPath scripts\windows\NuGet.v2.config` to validation commands. This opt-in configuration retains vulnerability auditing and locked restore without changing global NuGet configuration or certificate verification. It does not fix the original endpoint's connectivity; return to default V3 access when that is resolved.

The CUDA check compares 851 matrix-product outputs against a CPU double-precision reference with absolute tolerance 1e-5, including partial thread blocks. It verifies compiler/runtime/GPU operation only. It is not the CUDA inference backend, a model correctness test, or a throughput benchmark. CUDA model inference and cross-machine state handoff require separate implementation and validation.
