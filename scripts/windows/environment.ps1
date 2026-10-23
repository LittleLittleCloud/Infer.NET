# Dot-source this file before building in a fresh PowerShell / SSH session.
$toolRoot = Join-Path $env:LOCALAPPDATA 'Infer.NET'
$env:DOTNET_ROOT = Join-Path $toolRoot 'dotnet'
$env:DOTNET_ROOT_X64 = $env:DOTNET_ROOT
$env:PATH = "$env:DOTNET_ROOT;$(Join-Path $toolRoot 'Python');$(Join-Path $toolRoot 'Python\Scripts');$env:PATH"
$env:DOTNET_CLI_TELEMETRY_OPTOUT = '1'
$env:DOTNET_NOLOGO = '1'
$env:PYTHONUTF8 = '1'
