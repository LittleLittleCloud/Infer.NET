$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $root
. scripts\windows\environment.ps1
& scripts\windows\build-prefix.ps1
& python scripts\split\test-cuda.py
if($LASTEXITCODE -ne 0){throw 'CUDA numerical checks failed'}
& dotnet restore DeepInfer.slnx --configfile scripts\windows\NuGet.v2.config
if($LASTEXITCODE -ne 0){throw 'Restore failed'}
& dotnet build DeepInfer.slnx -c Release --no-restore
if($LASTEXITCODE -ne 0){throw 'Build failed'}
& dotnet test tests\DeepInfer.Tests -c Release --no-build
if($LASTEXITCODE -ne 0){throw 'Tests failed'}
& dotnet publish src\DeepInfer.Host -c Release -r win-x64 -o artifacts\deploy\build-windows -p:RestoreConfigFile=scripts/windows/NuGet.v2.config
if($LASTEXITCODE -ne 0){throw 'Native AOT publish failed'}
git rev-parse HEAD | Set-Content artifacts\deploy\build-windows\source.txt
