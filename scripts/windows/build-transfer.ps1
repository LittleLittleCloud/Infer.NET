$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $root
$vswhere="${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$installation=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$installation) {throw 'MSVC missing'}
$vcvars=Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$vars=& $env:ComSpec /d /c "`"$vcvars`" >nul && set"
if ($LASTEXITCODE -ne 0){throw 'vcvars64 failed'}
foreach($line in $vars){if($line -match '^([^=]+)=(.*)$'){[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}}
$out=Join-Path $root 'artifacts\state-transfer'
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $out
try {
 & nvcc -std=c++17 -arch=sm_89 -O2 --shared (Join-Path $root 'native\StateTransfer\transfer_cuda.cu') -o transfer_cuda.dll
 if($LASTEXITCODE -ne 0){throw 'CUDA transfer build failed'}
 Get-FileHash transfer_cuda.dll -Algorithm SHA256
} finally {Pop-Location}
