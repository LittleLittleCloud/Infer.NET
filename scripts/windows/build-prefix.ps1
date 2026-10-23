$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $root
$vswhere="${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$installation=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$installation) {throw 'MSVC missing'}
$vcvars=Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$previousPlatform=[Environment]::GetEnvironmentVariable('Platform','Process')
$vars=& $env:ComSpec /d /c "`"$vcvars`" >nul && set"
if ($LASTEXITCODE -ne 0){throw 'vcvars64 failed'}
foreach($line in $vars){if($line -match '^([^=]+)=(.*)$'){[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}}
$out=Join-Path $root 'native\DeepInfer.Cuda\build'
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $out
try {
 & nvcc -std=c++17 -arch=sm_89 -O2 --shared (Join-Path $root 'native\DeepInfer.Cuda\split_prefix.cu') -o deepinfer_cuda.dll
 if($LASTEXITCODE -ne 0){throw 'CUDA prefix build failed'}
 (Get-FileHash (Join-Path $root 'native\DeepInfer.Cuda\split_prefix.cu') -Algorithm SHA256).Hash | Set-Content deepinfer_cuda.source.sha256
 Get-FileHash deepinfer_cuda.dll -Algorithm SHA256
} finally {
 Pop-Location
 # vcvars sets Platform=x64; leaking it redirects subsequent .NET output to bin/x64.
 [Environment]::SetEnvironmentVariable('Platform',$previousPlatform,'Process')
}
