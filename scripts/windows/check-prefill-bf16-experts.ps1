param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $root
. scripts\windows\environment.ps1
$out=Join-Path $root $OutputDirectory
if(Test-Path $out){throw 'Output directory already exists'}
New-Item -ItemType Directory $out | Out-Null
$vswhere="${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$installation=& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$installation) {throw 'MSVC missing'}
$vcvars=Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$vars=& $env:ComSpec /d /c "`"$vcvars`" >nul && set"
if ($LASTEXITCODE -ne 0){throw 'vcvars64 failed'}
foreach($line in $vars){if($line -match '^([^=]+)=(.*)$'){[Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')}}
$exe=Join-Path $out 'prefill_bf16_experts_checks.exe'
& nvcc -std=c++17 -arch=sm_89 -O2 native\DeepInfer.Cuda\prefill_bf16_experts_checks.cu -o $exe
if($LASTEXITCODE -ne 0){throw 'BF16 expert compilation failed'}
git rev-parse HEAD | Set-Content (Join-Path $out 'source.txt')
& $exe | Tee-Object -FilePath (Join-Path $out 'results.jsonl')
$code=$LASTEXITCODE
$code | Set-Content (Join-Path $out 'exit.txt')
Get-FileHash $exe -Algorithm SHA256 | Format-List | Out-File (Join-Path $out 'binary-hash.txt')
if($code -ne 0){throw 'BF16 expert validation failed'}
