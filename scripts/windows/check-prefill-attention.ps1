param([Parameter(Mandatory=$true)][string]$OutputDirectory)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $root
. scripts\windows\environment.ps1
# Reuse the established MSVC/CUDA environment, without touching deployed packages.
& scripts\windows\build-prefix.ps1
$out=Join-Path $root $OutputDirectory
if(Test-Path $out){throw 'Output directory already exists'}
New-Item -ItemType Directory $out | Out-Null
$exe=Join-Path $out 'prefill_attention_checks.exe'
& nvcc -std=c++17 -arch=sm_89 -O2 native\DeepInfer.Cuda\prefill_attention_checks.cu -o $exe
if($LASTEXITCODE -ne 0){throw 'Prefill attention compilation failed'}
git rev-parse HEAD | Set-Content (Join-Path $out 'source.txt')
& nvidia-smi --query-gpu=name,memory.used,pstate,temperature.gpu,power.draw --format=csv | Set-Content (Join-Path $out 'device-before.txt')
& $exe | Tee-Object -FilePath (Join-Path $out 'results.jsonl')
$code=$LASTEXITCODE
$code | Set-Content (Join-Path $out 'exit.txt')
Get-FileHash $exe -Algorithm SHA256 | Format-List | Out-File (Join-Path $out 'binary-hash.txt')
if($code -ne 0){throw 'Prefill attention validation failed'}
