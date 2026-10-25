$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $root
& (Join-Path $PSScriptRoot 'build-prefix.ps1')
if($LASTEXITCODE -ne 0){throw 'Prefix build failed'}
$out=Join-Path $root 'artifacts\dual-host\issue33-codec-fixtures'
if(!(Test-Path $out)){
 & python scripts/q8-kv-fixtures.py --output $out
 if($LASTEXITCODE -ne 0){throw 'Fixture generation failed'}
}
& nvcc -std=c++17 -arch=sm_89 -O2 native/DeepInfer.Cuda/q8_checks.cu -o native/DeepInfer.Cuda/build/q8-checks.exe
if($LASTEXITCODE -ne 0){throw 'Q8 check build failed'}
& native/DeepInfer.Cuda/build/q8-checks.exe (Join-Path $out 'input.f32') (Join-Path $out 'expected.q8')
if($LASTEXITCODE -ne 0){throw 'Q8 check failed'}
