param(
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet('bf16','fp32')][string]$Router='fp32',
    [ValidateSet('bf16','fp32')][string]$Shared='fp32'
)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$model=(Get-Content -Raw $Config | ConvertFrom-Json).weights
if(-not $model -or -not (Test-Path $model)){throw 'Configured prefix weights are unavailable'}
$dll=Join-Path $root 'native\DeepInfer.Cuda\build\deepinfer_cuda.dll'
if(-not (Test-Path $dll)){throw 'Build the current CUDA prefix DLL first'}
$env:DEEPINFER_CUDA_ROUTER_FP32=if($Router -eq 'fp32'){'1'}else{'0'}
$env:DEEPINFER_CUDA_SHARED_FP32=if($Shared -eq 'fp32'){'1'}else{'0'}
$env:DEEPINFER_CUDA_ROUTE_PROBE='1'
Set-Location $root
python (Join-Path $root 'scripts\split\check-bf16-prefix.py') --weights $model --library $dll --output $Output --tokens 128 --save-trace
if($LASTEXITCODE -ne 0){throw "BF16 prefix diagnostic failed with code $LASTEXITCODE"}
