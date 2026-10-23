param(
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet(256,512,1024,2048)][int]$Chunk=1024,
    [ValidateSet(1,5)][int]$Mode=5,
    [ValidateSet(2,4)][int]$KvBytes=4
)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$model=(Get-Content -Raw $Config | ConvertFrom-Json).weights
if(-not $model -or -not (Test-Path $model)){throw 'Configured prefix weights are unavailable'}
$dll=Join-Path $root 'native\DeepInfer.Cuda\build\deepinfer_cuda.dll'
if(-not (Test-Path $dll)){throw 'Build the current CUDA prefix DLL first'}
Set-Location $root
python (Join-Path $root 'scripts\split\check-cuda-long-chunk.py') --weights $model --library $dll --output $Output --chunk $Chunk --mode $Mode --kv-bytes $KvBytes
if($LASTEXITCODE -ne 0){throw "Long CUDA chunk diagnostic failed with code $LASTEXITCODE"}
