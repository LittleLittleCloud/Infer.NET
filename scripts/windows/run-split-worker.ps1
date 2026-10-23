# Private connection/token config arrives on stdin, never in process arguments.
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $root
. scripts\windows\environment.ps1
$config=[Console]::ReadLine() | ConvertFrom-Json
if($config.runId -notmatch '^[a-zA-Z0-9_-]{1,80}$'){throw 'Invalid run ID'}
$out=Join-Path $root ('artifacts\split\'+$config.runId)
if(Test-Path $out){throw 'Run already exists; inspect its PID and exit status before retrying'}
New-Item -ItemType Directory $out | Out-Null
$env:DEEPINFER_TOKEN=$config.token
$sha=git rev-parse HEAD
$sha | Set-Content (Join-Path $out 'source.txt')
if((Get-FileHash (Join-Path $root 'native\DeepInfer.Cuda\split_prefix.cu')).Hash -ne (Get-Content (Join-Path $root 'native\DeepInfer.Cuda\build\deepinfer_cuda.source.sha256')).Trim()){throw 'Native CUDA build does not match current source'}
$exe=Join-Path $root 'src\DeepInfer.Host\bin\Release\net10.0\deepinfer.exe'
if((Get-FileHash (Join-Path (Split-Path $exe) 'deepinfer_cuda.dll')).Hash -ne (Get-FileHash (Join-Path $root 'native\DeepInfer.Cuda\build\deepinfer_cuda.dll')).Hash){throw 'Runtime CUDA DLL differs from native build; rebuild Host before running'}
$bundle=Join-Path $root 'artifacts\split\prefix.bin'
Get-FileHash $exe,(Join-Path (Split-Path $exe) 'deepinfer_cuda.dll'),(Join-Path (Split-Path $exe) 'DeepInfer.Backends.dll') -Algorithm SHA256 | ConvertTo-Json | Set-Content (Join-Path $out 'artifacts.json')
$process=Start-Process -FilePath $exe -ArgumentList @('split-worker','--kv',$config.kv,'--weights',('"'+$bundle+'"'),'--endpoint',$config.endpoint) -PassThru -NoNewWindow -RedirectStandardOutput (Join-Path $out 'stdout.log') -RedirectStandardError (Join-Path $out 'stderr.log')
$null=$process.Handle # Retain the process handle so ExitCode remains available after exit.
$process.Id | Set-Content (Join-Path $out 'pid.txt')
Remove-Item Env:DEEPINFER_TOKEN
$process.WaitForExit()
$process.Refresh()
if($null -eq $process.ExitCode){throw 'Worker exited without a readable exit status'}
$process.ExitCode | Set-Content (Join-Path $out 'exit.txt')
exit $process.ExitCode
