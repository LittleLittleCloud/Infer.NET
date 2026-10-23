param([Parameter(Mandatory=$true)][string]$Config,[Parameter(Mandatory=$true)][string]$RunDirectory)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $root
if(Test-Path $RunDirectory){throw 'Run directory exists; inspect old PID before another start'}
New-Item -ItemType Directory $RunDirectory | Out-Null
$exe=Join-Path $root 'artifacts\deploy\build-windows\deepinfer.exe'
# Development foreground runner. Keep this SSH/terminal session alive.
# Production startup independent of login/SSH uses install-windows.ps1 and SCM.
$log=Join-Path $RunDirectory 'agent.log'
$p=Start-Process -FilePath $exe -ArgumentList @('split-agent','--config',('"'+$Config+'"'),'--log-file',('"'+$log+'"')) -PassThru -NoNewWindow
$null=$p.Handle
$p.Id | Set-Content (Join-Path $RunDirectory 'pid.txt')
git rev-parse HEAD | Set-Content (Join-Path $RunDirectory 'source.txt')
Get-FileHash $exe,(Join-Path (Split-Path $exe) 'deepinfer_cuda.dll') | ConvertTo-Json | Set-Content (Join-Path $RunDirectory 'artifacts.json')
Write-Output ('Started deployment agent PID '+$p.Id)

$p.WaitForExit()
$p.Refresh()
$p.ExitCode | Set-Content (Join-Path $RunDirectory 'exit.txt')
exit $p.ExitCode
