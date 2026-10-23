param([Parameter(Mandatory=$true)][string]$RunDirectory)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$agentPid=[int](Get-Content (Join-Path $RunDirectory 'pid.txt'))
$process=Get-Process -Id $agentPid -ErrorAction SilentlyContinue
if(!$process){Write-Output 'Agent already exited';return}
$expected=Join-Path $root 'artifacts\deploy\build-windows\deepinfer.exe'
if($process.Path -ne $expected){throw 'PID executable does not match this deployment trial; refusing stop'}
Stop-Process -Id $agentPid
$process.WaitForExit()
Write-Output 'Stopped verified deployment trial agent'
