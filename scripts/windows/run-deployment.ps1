param([Parameter(Mandatory=$true)][string]$Config,[Parameter(Mandatory=$true)][string]$RunDirectory)
$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $root
if(Test-Path $RunDirectory){throw 'Run directory exists; inspect old PID before another start'}
New-Item -ItemType Directory $RunDirectory | Out-Null
$exe=Join-Path $root 'artifacts\deploy\build-windows\deepinfer.exe'
# WMI creates a process under this account outside the SSH session's job object.
# This does not install a service or require elevation; production uses the SCM installer.
$log=Join-Path $RunDirectory 'agent.log'
$command='"'+$exe+'" split-agent --config "'+$Config+'" --log-file "'+$log+'"'
$created=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command;CurrentDirectory=$root}
if($created.ReturnValue -ne 0){throw ('Cannot launch detached agent: '+$created.ReturnValue)}
$created.ProcessId | Set-Content (Join-Path $RunDirectory 'pid.txt')
git rev-parse HEAD | Set-Content (Join-Path $RunDirectory 'source.txt')
Get-FileHash $exe,(Join-Path (Split-Path $exe) 'deepinfer_cuda.dll') | ConvertTo-Json | Set-Content (Join-Path $RunDirectory 'artifacts.json')
Write-Output ('Started deployment agent PID '+$created.ProcessId)
