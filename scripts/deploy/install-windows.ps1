param([Parameter(Mandatory=$true)][string]$Executable,[Parameter(Mandatory=$true)][string]$Config,[switch]$Uninstall)
$ErrorActionPreference='Stop'
# Review this script first and run in administrator PowerShell. No firewall/driver changes.
$principal=New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if(!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){throw 'Administrator PowerShell is required to install or remove this service.'}
$name='InferNetCuda'
if($Uninstall){
 $service=Get-Service $name -ErrorAction SilentlyContinue
 if($service){Stop-Service $name; & sc.exe delete $name; if($LASTEXITCODE -ne 0){throw 'Service removal failed'}}
 return
}
if(Get-Service $name -ErrorAction SilentlyContinue){throw 'Service exists; drain requests and uninstall the old service before switching releases.'}
$exe=(Resolve-Path $Executable).Path;$cfg=(Resolve-Path $Config).Path
# Use a dedicated standard local account. Password is entered locally, never sent to the agent or command line.
$credential=Get-Credential -Message 'Dedicated standard local Infer.NET service account'
$bin='"'+$exe+'" split-agent --config "'+$cfg+'" --windows-service'
New-Service -Name $name -DisplayName 'Infer.NET CUDA' -BinaryPathName $bin -StartupType Automatic -Credential $credential | Out-Null
& sc.exe failure $name reset= 86400 actions= restart/10000/restart/30000/restart/60000
if($LASTEXITCODE -ne 0){throw 'Recovery policy failed'}
& sc.exe failureflag $name 1
if($LASTEXITCODE -ne 0){throw 'Recovery policy flag failed'}
Start-Service $name
Get-Service $name
