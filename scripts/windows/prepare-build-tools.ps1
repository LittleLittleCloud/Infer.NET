$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$directory = Join-Path $env:LOCALAPPDATA 'Infer.NET\installers'
New-Item -ItemType Directory -Force $directory | Out-Null
$installer = Join-Path $directory 'vs_BuildTools.exe'
& curl.exe --fail --location --retry 3 --output $installer 'https://aka.ms/vs/17/release/vs_BuildTools.exe'
if ($LASTEXITCODE -ne 0) { throw 'Build Tools download failed.' }
$signature = Get-AuthenticodeSignature $installer
if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
    throw "Invalid Microsoft installer signature: $($signature.Status)"
}
Write-Output "Verified installer: $installer"
Write-Output "SHA256: $((Get-FileHash $installer -Algorithm SHA256).Hash)"
Write-Output "Run in an administrator PowerShell:"
Write-Output "& '$installer' --quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
