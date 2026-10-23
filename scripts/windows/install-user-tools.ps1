param([string]$PythonVersion = '3.13.17')
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$sdkVersion = (Get-Content (Join-Path $root 'global.json') -Raw | ConvertFrom-Json).sdk.version
$toolRoot = Join-Path $env:LOCALAPPDATA 'Infer.NET'
$cache = Join-Path $toolRoot 'installers'
$dotnetRoot = Join-Path $toolRoot 'dotnet'
$pythonRoot = Join-Path $toolRoot 'Python'
New-Item -ItemType Directory -Force $cache | Out-Null

function Download([string]$Url, [string]$Destination) {
    & curl.exe --fail --location --retry 3 --output $Destination $Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Url" }
}

if (!(Test-Path (Join-Path $dotnetRoot "sdk\$sdkVersion"))) {
    $channel = ($sdkVersion.Split('.')[0..1] -join '.')
    $metadataPath = Join-Path $cache "dotnet-$channel-releases.json"
    Download "https://dotnetcli.blob.core.windows.net/dotnet/release-metadata/$channel/releases.json" $metadataPath
    $metadata = Get-Content $metadataPath -Raw | ConvertFrom-Json
    $sdk = @($metadata.releases | ForEach-Object { $_.sdks; $_.sdk } |
        Where-Object { $_.version -eq $sdkVersion }) | Select-Object -First 1
    $file = $sdk.files | Where-Object { $_.rid -eq 'win-x64' -and $_.name -like '*.zip' } | Select-Object -First 1
    if (!$file -or !$file.hash) { throw "No verified Windows x64 archive for SDK $sdkVersion." }
    $archive = Join-Path $cache "dotnet-sdk-$sdkVersion-win-x64.zip"
    Download $file.url $archive
    if ((Get-FileHash $archive -Algorithm SHA512).Hash -ne $file.hash) { throw 'SDK SHA512 mismatch.' }
    Expand-Archive -Path $archive -DestinationPath $dotnetRoot -Force
}

$python = Join-Path $pythonRoot 'python.exe'
if (!(Test-Path $python)) {
    $installer = Join-Path $cache "python-$PythonVersion-amd64.exe"
    Download "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe" $installer
    $signature = Get-AuthenticodeSignature $installer
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
        throw "Invalid Python installer signature: $($signature.Status)"
    }
    $log = Join-Path $cache "python-install-$(Get-Date -Format yyyyMMdd-HHmmss).log"
    $arguments = @('/quiet', 'InstallAllUsers=0', 'Include_launcher=0', 'Include_test=0',
        'AssociateFiles=0', 'Shortcuts=0', 'PrependPath=0', "TargetDir=`"$pythonRoot`"", '/log', "`"$log`"")
    $process = Start-Process -FilePath $installer -ArgumentList $arguments -Wait -PassThru
    if ($process.ExitCode -notin 0,3010) { throw "Python install failed: $($process.ExitCode); log $log" }
}

# Preserve the existing user PATH. environment.ps1 also puts these tools first
# in each build session because Windows normally places machine PATH first.
$paths = @($dotnetRoot, $pythonRoot, (Join-Path $pythonRoot 'Scripts'))
$oldPath = [Environment]::GetEnvironmentVariable('Path','User')
$remaining = @($oldPath -split ';' | Where-Object { $_ -and $_ -notin $paths })
[Environment]::SetEnvironmentVariable('Path', (($paths + $remaining) -join ';'), 'User')
[Environment]::SetEnvironmentVariable('DOTNET_ROOT', $dotnetRoot, 'User')
. (Join-Path $PSScriptRoot 'environment.ps1')
Push-Location $root
try {
    $actual = (& dotnet --version | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $actual -ne $sdkVersion) { throw "Expected SDK $sdkVersion, selected $actual" }
    & python --version
    if ($LASTEXITCODE -ne 0) { throw 'Python validation failed.' }
    Write-Output "User tools ready: SDK $actual; Python $python"
} finally { Pop-Location }
