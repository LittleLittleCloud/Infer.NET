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
    # The official full runtime ZIP also works in SSH sessions where the MSI
    # service is unavailable. Unlike the embeddable ZIP it supports venv/pip.
    $manifestPath = Join-Path $cache "windows-$PythonVersion.json"
    Download "https://www.python.org/ftp/python/$PythonVersion/windows-$PythonVersion.json" $manifestPath
    $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
    $pythonChannel = ($PythonVersion.Split('.')[0..1] -join '.')
    $release = $manifest.versions | Where-Object {
        $_.id -eq "pythoncore-$pythonChannel-64" -and $_.'sort-version' -eq $PythonVersion
    } | Select-Object -First 1
    if (!$release.url -or !$release.hash.sha256) { throw 'Python x64 release manifest entry missing.' }
    $archive = Join-Path $cache "python-$PythonVersion-amd64.zip"
    Download $release.url $archive
    if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne $release.hash.sha256) { throw 'Python SHA256 mismatch.' }
    Expand-Archive -Path $archive -DestinationPath $pythonRoot -Force
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
    & python -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Python pip setup failed.' }
    Write-Output "User tools ready: SDK $actual; Python $python"
} finally { Pop-Location }
