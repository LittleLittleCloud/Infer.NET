param(
    [ValidateSet('Managed','Native','All')][string]$Stage = 'All',
    [string]$VcVarsPath,
    [string]$NuGetConfigPath,
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunId = (Get-Date -Format yyyyMMdd-HHmmss)
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$output = Join-Path $root "artifacts\windows\$RunId"
New-Item -ItemType Directory -Path $output -ErrorAction Stop | Out-Null
Start-Transcript -Path (Join-Path $output 'validation.log') | Out-Null
$result = [ordered]@{ stage=$Stage; status='running'; commit=$null; exitCode=1 }
function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}
Push-Location $root
try {
    . (Join-Path $PSScriptRoot 'environment.ps1')
    $result.commit = (& git rev-parse HEAD | Out-String).Trim()
    if ($NuGetConfigPath) {
        $NuGetConfigPath = (Resolve-Path $NuGetConfigPath).Path
        $result.nugetConfigSHA256 = (Get-FileHash $NuGetConfigPath -Algorithm SHA256).Hash
    }
    $expected = (Get-Content global.json -Raw | ConvertFrom-Json).sdk.version
    $actual = (& dotnet --version | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or $actual -ne $expected) { throw "SDK mismatch: $actual / $expected" }
    if ($Stage -in 'Managed','All') {
        $restoreArguments = @('restore','DeepInfer.slnx','--locked-mode')
        if ($NuGetConfigPath) { $restoreArguments += @('--configfile',$NuGetConfigPath) }
        Invoke-Checked dotnet $restoreArguments
        Invoke-Checked dotnet @('build','DeepInfer.slnx','-c','Release','--no-restore')
        Invoke-Checked dotnet @('test','tests/DeepInfer.Tests','-c','Release','--no-build')
        Invoke-Checked python @('scripts/integration.py','--executable','src/DeepInfer.Host/bin/Release/net10.0/deepinfer.exe')
    }
    if ($Stage -in 'Native','All') {
        if (!$VcVarsPath) {
            $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
            $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if (!$installation) { throw 'MSVC C++ toolchain not found. Install Build Tools or supply -VcVarsPath.' }
            $VcVarsPath = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
        }
        if (!(Test-Path $VcVarsPath)) { throw "Missing vcvars64: $VcVarsPath" }
        $environment = & $env:ComSpec /d /c "`"$VcVarsPath`" >nul && set"
        if ($LASTEXITCODE -ne 0) { throw 'vcvars64 failed.' }
        foreach ($line in $environment) {
            if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process') }
        }
        # Reassert repository SDK after the VS developer environment modifies PATH.
        . (Join-Path $PSScriptRoot 'environment.ps1')
        Get-Command cl.exe,link.exe,nvcc.exe | Select-Object Name,Source | Format-Table
        Invoke-Checked nvcc @('--version')
        $cudaCheck = Join-Path $output 'cuda-environment-check.exe'
        Invoke-Checked nvcc @('-std=c++17','-arch=sm_89','-O2','native/DeepInfer.Cuda/environment_check.cu','-o',$cudaCheck)
        Invoke-Checked $cudaCheck @()
        $native = Join-Path $output 'native'
        $publishArguments = @('publish','src/DeepInfer.Host','-c','Release','-r','win-x64','-o',$native)
        if ($NuGetConfigPath) { $publishArguments += "-p:RestoreConfigFile=$NuGetConfigPath" }
        Invoke-Checked dotnet $publishArguments
        Invoke-Checked python @('scripts/integration.py','--executable',(Join-Path $native 'deepinfer.exe'))
        $result.cudaExecutableSHA256 = (Get-FileHash $cudaCheck -Algorithm SHA256).Hash
        $result.hostExecutableSHA256 = (Get-FileHash (Join-Path $native 'deepinfer.exe') -Algorithm SHA256).Hash
    }
    $result.status = 'passed'; $result.exitCode = 0
} catch {
    $result.status = 'failed'; $result.error = $_.Exception.Message
    Write-Warning $_
} finally {
    $result | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $output 'result.json') -Encoding UTF8
    Pop-Location
    Stop-Transcript | Out-Null
}
exit $result.exitCode
