param(
    [Parameter(Mandatory=$true)][int]$OldAgentPid,
    [Parameter(Mandatory=$true)][string]$OldAgentExe,
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][string]$RunDirectory
)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$library = Join-Path $root 'native\DeepInfer.Cuda\build\deepinfer_cuda.dll'
$benchmark = Join-Path $root 'scripts\split\benchmark-prefix-chunk.py'
$result = Join-Path $RunDirectory 'result'
if (Test-Path $RunDirectory) { throw "Run directory already exists: $RunDirectory" }
if (!(Test-Path $OldAgentExe) -or !(Test-Path $Config) -or !(Test-Path $library)) {
    throw 'Old agent, config, or candidate library is missing'
}
$agent = Get-Process -Id $OldAgentPid -ErrorAction Stop
if (![string]::Equals($agent.Path, $OldAgentExe, [StringComparison]::OrdinalIgnoreCase)) {
    throw "PID $OldAgentPid does not match the expected old agent executable"
}
$weights = (Get-Content -Raw $Config | ConvertFrom-Json).weights
if (!(Test-Path $weights)) { throw 'Configured model weights are unavailable' }
New-Item -ItemType Directory -Path $RunDirectory | Out-Null
Set-Location $root
git rev-parse HEAD | Set-Content (Join-Path $RunDirectory 'source.txt')
(Get-FileHash $library).Hash | Set-Content (Join-Path $RunDirectory 'candidate-dll-sha256.txt')
(Get-FileHash $OldAgentExe).Hash | Set-Content (Join-Path $RunDirectory 'old-agent-exe-sha256.txt')
$stopped = $false
try {
    Write-Output "Stopping verified old agent PID $OldAgentPid"
    Stop-Process -Id $OldAgentPid -ErrorAction Stop
    $stopped = $true
    $agent.WaitForExit()
    Start-Sleep -Seconds 3
    Write-Output 'Running full-prefix correctness and 512-token A/B'
    & python $benchmark --weights $weights --library $library --output $result --compare-fast --kv-bytes 4 2>&1 |
        Tee-Object -FilePath (Join-Path $RunDirectory 'benchmark.log')
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed with exit code $LASTEXITCODE" }
}
finally {
    if ($stopped) {
        Write-Output 'Restoring old agent'
        $log = Join-Path $RunDirectory 'restored-agent.log'
        $stdout = Join-Path $RunDirectory 'restored-agent.stdout.log'
        $stderr = Join-Path $RunDirectory 'restored-agent.stderr.log'
        $restored = Start-Process -FilePath $OldAgentExe -ArgumentList @(
            'split-agent', '--config', ('"' + $Config + '"'), '--log-file', ('"' + $log + '"')
        ) -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $restored.Id | Set-Content (Join-Path $RunDirectory 'restored-agent-pid.txt')
        Write-Output "Restored old agent PID $($restored.Id)"
    }
}
