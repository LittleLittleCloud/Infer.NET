param(
    [Parameter(Mandatory=$true)][int]$OldAgentPid,
    [Parameter(Mandatory=$true)][string]$OldAgentExe,
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [Parameter(Mandatory=$true)][string]$BaselineLibrary,
    [Parameter(Mandatory=$true)][string]$CandidateLibrary,
    [Parameter(Mandatory=$true)][string]$BaselineCommit,
    [Parameter(Mandatory=$true)][string]$CandidateCommit
)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$benchmark = Join-Path $root 'scripts\split\benchmark-tensor-core-prefix.py'
if (Test-Path $RunDirectory) { throw "Run directory already exists: $RunDirectory" }
foreach ($path in @($OldAgentExe, $Config, $BaselineLibrary, $CandidateLibrary, $benchmark)) {
    if (!(Test-Path $path)) { throw "Missing prerequisite: $path" }
}
$agent = Get-Process -Id $OldAgentPid -ErrorAction Stop
if (![string]::Equals($agent.Path, $OldAgentExe, [StringComparison]::OrdinalIgnoreCase)) {
    throw "PID $OldAgentPid is not the expected agent executable"
}
$weights = (Get-Content -Raw $Config | ConvertFrom-Json).weights
if (!$weights -or !(Test-Path $weights)) { throw 'Configured prefix weights are unavailable' }
New-Item -ItemType Directory -Path $RunDirectory | Out-Null
$manifest = [ordered]@{
    baselineCommit = $BaselineCommit
    candidateCommit = $CandidateCommit
    oldAgentPid = $OldAgentPid
    oldAgentExe = $OldAgentExe
    config = $Config
    baselineLibrarySha256 = (Get-FileHash $BaselineLibrary -Algorithm SHA256).Hash
    candidateLibrarySha256 = (Get-FileHash $CandidateLibrary -Algorithm SHA256).Hash
    benchmarkScriptSha256 = (Get-FileHash $benchmark -Algorithm SHA256).Hash
    gpuBefore = & nvidia-smi --query-gpu=name,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader
}
$manifest | ConvertTo-Json -Depth 3 | Set-Content (Join-Path $RunDirectory 'manifest.local.json')
. (Join-Path $PSScriptRoot 'environment.ps1')
$stopped = $false
try {
    Write-Output "Stopping verified resident agent PID $OldAgentPid for isolated full-prefix measurement"
    Stop-Process -Id $OldAgentPid -Force -ErrorAction Stop
    $stopped = $true
    $agent.WaitForExit()
    Start-Sleep -Seconds 3
    Set-Location $root
    & python $benchmark --weights $weights --baseline-library $BaselineLibrary `
        --candidate-library $CandidateLibrary --baseline-commit $BaselineCommit `
        --candidate-commit $CandidateCommit --output (Join-Path $RunDirectory 'result') `
        --trials 3 2>&1 | Tee-Object -FilePath (Join-Path $RunDirectory 'benchmark.log')
    $code = $LASTEXITCODE
    $code | Set-Content (Join-Path $RunDirectory 'benchmark.exit.txt')
    if ($code -ne 0) { Write-Warning "Tensor Core prefix benchmark failed with exit $code; restoring agent" }
}
finally {
    if ($stopped) {
        Write-Output 'Restoring the verified agent in the foreground; retain this SSH session'
        & $OldAgentExe split-agent --config $Config --log-file (Join-Path $RunDirectory 'restored-agent.log')
        if ($LASTEXITCODE -ne 0) { throw "Restored agent exited with code $LASTEXITCODE" }
    }
}
