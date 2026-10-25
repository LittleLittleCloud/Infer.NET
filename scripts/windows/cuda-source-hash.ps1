$ErrorActionPreference='Stop'
$root=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$parts=foreach($name in @('split_prefix.cu','prefill_batch.cuh','prefill_chunk.cuh','prefill_executor.cuh','q8_kv.cuh','../kv_layout.h')) {
    $digest=(Get-FileHash (Join-Path $root ('native\DeepInfer.Cuda\'+$name)) -Algorithm SHA256).Hash
    "${name}:$digest"
}
$hasher=[System.Security.Cryptography.SHA256]::Create()
try { [BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($parts -join "`n"))).Replace('-','') }
finally { $hasher.Dispose() }
