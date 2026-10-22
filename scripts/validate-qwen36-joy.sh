#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Run only with model benchmarking stopped: GPU checks are serial.
export DEEPINFER_METAL_PREFILL_GEMM=bf16-nax
export DEEPINFER_METAL_PREFILL_GEMM_IMPL=hybrid
export DEEPINFER_METAL_PREFILL_EXPERT_IMPL=packed
export DEEPINFER_METAL_PREFILL_EXPERT_FLOW=staged
export DEEPINFER_METAL_PREFILL_EXPERT_PIPELINE=serial
export DEEPINFER_METAL_PREFILL_EXPERT_TILES=adaptive
export DEEPINFER_METAL_PREFILL_EXPERT_GATE=fused
export DEEPINFER_METAL_PREFILL_DELTA_IMPL=device
# Expert/Delta fixtures do not allocate attention KV state.
unset DEEPINFER_METAL_PREFILL_ATTENTION_IMPL DEEPINFER_METAL_PREFILL_ATTENTION_PARTITION
for pair in 'experts:prefill-experts' 'flow:expert-flow' 'delta:prefill-delta'; do
  label=${pair%%:*}; script=${pair#*:}
  bash "scripts/check-metal-${script}.sh" > "artifacts/joy-final-${label}.log" 2>&1
  tail -1 "artifacts/joy-final-${label}.log"
done
export DEEPINFER_METAL_PREFILL_ATTENTION_IMPL=bf16-nax
export DEEPINFER_METAL_PREFILL_KEY_TILE=128
export DEEPINFER_METAL_KV=bf16
bash scripts/check-metal-prefill-attention.sh > artifacts/joy-final-attention.log 2>&1
tail -1 artifacts/joy-final-attention.log
bash scripts/check-metal-bf16-long.sh > artifacts/joy-final-attention-long.log 2>&1
tail -1 artifacts/joy-final-attention-long.log
DEEPINFER_METAL_PREFILL_ATTENTION_PARTITION=2048 bash scripts/check-metal-prefill-attention.sh > artifacts/joy-final-partition.log 2>&1
tail -1 artifacts/joy-final-partition.log
