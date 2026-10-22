#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/native-checks
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=15.0 -fobjc-arc \
 native/DeepInfer.Metal/bf16_checks.mm -framework Foundation -framework Metal \
 -o artifacts/native-checks/bf16-checks
./artifacts/native-checks/bf16-checks
DEEPINFER_METAL_PREFILL_GEMM=bf16-nax bash scripts/check-metal-prefill-experts.sh
DEEPINFER_METAL_KV=bf16 bash scripts/check-metal-prefill-attention.sh
