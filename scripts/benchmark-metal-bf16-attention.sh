#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/native-checks
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=15.0 -fobjc-arc \
 native/DeepInfer.Metal/prefill_attention_bf16_bench.mm -framework Foundation -framework Metal \
 -o artifacts/native-checks/prefill-attention-bf16-bench
./artifacts/native-checks/prefill-attention-bf16-bench
