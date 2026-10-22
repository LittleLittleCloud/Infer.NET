#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/native-checks
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=15.0 -fobjc-arc \
 native/DeepInfer.Metal/bf16_attention_long_checks.mm -framework Foundation -framework Metal \
 -o artifacts/native-checks/bf16-attention-long-checks
./artifacts/native-checks/bf16-attention-long-checks
