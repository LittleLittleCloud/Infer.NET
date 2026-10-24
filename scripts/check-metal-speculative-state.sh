#!/usr/bin/env bash
# Compiles the graph state-transaction checks; pass --run to execute them (uses Metal buffers, no model weights).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/native-checks
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=15.0 -fobjc-arc \
  native/DeepInfer.Metal/speculative_state_checks.mm -framework Foundation -framework Metal \
  -o artifacts/native-checks/speculative-state-checks
if [[ "${1:-}" == "--run" ]]; then ./artifacts/native-checks/speculative-state-checks; fi
