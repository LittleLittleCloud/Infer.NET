#!/usr/bin/env bash
# Compile-only: builds the DFlash2 and exact-verification shader libraries/pipelines; no GPU work is executed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/native-checks
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=15.0 -fobjc-arc \
 native/DeepInfer.Metal/dflash_compile_checks.mm -framework Foundation -framework Metal \
 -o artifacts/native-checks/dflash-compile-checks
./artifacts/native-checks/dflash-compile-checks
