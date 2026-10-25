#!/usr/bin/env bash
# Compile-only: builds the MTP and exact-verification shader libraries/pipelines; no GPU work is executed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/native-checks
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=15.0 -fobjc-arc \
 native/DeepInfer.Metal/mtp_compile_checks.mm -framework Foundation -framework Metal \
 -o artifacts/native-checks/mtp-compile-checks
./artifacts/native-checks/mtp-compile-checks
