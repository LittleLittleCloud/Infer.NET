#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/native-checks
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=15.0 -fobjc-arc \
 native/DeepInfer.Metal/long_context_checks.mm -framework Foundation -framework Metal \
 -o artifacts/native-checks/long-context-checks
./artifacts/native-checks/long-context-checks
