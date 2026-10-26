#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Execute under an exclusive GPU maintenance window; this script does not stop services.
mkdir -p artifacts/native-checks
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=15.0 -fobjc-arc \
  native/DeepInfer.Metal/target_block_checks.mm -framework Foundation -framework Metal \
  -o artifacts/native-checks/target-block-checks
exec artifacts/native-checks/target-block-checks "$@"
