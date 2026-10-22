#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
xcrun clang++ -std=c++17 -O2 -mmacosx-version-min=15.0 -fobjc-arc -dynamiclib bridge.mm \
  -framework Foundation -framework Metal -o build/libdeepinfer_metal.dylib
