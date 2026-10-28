#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:---cpu}"
out="${2:-artifacts/dual-host/issue103-lmhead-affine8/$(date -u +%Y%m%dT%H%M%SZ)-${mode#--}}"
case "$mode" in --cpu|--compile|--gpu|--bench) ;; *) echo "Use --cpu, --compile, --gpu or --bench [NEW_OUTPUT_DIRECTORY]" >&2; exit 2 ;; esac
mkdir -p "$(dirname "$out")"
mkdir "$out"
out="$(cd "$out" && pwd)"
{
  date -u
  git rev-parse HEAD
  git status --short
  printf 'mode=%s; GPU execution requires parent-exclusive scheduling; compile submits no GPU work\n' "$mode"
  shasum -a 256 native/DeepInfer.Metal/lmhead_affine8* scripts/check-metal-lmhead.sh
  shasum -a 256 native/DeepInfer.Metal/{bridge,quantized,gpu_graph,gpu_prefill,residency}.mm \
    src/DeepInfer.Backends/{Compute/Qwen36Gpu.cs,Models/Qwen36Model.cs,DeepInfer.Backends.csproj} \
    tests/DeepInfer.Tests/LmHeadSelectorTests.cs docs/evaluations/issue103/implementation.md site/experiments.json
} > "$out/manifest.txt"
git diff --binary > "$out/source.patch"
run() {
  local name="$1"; shift
  printf '%q ' "$@" > "$out/$name.command"; printf '\n' >> "$out/$name.command"
  set +e
  "$@" 2>&1 | tee "$out/$name.log"
  local statuses=("${PIPESTATUS[@]}")
  set -e
  printf '%s\n' "${statuses[0]}" > "$out/$name.exit"
  if [[ "${statuses[0]}" != 0 ]]; then exit "${statuses[0]}"; fi
  if [[ "${statuses[1]}" != 0 ]]; then exit "${statuses[1]}"; fi
}
if [[ "$mode" == --cpu ]]; then
  run cpu-build xcrun clang++ -std=c++17 -O2 -Wall -Wextra -Werror -ffp-contract=off \
    native/DeepInfer.Metal/lmhead_affine8_checks.cpp -o "$out/lmhead-cpu"
  run cpu-checks "$out/lmhead-cpu"
else
  run driver-build xcrun clang++ -std=c++17 -O2 -ffp-contract=off -mmacosx-version-min=15.0 -fobjc-arc \
    native/DeepInfer.Metal/lmhead_affine8_driver.mm -framework Foundation -framework Metal -o "$out/lmhead-driver"
  shasum -a 256 "$out/lmhead-driver" > "$out/binary-sha256.txt"
  if [[ "$mode" == --compile ]]; then
    run native-build bash native/DeepInfer.Metal/build.sh
    shasum -a 256 native/DeepInfer.Metal/build/libdeepinfer_metal.dylib >> "$out/binary-sha256.txt"
    run metal-compile "$out/lmhead-driver" --compile-only
  elif [[ "$mode" == --gpu ]]; then
    run gpu-checks "$out/lmhead-driver" --gpu
  else
    run microbench "$out/lmhead-driver" --bench 2 5 128
  fi
fi
shasum -a 256 "$out"/*.log > "$out/log-sha256.txt"
echo "Raw logs: $out"
