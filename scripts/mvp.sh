#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
base_port="${DEEPINFER_BASE_PORT:-5100}"
model_size="${DEEPINFER_MODEL_SIZE:-3b}"
case "$model_size" in
  3b|7b) model="qwen2.5-${model_size}-instruct"; weights="models/$model" ;;
  moe) model="qwen3.6-35b-a3b"; weights="models/qwen3.6-35b-a3b-5bit-mlx" ;;
  *) echo "DEEPINFER_MODEL_SIZE must be 3b, 7b or moe" >&2; exit 1 ;;
esac
host="./artifacts/native/osx-arm64/deepinfer"
tui="./artifacts/tui-native/osx-arm64/deepinfer-tui"
if [[ ! -x "$host" || ! -x "$tui" ]]; then
  dotnet build DeepInfer.slnx -c Release
  host="./src/DeepInfer.Host/bin/Release/net10.0/deepinfer"
  tui="./src/DeepInfer.Tui/bin/Release/net10.0/deepinfer-tui"
fi
mkdir -p artifacts
log_file="$(mktemp "${TMPDIR:-/tmp}/deepinfer-mvp.XXXXXX")"
if [[ "$model_size" == moe ]]; then
  if [[ ! -f "$weights/model-00005-of-00005.safetensors" ]]; then
    echo "Download the pinned MoE weights first: python3 scripts/download-qwen36-mlx.py" >&2
    exit 1
  fi
else
  python3 scripts/download-qwen.py --size "$model_size"
fi
"$host" demo --base-port "$base_port" --backend metal --weights "$weights" --tokenizer "$weights/tokenizer.json" --model "$model" --workers 1 --stay > "$log_file" 2>&1 &
cluster_pid=$!
cleanup() {
  kill -TERM "$cluster_pid" 2>/dev/null || true
  wait "$cluster_pid" 2>/dev/null || true
  rm -f "$log_file"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
ready=false
for ((i=0; i<1800; i++)); do
  if ! kill -0 "$cluster_pid" 2>/dev/null; then cat "$log_file"; exit 1; fi
  if grep -q 'Cluster stays online' "$log_file"; then ready=true; break; fi
  sleep 0.1
done
if [[ "$ready" != true ]]; then cat "$log_file"; echo "Cluster startup timed out." >&2; exit 1; fi
"$tui" "http://127.0.0.1:$base_port"
