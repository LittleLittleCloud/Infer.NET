#!/usr/bin/env bash
cd "$(dirname "$0")"
export DEEPINFER_MODEL_SIZE=7b
exec ./scripts/mvp.sh
