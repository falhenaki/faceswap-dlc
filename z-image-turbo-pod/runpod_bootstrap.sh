#!/usr/bin/env bash
# Run on the pod after git clone into this directory (see terraform/bootstrap_on_pod.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PORT="${PORT:-8000}"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
mkdir -p "$HF_HOME"

if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq || true
  apt-get install -y -qq git curl ca-certificates build-essential 2>/dev/null || true
fi

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi
pkill -f '[u]vicorn serve:app' 2>/dev/null || true
sleep 1

python3 -m pip install -q -U pip setuptools wheel
python3 -m pip install -q -r "$ROOT/requirements.txt"

exec python3 -m uvicorn serve:app --host 0.0.0.0 --port "$PORT" --workers 1
