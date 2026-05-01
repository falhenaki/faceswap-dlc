#!/bin/bash
set -euo pipefail
mkdir -p "${HF_HOME:-/workspace/hf_cache}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
exec uvicorn serve:app --host 0.0.0.0 --port "${PORT:-8000}"
