#!/usr/bin/env bash
# On a RunPod GPU pod (PyTorch+CUDA template), from workspace:
#   git clone <your fork with Deep-Live-Cam> dlc && cd dlc/remote-swap-server
#   export SWAP_SERVICE_API_KEY='your-secret'
#   bash runpod_bootstrap.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
mkdir -p "$MODEL_DIR"
MODEL="${INSWAPPER_MODEL_PATH:-$MODEL_DIR/inswapper_128.onnx}"
if [[ ! -f "$MODEL" ]]; then
  echo "Downloading inswapper to $MODEL ..."
  curl -fL -o "$MODEL" "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx?download=true"
fi
export INSWAPPER_MODEL_PATH="$MODEL"
if [[ -z "${SWAP_SERVICE_API_KEY:-}" ]]; then
  echo "Set SWAP_SERVICE_API_KEY before starting (same value as DLC_REMOTE_SWAP_API_KEY on your Mac)."
  exit 1
fi
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "[bootstrap] nvidia-smi not found (GPU driver?)"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
python3 -m pip install -q -U pip
python3 -m pip uninstall -y onnxruntime 2>/dev/null || true
python3 -m pip install -q -r "$ROOT/requirements.txt"
python3 -c "import onnxruntime as _o; print('[bootstrap] ORT providers', _o.get_available_providers())"
exec python3 -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
