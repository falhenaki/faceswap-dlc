#!/usr/bin/env bash
# On a RunPod GPU pod (PyTorch+CUDA template), from workspace:
#   git clone <your fork with Deep-Live-Cam> dlc && cd dlc/remote-swap-server
#   export SWAP_SERVICE_API_KEY='your-secret'
#   bash runpod_bootstrap.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
# Avoid "address already in use" when re-running bootstrap on a live pod.
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi
pkill -f '[u]vicorn app:app' 2>/dev/null || true
sleep 1
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
python3 -c "import torch; print('[bootstrap] torch.cuda.is_available=', torch.cuda.is_available(), 'count=', torch.cuda.device_count())" 2>/dev/null || true
for d in /usr/local/cuda/lib64 /usr/local/cuda-12/lib64 /usr/local/cuda-11.8/lib64 /usr/lib/x86_64-linux-gnu; do
  [[ -d "$d" ]] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq || true
  apt-get install -y -qq libcudnn8 2>/dev/null || true
  ldconfig 2>/dev/null || true
fi
python3 -m pip install -q -U pip
# CuDNN libs for ONNX Runtime CUDA EP (ORT needs libcudnn.so.8 next to CUDA).
python3 -m pip install -q "nvidia-cudnn-cu11==8.9.7.29" 2>/dev/null || true
CUDNN_LIB="$(python3 -c "
try:
 import os as _os
 import nvidia.cudnn as _c
 _p = _os.path.join(_os.path.dirname(_c.__file__), 'lib')
 print(_p if _os.path.isdir(_p) else '')
except Exception:
 print('')
" 2>/dev/null || true)"
[[ -n "${CUDNN_LIB:-}" ]] && export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
_py_site="$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
for d in "${_py_site}/nvidia/cudnn/lib" "${_py_site}/nvidia/cudnn/lib/x86_64-linux-gnu"; do
  [[ -d "$d" ]] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
python3 -m pip uninstall -y onnxruntime 2>/dev/null || true
python3 -m pip install -q -r "$ROOT/requirements.txt"
python3 -c "import onnxruntime as _o; print('[bootstrap] ORT providers', _o.get_available_providers())"
exec python3 -m uvicorn app:app --host 0.0.0.0 --port "${PORT}" --workers 1
