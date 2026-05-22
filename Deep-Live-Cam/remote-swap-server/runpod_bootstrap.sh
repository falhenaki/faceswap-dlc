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
# Default to HyperSwap 1a 256 (current SOTA for live swap). Override via SWAP_MODEL_TYPE
# = inswapper to roll back to the original inswapper_128 path.
SWAP_MODEL_TYPE="${SWAP_MODEL_TYPE:-hyperswap}"
case "$SWAP_MODEL_TYPE" in
  hyperswap)
    DEFAULT_MODEL="$MODEL_DIR/hyperswap_1a_256.onnx"
    DEFAULT_URL="https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx?download=true"
    ;;
  inswapper)
    DEFAULT_MODEL="$MODEL_DIR/inswapper_128.onnx"
    DEFAULT_URL="https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx?download=true"
    ;;
  *)
    echo "Unknown SWAP_MODEL_TYPE=$SWAP_MODEL_TYPE (expected hyperswap|inswapper)"
    exit 1
    ;;
esac
MODEL="${SWAP_MODEL_PATH:-$DEFAULT_MODEL}"
if [[ ! -f "$MODEL" ]]; then
  echo "Downloading $SWAP_MODEL_TYPE model to $MODEL ..."
  curl -fL -o "$MODEL" "$DEFAULT_URL"
fi
export SWAP_MODEL_TYPE
export SWAP_MODEL_PATH="$MODEL"
# Backwards-compat: keep INSWAPPER_MODEL_PATH set when running inswapper, for any
# tooling that still reads it.
[[ "$SWAP_MODEL_TYPE" == "inswapper" ]] && export INSWAPPER_MODEL_PATH="$MODEL"
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
# ORT 1.17 CUDA EP probes libcublasLt.so.12 at session-init even when linked to .so.11.
# Missing .so.12 triggers SIGABRT on CUDA 11.8 images. Ship the CUDA-12 cuBLAS stubs via pip.
python3 -m pip install -q "nvidia-cublas-cu12" 2>/dev/null || true
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
for d in \
  "${_py_site}/nvidia/cudnn/lib" \
  "${_py_site}/nvidia/cudnn/lib/x86_64-linux-gnu" \
  "${_py_site}/nvidia/cublas/lib"; do
  [[ -d "$d" ]] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
done
python3 -m pip uninstall -y onnxruntime 2>/dev/null || true
python3 -m pip install -q -r "$ROOT/requirements.txt"
python3 -c "import onnxruntime as _o; print('[bootstrap] ORT providers', _o.get_available_providers())"
# Uvicorn worker count: each worker is its own Python process with its own ORT
# session, so this is how we scale GPU concurrency. Pod has 48GB+ VRAM so 4 small
# (~400MB) ONNX sessions fit easily. Single-worker pinned the throughput to ~2
# req/s due to GIL contention on the synchronous swap endpoint.
UVICORN_WORKERS="${UVICORN_WORKERS:-4}"
exec python3 -m uvicorn app:app --host 0.0.0.0 --port "${PORT}" --workers "${UVICORN_WORKERS}"
