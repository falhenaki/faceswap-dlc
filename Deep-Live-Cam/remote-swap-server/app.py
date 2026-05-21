"""
Minimal face-swap HTTP service for GPU hosts (e.g. RunPod).

Supports two model families, selected by SWAP_MODEL_TYPE:
  - inswapper  (default): InsightFace inswapper_128.onnx. Source = dot(emb, emap),
    image normalized to [0,1], output clipped to [0,255].
  - hyperswap            : FaceFusion hyperswap_*_256.onnx. Source = raw normed
    embedding, image normalized to [-1,1] (mean=0.5, std=0.5), output denormalized.

Tensor names are the same across both families ('source', 'target', single output).
"""

from __future__ import annotations

import os
import secrets
from typing import Annotated, Optional

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from onnx import numpy_helper

app = FastAPI(title="DLC Remote Face-Swap", version="2.0.0")

_session: Optional[ort.InferenceSession] = None
_model_type: str = "inswapper"
_emap: Optional[np.ndarray] = None  # only used for inswapper
_input_names: list[str] = []
_output_names: list[str] = []
_input_size_hw: tuple[int, int] = (128, 128)


def _resolve_model_path() -> tuple[str, str]:
    """Return (path, model_type). Prefer SWAP_MODEL_PATH; fall back to INSWAPPER_MODEL_PATH."""
    model_type = os.environ.get("SWAP_MODEL_TYPE", "").strip().lower()
    path = os.environ.get("SWAP_MODEL_PATH", "").strip()
    if not path:
        # backwards compat with the original inswapper-only deployment
        path = os.environ.get("INSWAPPER_MODEL_PATH", "/workspace/models/inswapper_128.onnx")
    if not model_type:
        model_type = "hyperswap" if "hyperswap" in os.path.basename(path).lower() else "inswapper"
    if model_type not in ("inswapper", "hyperswap"):
        raise RuntimeError(f"Unsupported SWAP_MODEL_TYPE={model_type!r}")
    return path, model_type


def _load_model() -> None:
    global _session, _model_type, _emap, _input_names, _output_names, _input_size_hw

    path, model_type = _resolve_model_path()
    if not os.path.isfile(path):
        raise RuntimeError(f"Model not found: {path}")

    _model_type = model_type
    if model_type == "inswapper":
        # inswapper carries its identity-projection matrix in the last graph initializer.
        graph = onnx.load(path).graph
        _emap = numpy_helper.to_array(graph.initializer[-1])
    else:
        _emap = None

    so = ort.SessionOptions()
    so.log_severity_level = 2
    providers = [
        ("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"}),
        "CPUExecutionProvider",
    ]
    _session = ort.InferenceSession(path, sess_options=so, providers=providers)
    inputs = _session.get_inputs()
    outputs = _session.get_outputs()
    _input_names = [i.name for i in inputs]
    # HyperSwap emits two outputs ('output' image + 'mask'); we only need the image.
    # Inswapper emits one output. Either way, take the first.
    _output_names = [outputs[0].name]
    img_in = next(i for i in inputs if len(i.shape) == 4)
    _input_size_hw = (int(img_in.shape[2]), int(img_in.shape[3]))
    print(
        f"[remote-swap] Loaded {model_type} from {path} "
        f"inputs={_input_names} input_hw={_input_size_hw} "
        f"providers={_session.get_providers()}"
    )


def _verify_bearer(authorization: Optional[str]) -> None:
    expected = os.environ.get("SWAP_SERVICE_API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail="Server misconfigured: set SWAP_SERVICE_API_KEY")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer …")
    token = authorization[7:].strip()
    if not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.on_event("startup")
def startup() -> None:
    _load_model()


@app.get("/health")
def health() -> dict:
    ok = _session is not None
    prov = _session.get_providers() if _session else []
    out: dict = {
        "ok": ok,
        "model_type": _model_type,
        "providers": prov,
        "input_hw": list(_input_size_hw),
    }
    try:
        import torch

        out["torch_cuda"] = torch.cuda.is_available()
        out["torch_device_count"] = torch.cuda.device_count()
    except Exception:
        pass
    return out


def _prepare_image(aimg: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 -> float32 1x3xHxW with model-specific normalization."""
    # BGR -> RGB and to float[0,1]
    rgb = aimg[:, :, ::-1].astype(np.float32) / 255.0
    if _model_type == "hyperswap":
        # FaceFusion's prepare_crop_frame: (x - 0.5) / 0.5  -> [-1, 1]
        rgb = (rgb - 0.5) / 0.5
    # else inswapper expects [0,1]
    blob = rgb.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(blob, dtype=np.float32)


def _prepare_source(emb: np.ndarray) -> np.ndarray:
    """Convert client-supplied normed embedding to whatever the model wants."""
    emb = emb[:512].reshape(1, -1).astype(np.float32)
    if _model_type == "inswapper":
        latent = np.dot(emb, _emap)
        latent /= np.linalg.norm(latent)
        return latent
    # hyperswap: raw normed embedding
    return emb


def _postprocess(pred: np.ndarray) -> np.ndarray:
    """1x3xHxW float -> HxWx3 uint8 BGR."""
    img = pred[0].transpose(1, 2, 0)
    if _model_type == "hyperswap":
        # FaceFusion's normalize_crop_frame: x * 0.5 + 0.5 -> [0, 1]
        img = img * 0.5 + 0.5
    # both: clip [0,1], RGB->BGR, *255
    img = np.clip(img, 0.0, 1.0)
    return (img[:, :, ::-1] * 255.0).astype(np.uint8)


@app.post("/v1/swap")
def swap(
    authorization: Annotated[Optional[str], Header()] = None,
    width: int = Form(...),
    height: int = Form(...),
    aligned_bgr: UploadFile = File(...),
    embedding: UploadFile = File(...),
) -> Response:
    _verify_bearer(authorization)
    if _session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw_img = aligned_bgr.file.read()
    raw_emb = embedding.file.read()

    expected_len = width * height * 3
    if len(raw_img) != expected_len:
        raise HTTPException(
            status_code=400,
            detail=f"aligned_bgr bytes {len(raw_img)} != width*height*3={expected_len}",
        )

    exp_h, exp_w = _input_size_hw
    if width != exp_w or height != exp_h:
        raise HTTPException(
            status_code=400,
            detail=f"Expected aligned crop {exp_w}x{exp_h}, got {width}x{height}",
        )

    emb = np.frombuffer(raw_emb, dtype=np.float32)
    if emb.size < 512:
        raise HTTPException(status_code=400, detail="embedding must be at least 512 float32 values")

    aimg = np.frombuffer(raw_img, dtype=np.uint8).reshape((height, width, 3))

    blob = _prepare_image(aimg)
    source = _prepare_source(emb)

    # Map by tensor name so we work with any input ordering.
    feeds = {}
    for n in _input_names:
        if n == "target":
            feeds[n] = blob
        elif n == "source":
            feeds[n] = source
        else:
            raise HTTPException(status_code=500, detail=f"Unknown ONNX input name {n!r}")

    pred = _session.run(_output_names, feeds)[0]
    bgr_fake = _postprocess(pred)
    return Response(content=bgr_fake.tobytes(), media_type="application/octet-stream")
