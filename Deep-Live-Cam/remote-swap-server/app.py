"""
Minimal InsightFace-compatible inswapper HTTP service for GPU hosts (e.g. RunPod).
Tensor path matches insightface.model_zoo.inswapper.INSwapper.get (paste_back=False).
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

app = FastAPI(title="DLC Remote Inswapper", version="1.0.0")

_session: Optional[ort.InferenceSession] = None
_emap: Optional[np.ndarray] = None
_input_names: list[str] = []
_output_names: list[str] = []
_input_size_hw: tuple[int, int] = (128, 128)
_input_mean: float = 0.0
_input_std: float = 255.0


def _load_model() -> None:
    global _session, _emap, _input_names, _output_names, _input_size_hw
    path = os.environ.get("INSWAPPER_MODEL_PATH", "/workspace/models/inswapper_128.onnx")
    if not os.path.isfile(path):
        raise RuntimeError(f"Model not found: {path}")
    model = onnx.load(path)
    graph = model.graph
    _emap = numpy_helper.to_array(graph.initializer[-1])
    so = ort.SessionOptions()
    so.log_severity_level = 2  # warning; helps diagnose EP load failures in pod logs
    providers = [
        (
            "CUDAExecutionProvider",
            {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"},
        ),
        "CPUExecutionProvider",
    ]
    _session = ort.InferenceSession(path, sess_options=so, providers=providers)
    inputs = _session.get_inputs()
    outputs = _session.get_outputs()
    _input_names = [i.name for i in inputs]
    _output_names = [o.name for o in outputs]
    if len(_output_names) != 1:
        raise RuntimeError("Expected exactly one output from inswapper ONNX")
    shape = inputs[0].shape
    _input_size_hw = (int(shape[2]), int(shape[3]))
    print(
        f"[remote-swap] Loaded {path} inputs={_input_names} "
        f"input_size_hw={_input_size_hw} providers={_session.get_providers()}"
    )


def _verify_bearer(authorization: Optional[str]) -> None:
    expected = os.environ.get("SWAP_SERVICE_API_KEY", "").strip()
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: set SWAP_SERVICE_API_KEY on the pod",
        )
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
    ok = _session is not None and _emap is not None
    prov = _session.get_providers() if _session else []
    return {"ok": ok, "providers": prov, "input_hw": list(_input_size_hw)}


@app.post("/v1/swap")
def swap(
    authorization: Annotated[Optional[str], Header()] = None,
    width: int = Form(...),
    height: int = Form(...),
    aligned_bgr: UploadFile = File(...),
    embedding: UploadFile = File(...),
) -> Response:
    _verify_bearer(authorization)

    if _session is None or _emap is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw_img = aligned_bgr.file.read()
    raw_emb = embedding.file.read()
    expected_len = width * height * 3
    if len(raw_img) != expected_len:
        raise HTTPException(
            status_code=400,
            detail=f"aligned_bgr bytes {len(raw_img)} != width*height*3={expected_len}",
        )

    emb = np.frombuffer(raw_emb, dtype=np.float32)
    if emb.size < 512:
        raise HTTPException(status_code=400, detail="embedding must be at least 512 float32 values")
    emb = emb[:512].reshape(1, -1)

    aimg = np.frombuffer(raw_img, dtype=np.uint8).reshape((height, width, 3))
    exp_w, exp_h = _input_size_hw[1], _input_size_hw[0]
    if width != exp_w or height != exp_h:
        raise HTTPException(
            status_code=400,
            detail=f"Expected aligned crop {exp_w}x{exp_h}, got {width}x{height}",
        )

    blob = cv2.dnn.blobFromImage(
        aimg,
        1.0 / _input_std,
        (width, height),
        (_input_mean, _input_mean, _input_mean),
        swapRB=True,
    )
    latent = np.dot(emb, _emap)
    latent /= np.linalg.norm(latent)
    pred = _session.run(
        _output_names,
        {_input_names[0]: blob, _input_names[1]: latent},
    )[0]
    img_fake = pred.transpose((0, 2, 3, 1))[0]
    bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
    return Response(content=bgr_fake.tobytes(), media_type="application/octet-stream")
