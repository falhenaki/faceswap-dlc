"""HTTP API for Z-Image-Turbo (Tongyi-MAI/Z-Image-Turbo) on GPU."""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Optional

import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

MODEL_ID = os.environ.get("ZIMAGE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
TORCH_DTYPE = os.environ.get("TORCH_DTYPE", "bfloat16")  # bfloat16 | float16 | float32
API_KEY = os.environ.get("ZIMAGE_API_KEY", "").strip()

app = FastAPI(title="Z-Image-Turbo", version="1.0.0")
_bearer = HTTPBearer(auto_error=False)
_pipe = None
_ready = False


def _dtype():
    m = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return m.get(TORCH_DTYPE, torch.bfloat16)


@app.on_event("startup")
def load_model() -> None:
    global _pipe, _ready
    from diffusers import ZImagePipeline

    dt = _dtype()
    if (
        dt == torch.bfloat16
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        dt = torch.float16
        log.warning("CUDA bfloat16 not supported; using float16 instead")

    log.info("Loading %s (dtype=%s)...", MODEL_ID, dt)
    _pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dt,
        low_cpu_mem_usage=False,
    )
    if os.environ.get("ENABLE_MODEL_CPU_OFFLOAD", "").lower() in ("1", "true", "yes"):
        log.info("enable_model_cpu_offload()")
        _pipe.enable_model_cpu_offload()
    else:
        _pipe.to("cuda")

    backend = os.environ.get("ATTENTION_BACKEND", "").strip()
    if backend and hasattr(_pipe.transformer, "set_attention_backend"):
        log.info("Attention backend: %s", backend)
        _pipe.transformer.set_attention_backend(backend)

    _ready = True
    log.info("Model ready.")


def require_api_key(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)) -> None:
    if not API_KEY:
        return
    if creds is None or creds.scheme.lower() != "bearer" or creds.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


@app.get("/health")
def health():
    if not _ready or _pipe is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "model": MODEL_ID}


class GenerateBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    width: int = Field(1024, ge=256, le=2048, multiple_of=8)
    height: int = Field(1024, ge=256, le=2048, multiple_of=8)
    num_inference_steps: int = Field(9, ge=1, le=50)
    guidance_scale: float = Field(0.0, ge=0.0, le=15.0)
    seed: Optional[int] = None


@app.post("/generate")
def generate(body: GenerateBody, _: None = Depends(require_api_key)):
    if not _ready or _pipe is None:
        raise HTTPException(503, detail="Model still loading")

    gen = torch.Generator(device="cuda")
    if body.seed is not None:
        gen.manual_seed(body.seed)

    try:
        out = _pipe(
            prompt=body.prompt,
            height=body.height,
            width=body.width,
            num_inference_steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
            generator=gen,
        )
    except Exception as e:
        log.exception("Inference failed")
        raise HTTPException(500, detail=str(e)) from e

    image = out.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return {
        "format": "png",
        "width": body.width,
        "height": body.height,
        "image_base64": b64,
    }
