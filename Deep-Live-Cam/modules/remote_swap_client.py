"""HTTP client for remote inswapper service (RunPod / LAN GPU)."""

from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np
import requests
from requests.adapters import HTTPAdapter

from modules.typing import Frame

# A pooled session reuses the TLS connection across swap calls; without it,
# each frame pays a fresh TCP+TLS handshake to the RunPod proxy (~250ms).
_SESSION: Optional[requests.Session] = None
_SESSION_LOCK = threading.Lock()


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                s = requests.Session()
                # Enough pool capacity for several parallel swap workers.
                adapter = HTTPAdapter(pool_connections=8, pool_maxsize=16)
                s.mount("https://", adapter)
                s.mount("http://", adapter)
                _SESSION = s
    return _SESSION


def remote_swap_aligned(
    aligned_bgr: Frame,
    normed_embedding: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Send aligned face crop + source embedding; receive swapped aligned BGR crop (uint8 HxWx3).
    Env:
      DLC_REMOTE_SWAP_URL - e.g. https://xxx.proxy.runpod.net (no trailing slash required)
      DLC_REMOTE_SWAP_API_KEY - Bearer token (same value as SWAP_SERVICE_API_KEY on server)
      DLC_REMOTE_SWAP_TIMEOUT - seconds (default 30)
    """
    base = os.environ.get("DLC_REMOTE_SWAP_URL", "").strip().rstrip("/")
    key = os.environ.get("DLC_REMOTE_SWAP_API_KEY", "").strip()
    if not base or not key:
        return None

    timeout = float(os.environ.get("DLC_REMOTE_SWAP_TIMEOUT", "30"))
    h, w = aligned_bgr.shape[:2]
    if aligned_bgr.dtype != np.uint8:
        aligned_bgr = np.clip(aligned_bgr, 0, 255).astype(np.uint8)
    emb = np.ascontiguousarray(normed_embedding.astype(np.float32))

    try:
        r = _get_session().post(
            f"{base}/v1/swap",
            headers={"Authorization": f"Bearer {key}"},
            data={"width": str(w), "height": str(h)},
            files={
                "aligned_bgr": ("a.bin", aligned_bgr.tobytes(), "application/octet-stream"),
                "embedding": ("e.bin", emb.tobytes(), "application/octet-stream"),
            },
            timeout=timeout,
        )
    except requests.RequestException as e:
        print(f"[remote_swap_client] request failed: {e}")
        return None

    if r.status_code != 200:
        print(f"[remote_swap_client] HTTP {r.status_code}: {r.text[:500]}")
        return None

    out = np.frombuffer(r.content, dtype=np.uint8)
    if out.size != h * w * 3:
        print(
            f"[remote_swap_client] bad response size {out.size} expected {h * w * 3}"
        )
        return None
    return out.reshape((h, w, 3))
