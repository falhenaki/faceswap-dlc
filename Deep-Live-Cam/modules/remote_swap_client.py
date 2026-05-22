"""HTTP and WebSocket client for the remote swap service (RunPod / LAN GPU)."""

from __future__ import annotations

import os
import struct
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
      DLC_REMOTE_SWAP_PROTOCOL - 'http' (default) or 'ws' for persistent WebSocket.
        WS skips per-request multipart + HTTP overhead; ~3-5x faster on the
        L40S+HyperSwap path. Each calling thread keeps its own WS open.
    """
    if os.environ.get("DLC_REMOTE_SWAP_PROTOCOL", "http").strip().lower() == "ws":
        return remote_swap_aligned_ws(aligned_bgr, normed_embedding)

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


# --- WebSocket variant ---------------------------------------------------------
# Avoids per-request multipart + HTTP overhead by holding one persistent binary
# WS connection per worker thread. The wire format mirrors the server's
# /v1/ws/swap endpoint:
#   request : [u32 w][u32 h][u32 emb_bytes][float32 emb][uint8 w*h*3 bgr]
#   response: [uint8 w*h*3 bgr]   (or a text "error: ..." string on failure)

_WS_HEADER = "<III"  # little-endian uint32 x 3
_ws_tls = threading.local()


def _ws_url(base: str) -> str:
    # Convert https://...proxy.runpod.net -> wss://...proxy.runpod.net/v1/ws/swap
    if base.startswith("https://"):
        scheme = "wss://"
        rest = base[len("https://"):]
    elif base.startswith("http://"):
        scheme = "ws://"
        rest = base[len("http://"):]
    else:
        scheme = "wss://"
        rest = base
    return f"{scheme}{rest}/v1/ws/swap"


def _get_ws():
    """One WebSocket per worker thread, lazily opened, reconnected on failure."""
    ws = getattr(_ws_tls, "ws", None)
    if ws is not None and ws.connected:
        return ws
    import websocket  # type: ignore

    base = os.environ.get("DLC_REMOTE_SWAP_URL", "").strip().rstrip("/")
    key = os.environ.get("DLC_REMOTE_SWAP_API_KEY", "").strip()
    if not base or not key:
        return None
    url = _ws_url(base)
    ws = websocket.WebSocket()
    timeout = float(os.environ.get("DLC_REMOTE_SWAP_TIMEOUT", "30"))
    ws.connect(url, header=[f"Authorization: Bearer {key}"], timeout=timeout)
    ws.settimeout(timeout)
    _ws_tls.ws = ws
    return ws


def remote_swap_aligned_ws(
    aligned_bgr: Frame,
    normed_embedding: np.ndarray,
) -> Optional[np.ndarray]:
    """WebSocket variant of remote_swap_aligned. Same return contract."""
    base = os.environ.get("DLC_REMOTE_SWAP_URL", "").strip().rstrip("/")
    key = os.environ.get("DLC_REMOTE_SWAP_API_KEY", "").strip()
    if not base or not key:
        return None

    h, w = aligned_bgr.shape[:2]
    if aligned_bgr.dtype != np.uint8:
        aligned_bgr = np.clip(aligned_bgr, 0, 255).astype(np.uint8)
    if not aligned_bgr.flags["C_CONTIGUOUS"]:
        aligned_bgr = np.ascontiguousarray(aligned_bgr)
    emb = np.ascontiguousarray(normed_embedding.astype(np.float32))
    emb_bytes = emb.nbytes

    header = struct.pack(_WS_HEADER, w, h, emb_bytes)
    frame = header + emb.tobytes() + aligned_bgr.tobytes()

    # One reconnect attempt on transient failure (proxy idle drop, etc.).
    for attempt in (0, 1):
        try:
            ws = _get_ws()
            if ws is None:
                return None
            ws.send_binary(frame)
            resp = ws.recv()
            break
        except Exception as e:
            # Mark socket dead; reconnect on retry.
            try:
                if getattr(_ws_tls, "ws", None) is not None:
                    _ws_tls.ws.close()
            except Exception:
                pass
            _ws_tls.ws = None
            if attempt == 1:
                print(f"[remote_swap_client] ws failed: {e}")
                return None

    if isinstance(resp, (bytes, bytearray)):
        out = np.frombuffer(resp, dtype=np.uint8)
        if out.size != h * w * 3:
            print(
                f"[remote_swap_client] bad ws response size {out.size} expected {h*w*3}"
            )
            return None
        return out.reshape((h, w, 3))
    # text frame = error from server
    print(f"[remote_swap_client] ws error: {resp!r}")
    return None


def close_thread_ws() -> None:
    """Optional: workers can call this at shutdown to close their connection."""
    ws = getattr(_ws_tls, "ws", None)
    if ws is not None:
        try:
            ws.close()
        except Exception:
            pass
        _ws_tls.ws = None
