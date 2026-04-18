"""Thin wrapper around pyvirtualcam for piping live-swapped frames to Zoom / Meet.

On macOS this uses the OBS virtual-camera backend, so OBS must be installed
once and the Virtual Camera extension initialised (Start Virtual Camera in OBS
at least once). After that, DLC writes frames directly into the OBS virtual
camera; Zoom/Meet select "OBS Virtual Camera" as their camera source.

The module is optional: if pyvirtualcam is not installed, or the backend is
unavailable, all calls are silent no-ops so live preview still works.

Enable/disable via environment:
  DLC_VIRTUALCAM=0   disable even if pyvirtualcam is installed
  DLC_VIRTUALCAM_FPS override target FPS (default 20)
"""

from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np


class VirtualCam:
    def __init__(self) -> None:
        self._cam = None
        self._w = 0
        self._h = 0
        self._fps = int(os.environ.get("DLC_VIRTUALCAM_FPS", "20"))
        self._lock = threading.Lock()
        self._disabled = os.environ.get("DLC_VIRTUALCAM", "1") == "0"
        self._last_error: Optional[str] = None

    @property
    def active(self) -> bool:
        return self._cam is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _ensure(self, w: int, h: int) -> None:
        if self._disabled:
            return
        if self._cam is not None and (self._w, self._h) == (w, h):
            return
        try:
            import pyvirtualcam  # type: ignore
        except Exception as e:  # pyvirtualcam missing
            self._disabled = True
            self._last_error = f"pyvirtualcam not installed: {e}"
            return
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
        try:
            self._cam = pyvirtualcam.Camera(
                width=w,
                height=h,
                fps=self._fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
            self._w, self._h = w, h
            self._last_error = None
            print(f"[virtual_camera] attached to backend '{self._cam.backend}' {w}x{h}@{self._fps}")
        except Exception as e:
            self._cam = None
            self._disabled = True
            self._last_error = f"backend unavailable: {e}"
            print(f"[virtual_camera] disabled: {self._last_error}")

    def send(self, frame_bgr: np.ndarray) -> None:
        if self._disabled:
            return
        if frame_bgr is None or frame_bgr.ndim != 3:
            return
        h, w = frame_bgr.shape[:2]
        with self._lock:
            self._ensure(w, h)
            if self._cam is None:
                return
            try:
                if frame_bgr.dtype != np.uint8:
                    frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
                if not frame_bgr.flags["C_CONTIGUOUS"]:
                    frame_bgr = np.ascontiguousarray(frame_bgr)
                self._cam.send(frame_bgr)
                self._cam.sleep_until_next_frame()
            except Exception as e:
                self._last_error = str(e)
                # Drop on write error; retry on next frame by resetting.
                try:
                    self._cam.close()
                except Exception:
                    pass
                self._cam = None

    def close(self) -> None:
        with self._lock:
            if self._cam is not None:
                try:
                    self._cam.close()
                except Exception:
                    pass
                self._cam = None


_singleton: Optional[VirtualCam] = None


def get_virtual_cam() -> VirtualCam:
    global _singleton
    if _singleton is None:
        _singleton = VirtualCam()
    return _singleton


def send_frame(frame_bgr: np.ndarray) -> None:
    get_virtual_cam().send(frame_bgr)


def close() -> None:
    global _singleton
    if _singleton is not None:
        _singleton.close()
