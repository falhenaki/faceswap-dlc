#!/usr/bin/env python3
"""Headless live webcam face-swap (local CoreML or remote RunPod GPU).

Decoupled pipeline so display rate doesn't get pinned to swap rate:
  - Main: capture frame -> dispatch to detect_q -> composite using
    latest_face + latest_crop -> display + OBS Virtual Camera. Runs at the
    camera's native rate (~30 fps).
  - Detector thread: pops from detect_q at its own rate (~10-15 fps with
    CoreML); writes the latest Face position to shared state.
  - Swap workers (N): pop face+frame at their rate (limited by backend —
    RunPod proxy caps at ~4 req/s, local CoreML HyperSwap at ~1.5 req/s);
    compute just the aligned swap *crop* (no paste-back) and write it to
    shared state.
  - Composite step uses CURRENT detection + LATEST crop, so the face
    position tracks your live motion while the swap content updates at
    whatever rate the backend supports. Eliminates the "swap pasted at an
    old position" lag that pinned us to swap_rate display.

Usage (after sourcing env.remote so DLC_REMOTE_SWAP_URL / DLC_REMOTE_SWAP_API_KEY are set):
    venv/bin/python live_remote.py --source media/sources/user_source.jpg --mirror [--workers 3]

Press 'q' in the preview window to quit.
"""
import argparse
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

# InsightFace landmark_2d_106 region indices (verified by matching centroid against
# face.kps eye keypoints: see /tmp/dlc_landmarks_labeled.jpg).
# 43-51 and 96-104 are the *eyebrows* above these, not the eyes.
EYE_LEFT_IDX = list(range(33, 43))   # 10 pts around the viewer-left eye
EYE_RIGHT_IDX = list(range(87, 96))  # 9 pts around the viewer-right eye


def build_eye_mask(face, h: int, w: int, expand: float = 1.15, blur: int = 21) -> np.ndarray:
    """Soft mask covering both eye regions in the *original* frame's coords.

    `expand` enlarges each eye polygon around its centroid so eyelids and lashes
    get included; `blur` feathers the edge for a seamless paste.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None or not isinstance(lm, np.ndarray) or lm.shape[0] < 106:
        return mask
    for idx in (EYE_LEFT_IDX, EYE_RIGHT_IDX):
        pts = lm[idx].astype(np.float32)
        if not np.all(np.isfinite(pts)):
            continue
        c = pts.mean(axis=0)
        pts = (c + (pts - c) * expand).astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)
    blur = max(1, blur // 2 * 2 + 1)
    return cv2.GaussianBlur(mask, (blur, blur), 0)


def passthrough_eyes(swapped: np.ndarray, original: np.ndarray, face,
                     expand: float = 1.15) -> np.ndarray:
    """Composite the original frame's eye pixels back over the swapped frame."""
    h, w = swapped.shape[:2]
    mask = build_eye_mask(face, h, w, expand=expand)
    if mask.max() == 0:
        return swapped
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    return (alpha * original.astype(np.float32) +
            (1.0 - alpha) * swapped.astype(np.float32)).astype(np.uint8)


class LocalSwapper:
    """Loads inswapper_128 or hyperswap_*_256 ONNX directly via ORT.

    Mirrors what remote-swap-server/app.py does on the pod, so the local
    backend can use the same model selection. DLC's bundled face_swapper is
    inswapper-only — this lets us A/B against HyperSwap on the Mac.
    """

    def __init__(self, model_path: str, providers: list):
        from insightface.utils import face_align as _fa  # local-only dep
        self._face_align = _fa
        self.model_type = "hyperswap" if "hyperswap" in os.path.basename(model_path).lower() else "inswapper"
        so = ort.SessionOptions()
        so.log_severity_level = 2
        self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_names = [i.name for i in inputs]
        # HyperSwap returns ('output', 'mask'); inswapper returns ('output',). Take the first.
        self.output_names = [outputs[0].name]
        img_in = next(i for i in inputs if len(i.shape) == 4)
        self.input_size = int(img_in.shape[2])
        # inswapper's identity-projection matrix is the last initializer; hyperswap doesn't have it.
        self._emap = None
        if self.model_type == "inswapper":
            g = onnx.load(model_path).graph
            self._emap = numpy_helper.to_array(g.initializer[-1])
        print(f"[live] local swapper: {self.model_type} {self.input_size}x{self.input_size}  providers={[p for p in self.session.get_providers()]}")

    def _prep_image(self, aimg: np.ndarray) -> np.ndarray:
        rgb = aimg[:, :, ::-1].astype(np.float32) / 255.0
        if self.model_type == "hyperswap":
            rgb = (rgb - 0.5) / 0.5
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...].astype(np.float32))

    def _prep_source(self, normed_embedding: np.ndarray) -> np.ndarray:
        emb = normed_embedding.reshape(1, -1).astype(np.float32)
        if self.model_type == "inswapper":
            latent = np.dot(emb, self._emap)
            latent /= np.linalg.norm(latent)
            return latent
        return emb  # hyperswap takes raw normed embedding

    def _postprocess(self, pred: np.ndarray) -> np.ndarray:
        img = pred[0].transpose(1, 2, 0)
        if self.model_type == "hyperswap":
            img = img * 0.5 + 0.5
        img = np.clip(img, 0.0, 1.0)
        return (img[:, :, ::-1] * 255.0).astype(np.uint8)

    def compute_crop(self, source_face, target_face, frame: np.ndarray) -> np.ndarray:
        """Run inference only — return the aligned swapped crop (HxWx3 uint8 BGR).
        Caller is responsible for pasting it back into the target frame.
        """
        aimg, _M = self._face_align.norm_crop2(frame, target_face.kps, self.input_size)
        feeds = {}
        for n in self.input_names:
            if n == "target":
                feeds[n] = self._prep_image(aimg)
            elif n == "source":
                feeds[n] = self._prep_source(source_face.normed_embedding)
            else:
                raise RuntimeError(f"unknown ONNX input name {n!r}")
        pred = self.session.run(self.output_names, feeds)[0]
        return self._postprocess(pred)

    def swap(self, source_face, target_face, frame: np.ndarray) -> np.ndarray:
        """Compute crop + paste into the same frame. Convenience wrapper."""
        crop = self.compute_crop(source_face, target_face, frame)
        return paste_swap_crop(crop, target_face, frame, self.input_size, self._face_align)


def paste_swap_crop(crop: np.ndarray, target_face, dest_frame: np.ndarray,
                    input_size: int, face_align_mod) -> np.ndarray:
    """Paste an aligned swap crop onto `dest_frame` at the face position given
    by target_face.kps. Uses a face-shaped mask (built from target_face's 106
    landmarks) so the seam follows the jawline+forehead rather than a square.

    The crop was generated for the arcface-128 template, so the inverse of
    `norm_crop2(dest_frame, target_kps, input_size)`'s M maps it back.
    """
    h, w = dest_frame.shape[:2]
    _aimg, M_dest = face_align_mod.norm_crop2(dest_frame, target_face.kps, input_size)
    M_inv = cv2.invertAffineTransform(M_dest)
    warped = cv2.warpAffine(crop, M_inv, (w, h), borderValue=(0, 0, 0))

    # Face-shaped mask from 106-point landmarks (convex hull of jaw + forehead
    # estimate, gaussian-blurred for a soft seam). Built directly in frame
    # coords from the current detection — no warp needed.
    mask = _build_face_mask(target_face, h, w)
    # AND it with the warped crop's footprint so we don't try to blend pixels
    # the warpAffine left as black border.
    crop_footprint = cv2.warpAffine(
        np.full((input_size, input_size), 255, dtype=np.uint8),
        M_inv, (w, h),
    )
    mask = cv2.bitwise_and(mask, crop_footprint)

    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    return (dest_frame.astype(np.float32) * (1 - alpha) +
            warped.astype(np.float32) * alpha).astype(np.uint8)


def _build_face_mask(face, h: int, w: int, blur: int = 31) -> np.ndarray:
    """Soft mask of the face region using insightface's 106-point landmarks
    (convex hull of jawline 0-32 + an estimated forehead row above the eyebrows).
    Mirrors DLC's create_face_mask, lifted inline to keep this module dep-light.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    lm = getattr(face, "landmark_2d_106", None)
    if lm is None or not isinstance(lm, np.ndarray) or lm.shape[0] < 106:
        return mask
    if not np.all(np.isfinite(lm)):
        return mask
    try:
        landmarks_int = lm.astype(np.int32)
        face_outline = landmarks_int[0:33]
        eyebrows = landmarks_int[33:43]
        if eyebrows.shape[0] > 0:
            chin = landmarks_int[16]
            eyebrow_center = np.mean(eyebrows, axis=0).astype(np.int32)
            up_vector = eyebrow_center - chin
            norm = float(np.linalg.norm(up_vector))
            if norm > 0:
                up_vector = up_vector.astype(np.float32) / norm
                forehead_offset = up_vector * (norm * 1.0)
                forehead_points = (eyebrows + forehead_offset).astype(np.int32)
                top_center = np.mean(forehead_points, axis=0).astype(np.int32)
                forehead_points = ((forehead_points - top_center) * 1.2 + top_center).astype(np.int32)
                face_outline = np.concatenate((face_outline, forehead_points), axis=0)
        hull = cv2.convexHull(face_outline.astype(np.float32))
        if hull is None or len(hull) < 3:
            return mask
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        k = max(1, blur // 2 * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    except Exception as e:
        print(f"[face_mask] {e}", file=sys.stderr)
    return mask

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import modules.globals  # noqa: E402

from modules.core import default_execution_providers, decode_execution_providers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source", required=True, help="path to source face image")
    ap.add_argument("-c", "--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("-W", "--width", type=int, default=960)
    ap.add_argument("-H", "--height", type=int, default=540)
    ap.add_argument("--mirror", action="store_true", help="horizontally flip the preview")
    ap.add_argument("--backend", choices=["remote", "local"], default="remote",
                    help="remote = RunPod /v1/swap (needs env.remote); local = on-device CoreML/CUDA")
    ap.add_argument("--remote-protocol", choices=["http", "ws"], default="ws",
                    help="remote wire protocol. 'ws' (default) holds a persistent binary "
                         "WebSocket per worker — eliminates per-request multipart + HTTP "
                         "overhead. 'http' is the older POST /v1/swap path.")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel swap workers; default 3 for remote, 1 for local")
    ap.add_argument("--no-eye-passthrough", action="store_true",
                    help="disable eye passthrough (default: on; preserves blinks by "
                         "blending the original frame's eye region onto the swap)")
    ap.add_argument("--eye-expand", type=float, default=1.15,
                    help="how much to enlarge the eye polygon around its centroid (default 1.15)")
    ap.add_argument("--timing", action="store_true",
                    help="print per-frame detect/swap/eyes timing to stderr")
    ap.add_argument("--model", choices=["auto", "inswapper", "hyperswap"], default="auto",
                    help="local swap model. auto = hyperswap if hyperswap_1a_256.onnx exists, else inswapper. "
                         "Only used with --backend local (remote pod picks via SWAP_MODEL_TYPE).")
    ap.add_argument("--swap-model-path", default=None,
                    help="explicit local ONNX path; overrides --model auto-resolution")
    ap.add_argument("--det-size", type=int, default=640,
                    help="square detection input size. NOTE: CoreML EP for det_10g.onnx "
                         "currently only works at 640; smaller sizes crash with a shape "
                         "inference error. Kept here for CPU/CUDA runs (default 640).")
    ap.add_argument("--no-virtual-cam", action="store_true",
                    help="don't publish the swapped feed as an OBS Virtual Camera "
                         "(default: on if OBS Virtual Camera is available)")
    ap.add_argument("--audio-sync", action="store_true",
                    help="capture mic -> emit to --audio-output with delay auto-tracking "
                         "the live video latency (requires BlackHole or similar virtual "
                         "audio device; point Zoom's mic at the same device)")
    ap.add_argument("--audio-input", default=None,
                    help="substring of the audio input (mic) device name; default = "
                         "system default mic. Try 'AirPods' if MacBook speakers are "
                         "causing echo (mic picks up your Zoom output).")
    ap.add_argument("--audio-output", default="BlackHole",
                    help="substring of the audio output device name to send delayed "
                         "audio to (default: BlackHole)")
    ap.add_argument("--audio-delay-ms", type=float, default=None,
                    help="fixed audio delay in ms; default: auto-track video latency")
    ap.add_argument("--audio-delay-floor-ms", type=float, default=80.0,
                    help="don't drop the audio delay below this even if video latency "
                         "drops; prevents the audio running ahead during brief stalls")
    args = ap.parse_args()
    args.eye_passthrough = not args.no_eye_passthrough

    if args.backend == "local":
        # swap_face checks DLC_REMOTE_SWAP_URL at call time; clear it so it falls
        # back to the in-process inswapper session (CoreML on Apple Silicon).
        os.environ.pop("DLC_REMOTE_SWAP_URL", None)
        os.environ.pop("DLC_REMOTE_SWAP_API_KEY", None)
    else:
        if not os.environ.get("DLC_REMOTE_SWAP_URL"):
            print("DLC_REMOTE_SWAP_URL not set — source env.remote first, "
                  "or pass --backend local.", file=sys.stderr)
            return 2
        # Picked up by modules.remote_swap_client to choose ws vs http.
        os.environ["DLC_REMOTE_SWAP_PROTOCOL"] = args.remote_protocol

    if args.workers is None:
        # Remote: scale wins because network RTT dominates; 3 workers ~3x throughput.
        # Local: CoreML serializes on the Neural Engine, so >1 worker triggers
        # painful first-frame compile contention with little steady-state win.
        args.workers = 3 if args.backend == "remote" else 1

    modules.globals.execution_providers = decode_execution_providers(default_execution_providers())
    # Tell DLC's update_status() to skip the tkinter UI hop (we're headless).
    modules.globals.headless = True
    print(f"[live] local face detector providers: {modules.globals.execution_providers}")

    # Build a FaceAnalysis at our requested det_size BEFORE DLC's get_face_analyser
    # gets a chance to lazy-init at 640x640. Once insightface's detection model has
    # an input_size, subsequent prepare() calls are silently ignored — which is why
    # earlier overrides showed "warning: det_size is already set, ignore". We then
    # stash the configured instance into DLC's module global so its get_one_face()
    # picks it up.
    import insightface  # noqa: E402
    import modules.face_analyser as _fa_mod  # noqa: E402
    from modules.processors.frame._onnx_enhancer import build_provider_config  # noqa: E402

    _providers = build_provider_config()
    _fa = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=_providers,
        allowed_modules=["detection", "recognition", "landmark_2d_106"],
    )
    _fa.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))
    _fa_mod.FACE_ANALYSER = _fa  # DLC's lazy init will see this and skip its own
    print(f"[live] detection input size: {args.det_size}x{args.det_size}")

    from modules.face_analyser import get_one_face
    from modules.processors.frame.face_swapper import swap_face

    # Resolve which local swap model to use (only applies to --backend local).
    local_swapper = None
    if args.backend == "local":
        models_dir = os.path.join(THIS_DIR, "models")
        hyperswap_path = os.path.join(models_dir, "hyperswap_1a_256.onnx")
        inswapper_path = os.path.join(models_dir, "inswapper_128.onnx")
        if args.swap_model_path:
            chosen = args.swap_model_path
        elif args.model == "hyperswap":
            chosen = hyperswap_path
        elif args.model == "inswapper":
            chosen = inswapper_path
        else:  # auto
            chosen = hyperswap_path if os.path.isfile(hyperswap_path) else inswapper_path
        if not os.path.isfile(chosen):
            print(f"[live] swap model not found: {chosen}", file=sys.stderr)
            return 2
        if "hyperswap" in os.path.basename(chosen).lower() or args.swap_model_path:
            # Use our own ORT-backed swapper for HyperSwap (and for explicit overrides),
            # since DLC's bundled face_swapper only knows about inswapper.
            # Use a *minimal* CoreML config — DLC's build_provider_config() enables
            # MLComputeUnits=ALL + AllowLowPrecisionAccumulationOnGPU which causes
            # NaN / black output on some HyperSwap ops. Plain CoreML + CPU fallback
            # matches what worked in standalone testing.
            ort_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            local_swapper = LocalSwapper(chosen, ort_providers)

    # Optional virtual camera (OBS Virtual Camera on macOS); silently no-ops if
    # pyvirtualcam isn't installed or OBS Virtual Camera isn't initialized.
    if args.no_virtual_cam:
        os.environ["DLC_VIRTUALCAM"] = "0"
    from modules.virtual_camera import send_frame as vcam_send, close as vcam_close, get_virtual_cam

    # Optional audio sync: capture mic, emit to a virtual audio device (BlackHole)
    # with delay equal to the live video latency. Off by default; user opts in.
    audio_delayer = None
    if args.audio_sync:
        try:
            from audio_sync import AudioSyncDelay, find_device_index
        except Exception as e:
            print(f"[audio_sync] sounddevice not installed: {e}", file=sys.stderr)
            print("[audio_sync] pip install sounddevice", file=sys.stderr)
        else:
            out_idx = find_device_index(args.audio_output, "output")
            in_idx = (find_device_index(args.audio_input, "input")
                      if args.audio_input else None)
            if out_idx is None:
                print(f"[audio_sync] no output device matching '{args.audio_output}'.",
                      file=sys.stderr)
                print("[audio_sync] install BlackHole with: brew install blackhole-2ch",
                      file=sys.stderr)
            elif args.audio_input and in_idx is None:
                print(f"[audio_sync] no input device matching '{args.audio_input}'.",
                      file=sys.stderr)
            else:
                # Pick a samplerate the input device actually supports — AirPods
                # mic is 24kHz, MacBook mic is 48kHz; BlackHole accepts both.
                import sounddevice as _sd
                in_info = _sd.query_devices(in_idx) if in_idx is not None else _sd.query_devices(kind="input")
                samplerate = int(in_info["default_samplerate"])
                initial = args.audio_delay_ms if args.audio_delay_ms is not None else 300.0
                in_label = in_info["name"] if isinstance(in_info, dict) else "default"
                audio_delayer = AudioSyncDelay(
                    input_device=in_idx, output_device=out_idx,
                    samplerate=samplerate, initial_delay_ms=initial,
                )
                if audio_delayer.start():
                    print(f"[audio_sync] mic '{in_label}' @ {samplerate}Hz -> "
                          f"'{args.audio_output}' delaying {initial:.0f}ms "
                          f"(auto-tracking video lat)")
                else:
                    print(f"[audio_sync] failed: {audio_delayer.last_error}", file=sys.stderr)
                    audio_delayer = None

    src_bgr = cv2.imread(args.source)
    if src_bgr is None:
        print(f"could not read source image: {args.source}", file=sys.stderr)
        return 2
    source_face = get_one_face(src_bgr)
    if source_face is None:
        print("no face detected in source image", file=sys.stderr)
        return 2
    print(f"[live] backend={args.backend}  workers={args.workers}  "
          f"eye_passthrough={args.eye_passthrough}  source face loaded")

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(
            f"could not open camera index {args.camera}.\n"
            "On macOS: System Settings → Privacy & Security → Camera, "
            "enable the Terminal/iTerm app you launched this from, then re-run.",
            file=sys.stderr,
        )
        return 2

    stop_event = threading.Event()
    # Decoupled pipeline:
    #   - Main thread: capture + composite + display at native camera rate (~30 fps).
    #   - Detector thread: pops capture frames at its own rate (~10-15 fps on CoreML),
    #     publishes face position via `shared`.
    #   - Swap workers: pop face+frame at their own rate (~4 fps with remote, ~6 fps
    #     local), publish the swap crop via `shared`.
    #   - Main composites every captured frame using the LATEST face position + the
    #     LATEST swap crop, so display always tracks current head motion even when
    #     the swap content is a few hundred ms stale. This sidesteps the proxy/GPU
    #     throughput ceiling: visual smoothness ≠ swap update rate.
    detect_q: queue.Queue = queue.Queue(maxsize=1)
    swap_q: queue.Queue = queue.Queue(maxsize=args.workers)

    # Threads update fields below; main composites against them every frame.
    from insightface.utils import face_align as _face_align_mod
    state_lock = threading.Lock()
    shared = {
        "face": None,         # most-recent Face from detector
        "face_t": 0.0,
        "crop": None,         # most-recent swap crop (HxWx3 uint8)
        "crop_t": 0.0,
        "input_size": (local_swapper.input_size if local_swapper is not None
                       else int(os.environ.get("DLC_REMOTE_SWAP_INPUT_SIZE",
                                               "128" if args.backend == "local" else "256"))),
    }

    def _put_drop_oldest(q: queue.Queue, item) -> None:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

    def detector_thread():
        n = 0
        while not stop_event.is_set():
            try:
                fid, frame, t_cap = detect_q.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.time()
            face = get_one_face(frame)
            t1 = time.time()
            with state_lock:
                shared["face"] = face
                shared["face_t"] = t1
            # Dispatch a swap job using this frame+face if there's room.
            if face is not None:
                _put_drop_oldest(swap_q, (fid, frame, face))
            n += 1
            if args.timing and (n <= 5 or n % 30 == 0):
                state = "face" if face is not None else "noface"
                print(
                    f"[det f{n:>3}] detect {(t1-t0)*1000:6.0f}ms  ({state})",
                    file=sys.stderr, flush=True,
                )

    def swap_worker(idx: int):
        n = 0
        while not stop_event.is_set():
            try:
                fid, frame, face = swap_q.get(timeout=0.1)
            except queue.Empty:
                continue
            t0 = time.time()
            try:
                if local_swapper is not None:
                    crop = local_swapper.compute_crop(source_face, face, frame)
                else:
                    # Remote backend: align crop ourselves, ship to pod, get crop back.
                    input_size = shared["input_size"]
                    aimg, _ = _face_align_mod.norm_crop2(frame, face.kps, input_size)
                    from modules.remote_swap_client import remote_swap_aligned
                    crop = remote_swap_aligned(aimg, source_face.normed_embedding)
            except Exception as e:
                print(f"[w{idx}] swap error: {e}", file=sys.stderr)
                crop = None
            t1 = time.time()
            if crop is not None:
                with state_lock:
                    shared["crop"] = crop
                    shared["crop_t"] = t1
            n += 1
            if args.timing and (n <= 5 or n % 30 == 0):
                print(
                    f"[w{idx} f{n:>3}] swap {(t1-t0)*1000:6.0f}ms",
                    file=sys.stderr, flush=True,
                )

    threads = [threading.Thread(target=detector_thread, daemon=True)]
    threads += [threading.Thread(target=swap_worker, args=(i,), daemon=True)
                for i in range(args.workers)]
    for t in threads:
        t.start()

    win = "DLC live (remote pod) — press q to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[live] press q to quit")

    next_fid = 0

    # Metrics windows
    win_t = time.time()
    cap_n = 0           # frames captured this window
    swap_seen_t = 0.0   # last shared['crop_t'] we observed -> swap rate
    swap_seen_n = 0
    last_swap_t_observed = 0.0
    cap_fps = swap_fps = 0.0
    avg_crop_age_ms = 0.0
    crop_ages = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)
            cap_n += 1

            t_cap = time.time()
            fid = next_fid
            next_fid += 1
            _put_drop_oldest(detect_q, (fid, frame, t_cap))

            # Snapshot shared state; composite on the *current* frame using the
            # latest detected face position + the latest swap crop.
            with state_lock:
                cur_face = shared["face"]
                cur_face_t = shared["face_t"]
                cur_crop = shared["crop"]
                cur_crop_t = shared["crop_t"]
                cur_input_size = shared["input_size"]

            if cur_face is not None and cur_crop is not None:
                try:
                    display = paste_swap_crop(cur_crop, cur_face, frame,
                                              cur_input_size, _face_align_mod)
                    if args.eye_passthrough:
                        display = passthrough_eyes(display, frame, cur_face,
                                                   expand=args.eye_expand)
                    crop_ages.append(time.time() - cur_crop_t)
                except Exception as e:
                    print(f"[composite] {e}", file=sys.stderr)
                    display = frame.copy()
            else:
                display = frame.copy()

            # Count unique swap completions for the HUD.
            if cur_crop_t != last_swap_t_observed:
                swap_seen_n += 1
                last_swap_t_observed = cur_crop_t

            # Publish to OBS Virtual Camera (Zoom/Meet pick this as their input).
            vcam_send(display)

            now = time.time()
            if now - win_t >= 1.0:
                cap_fps = cap_n / (now - win_t)
                swap_fps = swap_seen_n / (now - win_t)
                if crop_ages:
                    avg_crop_age_ms = 1000.0 * (sum(crop_ages) / len(crop_ages))
                win_t = now
                cap_n = swap_seen_n = 0
                crop_ages = []

                # Audio delayer tracks the swap-content age (since lip-sync depends
                # on the painted content, not just paint rate).
                if audio_delayer is not None:
                    if args.audio_delay_ms is not None:
                        target_ms = args.audio_delay_ms
                    else:
                        target_ms = max(args.audio_delay_floor_ms, avg_crop_age_ms)
                    audio_delayer.set_target_delay_ms(target_ms)

            hud_audio = ""
            if audio_delayer is not None:
                hud_audio = f"  audio:{audio_delayer.current_delay_ms:.0f}ms"
            hud = (f"{args.backend} | swap {swap_fps:4.1f} fps  cam {cap_fps:4.1f} fps  "
                   f"crop-age {avg_crop_age_ms:5.0f}ms  w{args.workers}  "
                   f"eyes:{'on' if args.eye_passthrough else 'off'}{hud_audio}")
            cv2.putText(display, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(win, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        try:
            vcam_close()
        except Exception:
            pass
        if audio_delayer is not None:
            audio_delayer.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
