#!/usr/bin/env python3
"""
Call the remote Z-Image services POST /generate and save a PNG (no browser UI).

Needs ZIMAGE_SERVICE_URL — tunnel (http://127.0.0.1:…) or RunPod HTTPS proxy.

For https://*.proxy.runpod.net, POST is often blocked for plain urllib (Cloudflare 1010).
This script uses curl_cffi (Chrome TLS) when available — same as playground/requirements.txt.

Loads faceswap/Deep-Live-Cam/env.remote into the environment when keys are unset.
If the pod sets ZIMAGE_API_KEY, also export ZIMAGE_API_KEY (or pass via env).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parent.parent.parent  # faceswap/
DLC_ENV = REPO / "Deep-Live-Cam" / "env.remote"


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _via_urllib(url: str, body: bytes, headers: dict[str, str]) -> str:
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=1200) as r:
        return r.read().decode("utf-8")


def _via_curl_cffi(url: str, payload: dict, headers: dict[str, str]) -> str:
    from curl_cffi import requests as cr  # type: ignore[import-untyped]

    impersonate = os.environ.get("CURL_CFFI_IMPERSONATE", "chrome136").strip() or "chrome136"
    r = cr.post(url, json=payload, headers=headers or None, impersonate=impersonate, timeout=1200)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:800]}")
    return r.text


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate one image via Z-Image POST /generate")
    ap.add_argument(
        "--prompt",
        default="a photorealistic dog outdoors, detailed fur, natural lighting",
        help="Text prompt",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="dog.png",
        help="Output PNG path",
    )
    ap.add_argument(
        "--url",
        default="",
        help="Override ZIMAGE_SERVICE_URL",
    )
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    _load_env_file(DLC_ENV)
    base = (args.url or os.environ.get("ZIMAGE_SERVICE_URL", "")).rstrip("/")
    if not base:
        print(
            "Set ZIMAGE_SERVICE_URL to your pod base URL.\n"
            "Example: export ZIMAGE_SERVICE_URL=https://<pod>-8000.proxy.runpod.net\n"
            "Or after tunnel: export ZIMAGE_SERVICE_URL=http://127.0.0.1:18000",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = os.environ.get("ZIMAGE_API_KEY", "").strip()
    payload: dict = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "guidance_scale": 0.0,
    }
    if args.seed is not None:
        payload["seed"] = args.seed

    url = f"{base}/generate"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    scheme = urlparse(base).scheme.lower()
    raw: str | None = None

    if scheme == "https":
        try:
            raw = _via_curl_cffi(url, payload, {k: v for k, v in headers.items() if v})
        except ImportError:
            print(
                "HTTPS Z-Image needs curl_cffi (Cloudflare 1010 on plain Python TLS).\n"
                "  pip install curl_cffi\n"
                "Falling back to urllib — may fail with 403.",
                file=sys.stderr,
            )
        except Exception as e:  # retry urllib
            print(f"curl_cffi request failed: {e}; trying urllib…", file=sys.stderr)
            raw = None

    if raw is None:
        body = json.dumps(payload).encode("utf-8")
        try:
            raw = _via_urllib(url, body, headers)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:800]
            if e.code == 403 and "1010" in detail and scheme == "https":
                print(
                    f"HTTP {e.code} (Cloudflare).\n"
                    "Install: pip install curl_cffi\n"
                    f"Detail: {detail[:400]}",
                    file=sys.stderr,
                )
            else:
                print(f"HTTP {e.code}: {detail}", file=sys.stderr)
            sys.exit(1)
        except urllib.error.URLError as e:
            print(f"Request failed: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        data = json.loads(raw)
        b64 = data["image_base64"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Bad response: {e}\n{raw[:500]}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output).expanduser()
    out_path.write_bytes(base64.standard_b64decode(b64))
    print(f"Wrote {out_path.resolve()} ({len(payload['prompt'])} char prompt)", flush=True)


if __name__ == "__main__":
    main()
