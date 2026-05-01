#!/usr/bin/env python3
"""
Call the remote Z-Image service POST /generate and save a PNG (no browser UI).

Needs ZIMAGE_SERVICE_URL — same as the playground, e.g. after SSH tunnel:
  export ZIMAGE_SERVICE_URL=http://127.0.0.1:18000

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
            "Example after tunnel: export ZIMAGE_SERVICE_URL=http://127.0.0.1:18000",
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

    body = json.dumps(payload).encode("utf-8")
    url = f"{base}/generate"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:800]
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
