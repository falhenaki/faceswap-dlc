#!/usr/bin/env python3
"""
Local playground: static UI + JSON proxy to RunPod Z-Image (avoids browser CORS).

  export ZIMAGE_SERVICE_URL=https://YOUR_POD-8000.proxy.runpod.net
  # optional if the pod has ZIMAGE_API_KEY set:
  # export ZIMAGE_API_KEY=...

  python3 server.py
  open http://127.0.0.1:8765/
"""

from __future__ import annotations

import json
import os
import sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DIR = Path(__file__).resolve().parent
INDEX = (DIR / "index.html").read_text(encoding="utf-8")

REMOTE = os.environ.get(
    "ZIMAGE_SERVICE_URL", "https://758kfzcy39xrfd-8000.proxy.runpod.net"
).rstrip("/")
API_KEY = os.environ.get("ZIMAGE_API_KEY", "").strip()
BIND = os.environ.get("PLAYGROUND_HOST", "127.0.0.1")
PORT = int(os.environ.get("PLAYGROUND_PORT", "8765"))
# Generation can take many minutes over HTTPS to RunPod
REMOTE_TIMEOUT = int(os.environ.get("ZIMAGE_REMOTE_TIMEOUT", "900"))


def _remote_host() -> str:
    try:
        from urllib.parse import urlparse

        return urlparse(REMOTE).netloc or REMOTE
    except Exception:
        return REMOTE


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = INDEX.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/config":
            out = json.dumps({"remote_host": _remote_host()}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
        req = Request(
            f"{REMOTE}/generate",
            data=raw,
            method="POST",
            headers=headers,
        )
        try:
            with urlopen(req, timeout=REMOTE_TIMEOUT) as resp:
                out = resp.read()
                code = resp.getcode()
        except HTTPError as e:
            out = e.read() or b'{"detail":"upstream error"}'
            code = e.code
        except URLError as e:
            out = json.dumps({"detail": str(e.reason)}).encode("utf-8")
            code = 502
        except Exception as e:
            out = json.dumps({"detail": str(e)}).encode("utf-8")
            code = 500

        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main() -> None:
    if not REMOTE.startswith("https://"):
        print("Set ZIMAGE_SERVICE_URL to your RunPod https://…-8000.proxy.runpod.net", file=sys.stderr)
    httpd = ThreadingHTTPServer((BIND, PORT), Handler)
    print(f"Playground http://{BIND}:{PORT}/  →  {REMOTE}/generate", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
