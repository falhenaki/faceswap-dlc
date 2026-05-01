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

REMOTE = os.environ.get("ZIMAGE_SERVICE_URL", "").rstrip("/")
API_KEY = os.environ.get("ZIMAGE_API_KEY", "").strip()
BIND = os.environ.get("PLAYGROUND_HOST", "127.0.0.1")
PORT = int(os.environ.get("PLAYGROUND_PORT", "8765"))
# Generation can take many minutes over HTTPS to RunPod
REMOTE_TIMEOUT = int(os.environ.get("ZIMAGE_REMOTE_TIMEOUT", "900"))

# RunPod sits behind Cloudflare; Python-urllib's default User-Agent is often blocked
# ("The site owner has blocked access based on your browser's signature").
_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _browser_headers() -> dict[str, str]:
    ua = os.environ.get("ZIMAGE_BROWSER_UA", _DEFAULT_UA).strip() or _DEFAULT_UA
    h: dict[str, str] = {
        "User-Agent": ua,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": REMOTE,
        "Referer": REMOTE + "/",
    }
    return h


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
        headers = {**_browser_headers(), "Content-Type": "application/json"}
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
            raw_err = e.read() or b""
            out = raw_err or b'{"detail":"upstream error"}'
            code = e.code
            if raw_err.startswith(b"<") or b"blocked access" in raw_err.lower():
                msg = (
                    "RunPod/Cloudflare rejected the request (often bot protection). "
                    "The proxy now sends a browser User-Agent; if this persists, try "
                    "ZIMAGE_BROWSER_UA=... or use curl from the same machine to test."
                )
                out = json.dumps({"detail": msg, "upstream_snippet": raw_err[:500].decode("utf-8", "replace")}).encode(
                    "utf-8"
                )
                code = 502
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
    if not REMOTE:
        print(
            "Set ZIMAGE_SERVICE_URL to your RunPod base URL, e.g.\n"
            "  export ZIMAGE_SERVICE_URL=$(terraform -chdir=../terraform output -raw zimage_service_url)",
            file=sys.stderr,
        )
        sys.exit(1)
    if not REMOTE.startswith("https://"):
        print("ZIMAGE_SERVICE_URL should start with https://", file=sys.stderr)
        sys.exit(1)
    httpd = ThreadingHTTPServer((BIND, PORT), Handler)
    print(f"Playground http://{BIND}:{PORT}/  →  {REMOTE}/generate", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
