#!/usr/bin/env python3
"""
Local playground: static UI + JSON proxy to RunPod Z-Image (avoids browser CORS).

  export ZIMAGE_SERVICE_URL=https://YOUR_POD-8000.proxy.runpod.net
  # optional if the pod has ZIMAGE_API_KEY set:
  # export ZIMAGE_API_KEY=...

  pip install -r requirements.txt   # curl_cffi: Chrome TLS for Cloudflare
  python3 server.py
  open http://127.0.0.1:8765/
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from curl_cffi import requests as curl_requests

    _CURL_CFFI = True
except ImportError:
    curl_requests = None  # type: ignore[misc, assignment]
    _CURL_CFFI = False

DIR = Path(__file__).resolve().parent
INDEX = (DIR / "index.html").read_text(encoding="utf-8")

REMOTE = os.environ.get("ZIMAGE_SERVICE_URL", "").rstrip("/")
API_KEY = os.environ.get("ZIMAGE_API_KEY", "").strip()
BIND = os.environ.get("PLAYGROUND_HOST", "127.0.0.1")
PORT = int(os.environ.get("PLAYGROUND_PORT", "8765"))
# Generation can take many minutes over HTTPS to RunPod
REMOTE_TIMEOUT = int(os.environ.get("ZIMAGE_REMOTE_TIMEOUT", "900"))

# Cloudflare 1010: blocks Python’s TLS fingerprint (JA3), not just User-Agent.
# curl_cffi impersonates Chrome TLS; install: pip install -r requirements.txt
_DEFAULT_IMPERSONATE = "chrome136"
_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _is_local_http(u: str) -> bool:
    """http://127.0.0.1:port — use after `ssh -L …` to the pod (bypasses Cloudflare)."""
    try:
        p = urlparse(u)
        return p.scheme == "http" and p.hostname in ("127.0.0.1", "localhost")
    except Exception:
        return False


def _upstream_headers() -> dict[str, str]:
    base = REMOTE
    h: dict[str, str] = {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": base,
        "Referer": base + "/",
        "User-Agent": os.environ.get("ZIMAGE_UPSTREAM_UA", _DEFAULT_UA).strip()
        or _DEFAULT_UA,
    }
    return h


def _post_system_curl(url: str, body: bytes, headers: dict[str, str]) -> tuple[bytes, int]:
    """Last-resort: system curl (different TLS stack than Python)."""
    curl_bin = shutil.which("curl")
    if not curl_bin:
        return (b'{"detail":"curl not found in PATH"}', 500)
    outp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    outp.close()
    try:
        cmd = [
            curl_bin,
            "-sS",
            "-L",
            "--max-time",
            str(REMOTE_TIMEOUT),
            "-X",
            "POST",
            url,
            "-o",
            outp.name,
            "-w",
            "%{http_code}",
        ]
        for k, v in headers.items():
            cmd.extend(["-H", f"{k}: {v}"])
        cmd.extend(["-H", "Content-Type: application/json", "--data-binary", "@-"])
        proc = subprocess.run(
            cmd,
            input=body,
            capture_output=True,
            timeout=REMOTE_TIMEOUT + 30,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or b"").decode("utf-8", "replace")[:800]
            return (json.dumps({"detail": f"curl failed: {err}"}).encode("utf-8"), 502)
        # With -o file, -w "%{http_code}" is written to stdout only.
        code_str = proc.stdout.decode("ascii", "ignore").strip()
        try:
            code = int(code_str)
        except ValueError:
            code = 502
        with open(outp.name, "rb") as f:
            out = f.read()
        return (out, code)
    finally:
        try:
            os.unlink(outp.name)
        except OSError:
            pass


def _post_upstream(url: str, raw: bytes, headers: dict[str, str]) -> tuple[bytes, int]:
    """POST to RunPod; retry with system curl if Cloudflare returns 1010."""
    if _CURL_CFFI and curl_requests is not None:
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
            impersonate = os.environ.get(
                "CURL_CFFI_IMPERSONATE", _DEFAULT_IMPERSONATE
            ).strip() or _DEFAULT_IMPERSONATE
            r = curl_requests.post(
                url,
                json=payload,
                headers=headers,
                impersonate=impersonate,
                timeout=REMOTE_TIMEOUT,
            )
            out, code = r.content, r.status_code
        except Exception as e:
            return (json.dumps({"detail": str(e)}).encode("utf-8"), 500)
    else:
        h = dict(headers)
        h["Content-Type"] = "application/json"
        try:
            req = Request(url, data=raw, method="POST", headers=h)
            with urlopen(req, timeout=REMOTE_TIMEOUT) as resp:
                out, code = resp.read(), resp.getcode()
        except HTTPError as e:
            out, code = e.read() or b'{"detail":"upstream error"}', e.code
        except URLError as e:
            return (json.dumps({"detail": str(e.reason)}).encode("utf-8"), 502)
        except Exception as e:
            return (json.dumps({"detail": str(e)}).encode("utf-8"), 500)

    cf_block = (
        code == 403
        and (
            b"1010" in out
            or b"cloudflare_error" in out.lower()
            or b"browser_signature" in out.lower()
        )
    )
    if cf_block and os.environ.get("ZIMAGE_TRY_SYSTEM_CURL", "1") != "0":
        out2, code2 = _post_system_curl(url, raw, headers)
        if code2 == 403 and b"1010" in out2:
            return (
                json.dumps(
                    {
                        "detail": (
                            "Cloudflare still blocks this path (1010). Bypass it: SSH port-forward "
                            "to the pod, then set ZIMAGE_SERVICE_URL=http://127.0.0.1:<port> "
                            "(no HTTPS proxy — see README)."
                        )
                    }
                ).encode("utf-8"),
                502,
            )
        return (out2, code2)
    return (out, code)


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
        url = f"{REMOTE}/generate"
        headers = dict(_upstream_headers())
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"

        out, code = _post_upstream(url, raw, headers)

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
    if not (REMOTE.startswith("https://") or _is_local_http(REMOTE)):
        print(
            "ZIMAGE_SERVICE_URL must be https://…proxy.runpod.net OR\n"
            "  http://127.0.0.1:<port> after SSH tunnel (bypasses Cloudflare 1010).\n"
            "  ssh -i KEY -N -L 18000:127.0.0.1:8000 <user>@ssh.runpod.io\n"
            "  export ZIMAGE_SERVICE_URL=http://127.0.0.1:18000",
            file=sys.stderr,
        )
        sys.exit(1)
    httpd = ThreadingHTTPServer((BIND, PORT), Handler)
    if _is_local_http(REMOTE):
        mode = "direct HTTP (SSH tunnel — no Cloudflare)"
    elif _CURL_CFFI:
        mode = f"curl_cffi impersonate={os.environ.get('CURL_CFFI_IMPERSONATE', _DEFAULT_IMPERSONATE)}"
    else:
        mode = "urllib (install curl-cffi)"
    print(
        f"Playground http://{BIND}:{PORT}/  →  {REMOTE}/generate  [{mode}]",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)


if __name__ == "__main__":
    main()
