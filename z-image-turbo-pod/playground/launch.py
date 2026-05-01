#!/usr/bin/env python3
"""
One-command Z-Image playground: SSH tunnel (bypasses Cloudflare 1010) + local UI.

Prereqs (once):
  - faceswap/Deep-Live-Cam/env.remote with RUNPOD_API_KEY
  - Private key for RunPod SSH: default faceswap/ssh_id_runpod, or RUNPOD_SSH_KEY / --ssh-key
    (same file as -i in RunPod → Connect → SSH; their example path may not match yours)
  - terraform state for z-image pod, OR set ZIMAGE_POD_ID

  pip install -r requirements.txt

Usage:
  python3 launch.py
  python3 launch.py --no-browser
  ZIMAGE_SERVICE_URL=https://... python3 launch.py --skip-tunnel   # HTTPS (may hit CF)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

PLAYGROUND = Path(__file__).resolve().parent
REPO = PLAYGROUND.parent.parent  # faceswap/
DLC_ENV = REPO / "Deep-Live-Cam" / "env.remote"
TF_DIR = PLAYGROUND.parent / "terraform"
TF_BIN = REPO / ".tools" / "terraform"
SERVER = PLAYGROUND / "server.py"
REQ = PLAYGROUND / "requirements.txt"
DEFAULT_SSH_KEY = REPO / "ssh_id_runpod"
LOCAL_TUNNEL_PORT = int(os.environ.get("ZIMAGE_TUNNEL_LOCAL_PORT", "18000"))
RUNPOD_API = "https://rest.runpod.io/v1"


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


def _terraform_output(name: str) -> str | None:
    exe = TF_BIN if TF_BIN.is_file() else Path(shutil.which("terraform") or "")
    if not exe or not TF_DIR.is_dir():
        return None
    try:
        out = subprocess.run(
            [str(exe), f"-chdir={TF_DIR}", "output", "-raw", name],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _api_get(path: str, api_key: str) -> dict:
    req = urllib.request.Request(
        f"{RUNPOD_API}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _wait_port(host: str, port: int, timeout: float = 45.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((host, port))
            s.close()
            return True
        except OSError:
            time.sleep(0.3)
    return False


def _pip_install() -> None:
    if not REQ.is_file():
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(REQ)],
        check=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Z-Image playground with auto SSH tunnel")
    ap.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    ap.add_argument(
        "--skip-tunnel",
        action="store_true",
        help="Use ZIMAGE_SERVICE_URL as-is (HTTPS proxy; Cloudflare may block)",
    )
    ap.add_argument("--pod-id", default=os.environ.get("ZIMAGE_POD_ID", ""), help="RunPod pod id")
    ap.add_argument(
        "--ssh-key",
        default=os.environ.get("RUNPOD_SSH_KEY", str(DEFAULT_SSH_KEY)),
        help="Path to SSH private key",
    )
    ap.add_argument(
        "--ssh-user",
        default=os.environ.get("ZIMAGE_SSH_USER", "root"),
        help="SSH user for direct IP login (RunPod templates usually root)",
    )
    ap.add_argument("--local-port", type=int, default=LOCAL_TUNNEL_PORT, help="Local forward port")
    args = ap.parse_args()

    _load_env_file(DLC_ENV)
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print(
            f"Missing RUNPOD_API_KEY. Add it to {DLC_ENV} or export it.",
            file=sys.stderr,
        )
        sys.exit(1)

    _pip_install()

    tunnel_proc: subprocess.Popen | None = None

    if args.skip_tunnel:
        if not os.environ.get("ZIMAGE_SERVICE_URL"):
            print("With --skip-tunnel, set ZIMAGE_SERVICE_URL.", file=sys.stderr)
            sys.exit(1)
    else:
        pod_id = args.pod_id.strip() or _terraform_output("pod_id") or ""
        if not pod_id:
            print(
                "Set ZIMAGE_POD_ID or run terraform apply so pod_id is in outputs.",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            pod = _api_get(f"/pods/{pod_id}", api_key)
        except urllib.error.HTTPError as e:
            print(f"RunPod API: {e.code} {e.reason}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"RunPod API error: {e}", file=sys.stderr)
            sys.exit(1)

        public_ip = (pod.get("publicIp") or "").strip()
        mappings = pod.get("portMappings") or {}
        ssh_port = mappings.get("22") or mappings.get(22)

        if not public_ip or not ssh_port:
            print(
                "Pod has no publicIp / SSH port mapping yet. Start the pod or use Community Cloud with public IP.\n"
                "Fallback: set ZIMAGE_SERVICE_URL to the https proxy URL and run:\n"
                "  python3 launch.py --skip-tunnel",
                file=sys.stderr,
            )
            sys.exit(1)

        key_path = Path(args.ssh_key).expanduser()
        if not key_path.is_file():
            print(f"SSH key not found: {key_path}", file=sys.stderr)
            sys.exit(1)

        ssh_bin = shutil.which("ssh")
        if not ssh_bin:
            print("ssh not found in PATH.", file=sys.stderr)
            sys.exit(1)

        local_port = args.local_port
        target = f"{args.ssh_user}@{public_ip}"
        ssh_cmd = [
            ssh_bin,
            "-N",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ServerAliveInterval=30",
            "-i",
            str(key_path),
            "-L",
            f"127.0.0.1:{local_port}:127.0.0.1:8000",
            "-p",
            str(ssh_port),
            target,
        ]

        print(
            f"SSH tunnel 127.0.0.1:{local_port} → pod :8000 via {target}:{ssh_port}",
            flush=True,
        )
        tunnel_proc = subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        if not _wait_port("127.0.0.1", local_port, timeout=60.0):
            err = (
                tunnel_proc.stderr.read().decode("utf-8", "replace")
                if tunnel_proc.stderr
                else ""
            )
            tunnel_proc.terminate()
            print(
                "Tunnel failed to open local port. Check SSH key and RunPod Connect instructions.\n"
                f"ssh stderr: {err[:500]}",
                file=sys.stderr,
            )
            if "Connection refused" in err:
                print(
                    "Hint: runpod often returns connection refused when the pod is stopped "
                    "or still starting—start it in the dashboard and retry.",
                    file=sys.stderr,
                )
            sys.exit(1)

        os.environ["ZIMAGE_SERVICE_URL"] = f"http://127.0.0.1:{local_port}"

    ui_port = int(os.environ.get("PLAYGROUND_PORT", "8765"))
    if not args.no_browser:

        def _open() -> None:
            time.sleep(1.2)
            import webbrowser

            webbrowser.open(f"http://127.0.0.1:{ui_port}/")

        threading.Thread(target=_open, daemon=True).start()

    os.chdir(PLAYGROUND)
    server_proc = subprocess.Popen([sys.executable, str(SERVER)])

    if tunnel_proc is not None:

        def _handle_signal(sig: int, _frame: object) -> None:
            try:
                server_proc.terminate()
            except Exception:
                pass
            try:
                if tunnel_proc is not None:
                    tunnel_proc.terminate()
                    tunnel_proc.wait(timeout=5)
            except Exception:
                pass
            sys.exit(128 + sig)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    try:
        server_proc.wait()
    except KeyboardInterrupt:
        server_proc.terminate()
    finally:
        if tunnel_proc is not None:
            try:
                tunnel_proc.terminate()
                tunnel_proc.wait(timeout=5)
            except Exception:
                pass


if __name__ == "__main__":
    main()
