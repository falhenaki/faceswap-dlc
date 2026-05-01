#!/usr/bin/env bash
# Run inside the RunPod container as docker_start_cmd (see main.tf).
set -euo pipefail
if ! command -v git >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq git curl ca-certificates
fi
cd /workspace
: "${ZIMAGE_REPO_URL:?ZIMAGE_REPO_URL must be set (git clone URL for this repo)}"
SUBPATH="${ZIMAGE_REPO_SUBPATH:-z-image-turbo-pod}"
rm -rf zimg-src
git clone --depth 1 "$ZIMAGE_REPO_URL" zimg-src
cd "zimg-src/${SUBPATH}"
exec bash runpod_bootstrap.sh
