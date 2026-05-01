#!/usr/bin/env bash
# Build linux/amd64 image and push to GHCR (requires: docker buildx, docker login ghcr.io).
# Usage: GHCR_USER=falhenaki ./scripts/build-and-push-ghcr.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
USER="${GHCR_USER:-${GITHUB_REPOSITORY_OWNER:-}}"
if [[ -z "$USER" ]]; then
  echo "Set GHCR_USER (GitHub username for ghcr.io/USER/...)" >&2
  exit 1
fi
TAG="${GHCR_TAG:-ghcr.io/${USER}/faceswap-z-image-turbo:latest}"
docker buildx create --use 2>/dev/null || true
docker buildx build --platform linux/amd64 \
  -f "$ROOT/Dockerfile" \
  -t "$TAG" \
  --push \
  "$ROOT"
echo "Pushed $TAG — set Terraform container_image to this tag if different from default."
