#!/usr/bin/env bash
# Anonymous push to ttl.sh (no GHCR token). Image expires per tag (max ~24h on ttl.sh).
# Usage: ./scripts/push-ttl.sh
# Then: terraform apply -var="container_image=$(cat pushed-image.var)"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
UUID=$(uuidgen | tr '[:upper:]' '[:lower:]')
TAG="ttl.sh/${UUID}:1d"
docker buildx create --use 2>/dev/null || true
docker buildx build --platform linux/amd64 -t "$TAG" --push .
echo "$TAG" > terraform/pushed-image.var
echo "Pushed $TAG (save this string — image expires with ttl.sh policy)."
echo "terraform apply -var=container_image=$TAG"
