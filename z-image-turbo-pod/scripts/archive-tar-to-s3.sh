#!/usr/bin/env bash
# Optional: docker save → S3 object (backup / migration). RunPod still needs a registry URL (use push-ecr.sh).
#
# Usage:
#   export S3_URI=s3://my-bucket/path/faceswap-z-image-amd64.tar
#   ./scripts/archive-tar-to-s3.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${S3_URI:?Set S3_URI e.g. s3://bucket/prefix/image.tar}"

docker buildx create --use 2>/dev/null || true
docker buildx build --platform linux/amd64 -f "$ROOT/Dockerfile" -t faceswap-z-image:archive --load "$ROOT"
docker save faceswap-z-image:archive | aws s3 cp - "$S3_URI"
echo "Uploaded $S3_URI — import elsewhere with: aws s3 cp $S3_URI - | docker load"
