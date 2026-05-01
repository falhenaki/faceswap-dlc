#!/usr/bin/env bash
# Build linux/amd64 and push to Amazon ECR **Public** (anonymous docker pull — good for RunPod).
# Creates the repo if missing. Repository URI is printed at the end.
#
# Usage:
#   export AWS_REGION=us-east-1   # ECR Public API is us-east-1
#   export ECR_REPOSITORY=faceswap-z-image-turbo
#   ./scripts/push-ecr-public.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${AWS_REGION:=us-east-1}"
REPO="${ECR_REPOSITORY:-faceswap-z-image-turbo}"
TAG="${ECR_IMAGE_TAG:-latest}"

if ! aws ecr-public describe-repositories --repository-names "$REPO" --region "$AWS_REGION" &>/dev/null; then
  aws ecr-public create-repository --repository-name "$REPO" --region "$AWS_REGION"
fi

URI_BASE="$(aws ecr-public describe-repositories --repository-names "$REPO" --region "$AWS_REGION" --query 'repositories[0].repositoryUri' --output text)"
URI="${URI_BASE}:${TAG}"

aws ecr-public get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin public.ecr.aws

docker buildx create --use 2>/dev/null || true
docker buildx build --platform linux/amd64 -f "$ROOT/Dockerfile" -t "$URI" --push "$ROOT"

echo "Pushed $URI"
echo "terraform apply -var=container_image=$URI"
