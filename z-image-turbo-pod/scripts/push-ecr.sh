#!/usr/bin/env bash
# Build linux/amd64 and push to Amazon ECR.
# ECR stores image data in AWS (S3-backed); RunPod pulls with a normal registry URL — not s3://.
#
# Prereqs: aws CLI configured, Docker buildx, permission ecr:* / sts:GetCallerIdentity.
#
# Usage:
#   export AWS_REGION=us-east-1
#   export ECR_REPOSITORY=faceswap-z-image-turbo   # optional
#   ./scripts/push-ecr.sh
#
# Then set Terraform:
#   terraform apply -var="container_image=<printed URI>"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${AWS_REGION:?Set AWS_REGION (e.g. us-east-1)}"
REPO="${ECR_REPOSITORY:-faceswap-z-image-turbo}"
TAG="${ECR_IMAGE_TAG:-latest}"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
URI="${REGISTRY}/${REPO}:${TAG}"

if ! aws ecr describe-repositories --repository-names "$REPO" --region "$AWS_REGION" &>/dev/null; then
  aws ecr create-repository --repository-name "$REPO" --region "$AWS_REGION" --image-scanning-configuration scanOnPush=true
fi

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$REGISTRY"

docker buildx create --use 2>/dev/null || true
docker buildx build --platform linux/amd64 \
  -f "$ROOT/Dockerfile" \
  -t "$URI" \
  --push \
  "$ROOT"

echo "Pushed $URI"
echo "terraform -chdir=terraform apply -var=container_image=$URI"
echo "If the repo is private, configure RunPod to pull from ECR (registry credentials) or use ecr-public."
