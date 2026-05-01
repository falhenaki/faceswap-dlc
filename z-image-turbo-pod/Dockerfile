# Pre-built image for RunPod (no git clone on the GPU).
# Build: docker buildx build --platform linux/amd64 -t ghcr.io/OWNER/faceswap-z-image-turbo:latest --push .
# CI: .github/workflows/z-image-ghcr.yml

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve.py .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
