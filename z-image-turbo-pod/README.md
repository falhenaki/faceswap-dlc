# Z-Image-Turbo on GPU

HTTP service for [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) using `diffusers.ZImagePipeline`.

## API

- `GET /health` — `{"status":"ok",...}` when the model is loaded; `503` while loading.
- `POST /generate` — JSON body → PNG as base64 (see `serve.py`).

If `ZIMAGE_API_KEY` is set on the pod, send `Authorization: Bearer <key>` on `/generate`.

## Docker image (what RunPod runs)

The GPU pod uses a **pre-built image** — no `git clone` on the machine.

1. **Build and push** the **linux/amd64** image (RunPod GPUs are x86):

   **GHCR (persistent)** — needs a token with **`write:packages`** (and often `read:packages`):
   ```bash
   echo "$(gh auth token)" | docker login ghcr.io -u YOUR_GH_USER --password-stdin
   cd z-image-turbo-pod
   GHCR_USER=YOUR_GH_USER ./scripts/build-and-push-ghcr.sh
   ```
   If `gh auth token` push is **denied: wrong scopes**, run  
   `gh auth refresh -h github.com -s write:packages -s read:packages`  
   and complete the browser/device flow, then push again.

   **ttl.sh (no login, ephemeral)** — if GHCR is blocked, anonymous registry (image expires per tag, often **≤24h**):
   ```bash
   cd z-image-turbo-pod && ./scripts/push-ttl.sh
   terraform -chdir=terraform apply -var="container_image=$(cat terraform/pushed-image.var)"
   ```

   **AWS (S3-backed via ECR)** — RunPod’s `image_name` must be a **registry** URL (`docker pull`), not `s3://…`. **Amazon ECR** stores layers on AWS (S3-backed) and exposes the normal pull API:
   ```bash
   export AWS_REGION=us-east-1   # pick a region near you / the pod
   cd z-image-turbo-pod && ./scripts/push-ecr.sh
   ```
   Use the printed `…dkr.ecr…amazonaws.com/…:tag` as `container_image`. Private ECR needs RunPod registry credentials; **public ECR** (`public.ecr.aws/...`) avoids that for pulls.

   **Raw tar on S3 (not for RunPod `image_name`)** — to stash a tarball in a bucket (backup), use `./scripts/archive-tar-to-s3.sh` after setting `S3_URI`. Someone with Docker can `docker load` after download; for deployment, prefer **ECR** so RunPod pulls OCI directly.

   Default `container_image` in Terraform remains **`ghcr.io/falhenaki/faceswap-z-image-turbo:latest`** if you use GHCR; override with `-var` or `terraform.tfvars` for ttl / ECR / etc.

2. **GHCR visibility** — For RunPod to pull without registry credentials, set the package to **public** (GitHub → Packages → package → Package settings → Change visibility). Private images need RunPod pull secrets (not wired in this Terraform).

3. **Terraform** — `container_image` defaults to `ghcr.io/falhenaki/faceswap-z-image-turbo:latest`. Override in `terraform.tfvars` if you publish under another user/org.

```bash
cd z-image-turbo-pod/terraform
export RUNPOD_API_KEY=...
# optional: export TF_VAR_ssh_public_key="$(cat ~/.ssh/id_ed25519.pub)"
terraform init
terraform apply
```

4. **First boot** — Weights download to `/workspace/hf_cache` (attached volume). Until then `/health` returns `503`. Often **15–40+ minutes**.

5. **Spot** — `interruptible = true` by default (easier scheduling; can be **outbid**). Resume: `scripts/pod start`.

6. **Lifecycle** — `scripts/pod start|stop|status|health|url|destroy` (reads `RUNPOD_API_KEY` from `../Deep-Live-Cam/env.remote` if unset).

## Docker / Kubernetes (cluster)

See `Dockerfile` and `k8s/pod-and-service.yaml` (set image to your GHCR tag).

## Environment (pod)

| Variable | Meaning |
| --- | --- |
| `PORT` | HTTP port (Terraform sets to `8000`) |
| `HF_HOME` / `TRANSFORMERS_CACHE` | Default `/workspace/hf_cache` |
| `ZIMAGE_MODEL_ID` | Default `Tongyi-MAI/Z-Image-Turbo` |
| `TORCH_DTYPE` | `bfloat16`, `float16`, or `float32` |
| `ENABLE_MODEL_CPU_OFFLOAD` | `true` to save VRAM |
| `ATTENTION_BACKEND` | Optional `flash` / `_flash_3` |
| `HF_TOKEN` | Optional |
| `ZIMAGE_API_KEY` | Optional Bearer for `/generate` |

## VRAM

Official guidance is **~16 GB** for comfortable full-GPU inference. Terraform defaults prefer a wide GPU list; use `enable_model_cpu_offload = true` on **8–12 GB** cards.
