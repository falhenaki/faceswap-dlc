# Z-Image-Turbo on GPU

HTTP service for [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) using `diffusers.ZImagePipeline`.

## API

- `GET /health` — `{"status":"ok",...}` when the model is loaded; `503` while loading.
- `POST /generate` — JSON body → PNG as base64 (see `serve.py`).

If `ZIMAGE_API_KEY` is set on the pod, send `Authorization: Bearer <key>` on `/generate`.

## Docker image (what RunPod runs)

The GPU pod uses a **pre-built image** — no `git clone` on the machine.

1. **Build and push** the **linux/amd64** image (RunPod GPUs are x86):
   ```bash
   docker login ghcr.io -u YOUR_GH_USER
   cd z-image-turbo-pod
   docker buildx build --platform linux/amd64 \
     -t ghcr.io/YOUR_GH_USER/faceswap-z-image-turbo:latest --push .
   ```
   Match `YOUR_GH_USER` to `container_image` in Terraform (default expects `ghcr.io/falhenaki/faceswap-z-image-turbo:latest`).

   **GitHub Actions:** add a workflow under `.github/workflows/` that runs `docker/build-push-action` with `packages: write` (your `gh` token needs the **`workflow`** scope to push workflow files from the CLI).

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
