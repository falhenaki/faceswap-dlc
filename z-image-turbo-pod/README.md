# Z-Image-Turbo on GPU

HTTP service for [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) using `diffusers.ZImagePipeline`.

## API

- `GET /health` — `{"status":"ok",...}` when the model is loaded; `503` while loading.
- `POST /generate` — JSON body → PNG as base64 (see `serve.py`).

If `ZIMAGE_API_KEY` is set on the pod, send `Authorization: Bearer <key>` on `/generate`.

## RunPod (Terraform)

1. **Push `z-image-turbo-pod/` to the branch RunPod clones** (usually `main` on
   `zimage_code_git_url`). The pod runs `git clone --depth 1` on every start; if
   that revision does not contain this directory, the container will exit.

2. Copy variables and fill in your git URL:

   ```bash
   cd z-image-turbo-pod/terraform
   cp terraform.tfvars.example terraform.tfvars
   # edit: zimage_code_git_url, optional ssh_public_key, zimage_api_key, hf_token
   ```

3. Export your RunPod API key and apply:

   ```bash
   export RUNPOD_API_KEY=...
   terraform init
   terraform apply
   ```

4. Wait for first boot: weights download to `/workspace/hf_cache` (persists on the attached volume). Until then, `/health` returns `503`. Typical total time is **15–40+ minutes** depending on GPU datacenter bandwidth.

5. Test:

   ```bash
   terraform output -raw zimage_service_url
   curl -sS "$(terraform output -raw zimage_service_url)/health"
   ```

## Docker / Kubernetes

See `Dockerfile` and `k8s/pod-and-service.yaml`.

## Environment (pod)

| Variable | Meaning |
| --- | --- |
| `ZIMAGE_MODEL_ID` | Default `Tongyi-MAI/Z-Image-Turbo` |
| `TORCH_DTYPE` | `bfloat16` (default), `float16`, or `float32` |
| `ENABLE_MODEL_CPU_OFFLOAD` | `true` to save VRAM (slower) |
| `ATTENTION_BACKEND` | Optional `flash` / `_flash_3` if supported |
| `HF_TOKEN` | Optional Hugging Face token |
| `ZIMAGE_API_KEY` | Optional bearer secret for `POST /generate` |

## VRAM

Official guidance is **~16 GB** for comfortable full-GPU inference. Terraform defaults prefer **RTX A4000 / 3090 / 4080 SUPER** class GPUs. On **12 GB**, set `enable_model_cpu_offload = true` in `terraform.tfvars`.
