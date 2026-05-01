variable "pod_name" {
  type        = string
  description = "RunPod pod display name"
  default     = "z-image-turbo"
}

variable "container_image" {
  type        = string
  description = "Pre-built image (push via .github/workflows/z-image-ghcr.yml). Must be linux/amd64 + CUDA for RunPod GPU."
  default     = "ghcr.io/falhenaki/faceswap-z-image-turbo:latest"
}

variable "cloud_type" {
  type        = string
  description = "SECURE or COMMUNITY"
  default     = "COMMUNITY"
}

variable "gpu_type_ids" {
  type        = list(string)
  description = "Wide list for Community Cloud scheduling; 8–12 GB may need enable_model_cpu_offload=true."
  default = [
    "NVIDIA GeForce RTX 3070",
    "NVIDIA GeForce RTX 3080",
    "NVIDIA GeForce RTX 3080 Ti",
    "NVIDIA GeForce RTX 4070 Ti",
    "NVIDIA RTX A4000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A4500",
    "NVIDIA GeForce RTX 4080 SUPER",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA L4",
    "Tesla V100-PCIE-16GB",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA GeForce RTX 4080",
    "Tesla T4",
    "NVIDIA RTX A2000",
    "NVIDIA GeForce RTX 5080",
  ]
}

variable "data_center_ids" {
  type        = list(string)
  description = "RunPod REST API enum only; short list first, expand via tfvars on 500 capacity errors."
  default = [
    "US-CA-2",
    "US-TX-3",
    "US-TX-4",
    "US-KS-2",
    "US-IL-1",
    "US-NC-1",
    "US-DE-1",
    "US-GA-2",
    "CA-MTL-1",
    "CA-MTL-3",
    "EU-CZ-1",
    "EU-NL-1",
    "EU-RO-1",
  ]
}

variable "container_disk_in_gb" {
  type        = number
  description = "Ephemeral disk; large values can block Community scheduling"
  default     = 50
}

variable "pod_volume_in_gb" {
  type        = number
  description = "HF hub cache on /workspace/hf_cache (persists across stop/start)"
  default     = 50
}

variable "interruptible" {
  type        = bool
  description = "Spot (true) schedules when on-demand returns 500; can be outbid — use `scripts/pod start` to resume."
  default     = true
}

variable "zimage_model_id" {
  type        = string
  description = "Hugging Face model id"
  default     = "Tongyi-MAI/Z-Image-Turbo"
}

variable "torch_dtype" {
  type        = string
  description = "bfloat16 (Ampere+) or float16"
  default     = "bfloat16"
}

variable "hf_token" {
  type        = string
  sensitive   = true
  description = "Optional Hugging Face token if you use gated assets"
  default     = ""
}

variable "zimage_api_key" {
  type        = string
  sensitive   = true
  description = "Optional Bearer secret for POST /generate (leave empty to disable auth)"
  default     = ""
}

variable "enable_model_cpu_offload" {
  type        = bool
  description = "Set true for 8–12 GB VRAM (slower but schedules on more GPUs)"
  default     = true
}

variable "attention_backend" {
  type        = string
  description = "Optional: flash | _flash_3 if your stack supports it (see diffusers docs)"
  default     = ""
}

variable "ssh_public_key" {
  type        = string
  description = "ssh-ed25519 line for RunPod root SSH on TCP 22"
  default     = ""
}
