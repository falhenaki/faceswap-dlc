variable "pod_name" {
  type        = string
  description = "RunPod pod display name"
  default     = "z-image-turbo"
}

variable "container_image" {
  type        = string
  description = "PyTorch + CUDA 12 runtime matches Dockerfile; required for ZImagePipeline + recent diffusers."
  default     = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
}

variable "cloud_type" {
  type        = string
  description = "SECURE or COMMUNITY"
  default     = "COMMUNITY"
}

variable "gpu_type_ids" {
  type        = list(string)
  description = "Z-Image-Turbo is ~6B; 16 GB VRAM is comfortable. 12 GB may need ENABLE_MODEL_CPU_OFFLOAD."
  default = [
    "NVIDIA RTX A4000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A4500",
    "NVIDIA GeForce RTX 4080 SUPER",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4070 Ti",
    "NVIDIA GeForce RTX 3080 Ti",
    "NVIDIA GeForce RTX 3080",
  ]
}

variable "data_center_ids" {
  type        = list(string)
  description = "NYC-adjacent first; widen if capacity errors."
  default = [
    "US-MD-1",
    "CA-MTL-1",
    "CA-MTL-3",
    "CA-MTL-4",
    "US-NC-1",
    "US-NC-2",
    "US-IL-1",
    "US-GA-2",
    "US-KS-2",
    "US-TX-3",
    "US-TX-4",
    "US-CA-2",
    "US-WA-1",
  ]
}

variable "container_disk_in_gb" {
  type        = number
  description = "Ephemeral disk for pip wheels and transient unpack"
  default     = 100
}

variable "pod_volume_in_gb" {
  type        = number
  description = "HF hub cache on /workspace/hf_cache (persists across stop/start)"
  default     = 60
}

variable "zimage_code_git_url" {
  type        = string
  description = "Git URL RunPod clones on boot (must contain zimage_code_repo_subpath)"
}

variable "zimage_code_repo_subpath" {
  type        = string
  description = "Path from repo root to z-image-turbo-pod directory"
  default     = "z-image-turbo-pod"
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
  description = "Set true for tight VRAM (slower)"
  default     = false
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
