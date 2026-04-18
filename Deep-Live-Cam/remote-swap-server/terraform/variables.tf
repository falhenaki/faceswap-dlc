variable "pod_name" {
  type        = string
  description = "RunPod pod display name"
  default     = "dlc-remote-swap"
}

variable "container_image" {
  type        = string
  description = "Smaller devel image schedules reliably on Community Cloud; override to CUDA 12 if you need ORT 1.19+."
  default     = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
}

variable "cloud_type" {
  type        = string
  description = "SECURE or COMMUNITY"
  default     = "COMMUNITY"
}

variable "gpu_type_ids" {
  type        = list(string)
  description = <<-EOT
    Ordered GPU preferences; cheapest-first for single-user batch workloads.
    inswapper_128 is tiny (<1GB VRAM); even an RTX 3070 saturates at >100 FPS.
    Costs below are on-demand hourly on Community Cloud as of 2026-04.
  EOT
  default = [
    "NVIDIA GeForce RTX 3070",        # $0.13/hr, 8GB
    "NVIDIA RTX A4000",               # $0.17/hr, 16GB
    "NVIDIA GeForce RTX 3080",        # $0.17/hr, 10GB
    "NVIDIA GeForce RTX 3080 Ti",     # $0.18/hr, 12GB
    "NVIDIA RTX A4500",               # $0.19/hr, 20GB
    "NVIDIA GeForce RTX 3090",        # $0.22/hr, 24GB
    "NVIDIA GeForce RTX 4070 Ti",     # $0.19/hr, 12GB
    "NVIDIA RTX 4000 Ada Generation", # $0.20/hr, 20GB
    "NVIDIA GeForce RTX 4090",        # $0.34/hr, 24GB (last resort)
  ]
}

variable "data_center_ids" {
  type        = list(string)
  description = <<-EOT
    RunPod data center IDs, ordered by proximity to the Northeast US (NYC).
    Round-trip to NYC: US-MD-1 ~15ms, US-PA-1 ~8ms (if available), CA-MTL-* ~20ms,
    US-NC-* ~25ms, US-IL-1 ~30ms. Widen down-list on availability errors.
  EOT
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
  description = "Container disk (ephemeral); raise if image pull/install runs out of space"
  default     = 50
}

variable "pod_volume_in_gb" {
  type        = number
  description = "Attached /workspace volume (persists across restarts)"
  default     = 20
}

variable "swap_service_api_key" {
  type        = string
  sensitive   = true
  description = "Bearer secret for /v1/swap (same value as DLC_REMOTE_SWAP_API_KEY on your Mac)"
}

variable "swap_code_git_url" {
  type        = string
  description = "Git URL to clone on first boot (must contain remote-swap-server under subpath)"
}

variable "swap_code_repo_subpath" {
  type        = string
  description = "Path from repo root to remote-swap-server directory"
  default     = "Deep-Live-Cam/remote-swap-server"
}

variable "ssh_public_key" {
  type        = string
  description = "ssh-ed25519 line for RunPod; injected as PUBLIC_KEY for root SSH on TCP 22"
  default     = ""
}
