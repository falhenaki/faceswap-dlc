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
  description = "Ordered GPU preferences; first available is used when gpu_type_priority is availability"
  default = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4080",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A4000",
    "NVIDIA RTX A4500",
    "NVIDIA RTX A5000",
    "NVIDIA L4",
    "NVIDIA RTX 4000 Ada Generation",
  ]
}

variable "data_center_ids" {
  type        = list(string)
  description = "RunPod data center IDs. Widen if provisioning returns 500 (no capacity)."
  default = [
    "US-CA-2",
    "US-TX-3",
    "US-TX-4",
    "US-KS-2",
    "US-KS-3",
    "US-IL-1",
    "US-GA-1",
    "US-GA-2",
    "US-NC-1",
    "US-DE-1",
    "US-WA-1",
    "CA-MTL-1",
    "CA-MTL-2",
    "CA-MTL-3",
    "EU-CZ-1",
    "EU-RO-1",
    "EU-NL-1",
    "EU-FR-1",
    "EU-SE-1",
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
