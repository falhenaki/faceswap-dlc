terraform {
  required_version = ">= 1.5.0"
  required_providers {
    runpod = {
      source  = "decentralized-infrastructure/runpod"
      version = "~> 1.0"
    }
  }
}

provider "runpod" {}

locals {
  internal_port     = 8000
  zimage_public_url = "https://${runpod_pod.z_image_turbo.id}-${local.internal_port}.proxy.runpod.net"
}

resource "runpod_pod" "z_image_turbo" {
  name                 = var.pod_name
  image_name           = var.container_image
  support_public_ip    = var.cloud_type == "COMMUNITY"
  compute_type         = "GPU"
  gpu_count            = 1
  gpu_type_ids         = var.gpu_type_ids
  gpu_type_priority    = "availability"
  cloud_type           = var.cloud_type
  data_center_ids      = var.data_center_ids
  data_center_priority = "availability"
  container_disk_in_gb = var.container_disk_in_gb
  volume_in_gb         = var.pod_volume_in_gb
  volume_mount_path    = "/workspace"
  interruptible        = var.interruptible

  ports = ["${local.internal_port}/http", "22/tcp"]

  env = merge(
    {
      PORT               = tostring(local.internal_port)
      HF_HOME            = "/workspace/hf_cache"
      TRANSFORMERS_CACHE = "/workspace/hf_cache/hub"
      ZIMAGE_MODEL_ID    = var.zimage_model_id
      TORCH_DTYPE        = var.torch_dtype
    },
    var.hf_token == "" ? {} : { HF_TOKEN = var.hf_token },
    var.zimage_api_key == "" ? {} : { ZIMAGE_API_KEY = var.zimage_api_key },
    var.enable_model_cpu_offload ? { ENABLE_MODEL_CPU_OFFLOAD = "true" } : {},
    var.attention_backend == "" ? {} : { ATTENTION_BACKEND = var.attention_backend },
    var.ssh_public_key == "" ? {} : { PUBLIC_KEY = var.ssh_public_key }
  )
}
