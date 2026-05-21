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
  internal_port = 8000
  # https://docs.runpod.io/pods/configuration/expose-ports
  swap_public_url = "https://${runpod_pod.dlc_remote_swap.id}-${local.internal_port}.proxy.runpod.net"
}

resource "runpod_pod" "dlc_remote_swap" {
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

  ports = ["${local.internal_port}/http", "22/tcp"]

  env = merge(
    {
      SWAP_SERVICE_API_KEY = var.swap_service_api_key
      SWAP_MODEL_TYPE      = var.swap_model_type
      SWAP_REPO_URL        = var.swap_code_git_url
      SWAP_REPO_SUBPATH    = var.swap_code_repo_subpath
      PORT                 = tostring(local.internal_port)
    },
    var.ssh_public_key == "" ? {} : { PUBLIC_KEY = var.ssh_public_key }
  )

  docker_start_cmd = [
    "/bin/bash",
    "-lc",
    file("${path.module}/bootstrap_on_pod.sh"),
  ]
}
