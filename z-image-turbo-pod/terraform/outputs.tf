output "pod_id" {
  value       = runpod_pod.z_image_turbo.id
  description = "RunPod pod id (proxy hostname prefix)"
}

output "zimage_service_url" {
  value       = local.zimage_public_url
  description = "HTTPS base URL (RunPod proxy)"
}

output "curl_example" {
  value       = <<-EOT
    # Wait until /health returns {"status":"ok"} (first boot: model download + load, often 15–30+ min).
    curl -sS "${local.zimage_public_url}/health"
    curl -sS -X POST "${local.zimage_public_url}/generate" \\
      -H 'Content-Type: application/json' \\
      -d '{"prompt":"a red apple on a wooden table","width":1024,"height":1024,"seed":42}' | head -c 200
  EOT
  description = "Smoke-test commands"
}

output "runpod_ssh_hint" {
  value       = <<-EOT
    Use the Connect tab for the exact ssh.runpod.io username (pod id + suffix).
    Example: ssh ${runpod_pod.z_image_turbo.id}-XXXXXXXX@ssh.runpod.io -i ~/.ssh/your_key
  EOT
  description = "SSH via RunPod proxy"
}
