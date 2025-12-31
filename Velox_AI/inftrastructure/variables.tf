variable "project_id" {
  description = "The ID of the GCP project"
  type        = string
  default     = "velox-ai-prod-2025"
}

variable "region" {
  description = "The region for resources"
  type        = string
  default     = "europe-west2"
}