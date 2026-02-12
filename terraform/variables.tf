variable "use_localstack" {
  type    = bool
  default = true
}

variable "gemini_api_key" {
  type        = string
  description = "The API key for Gemini 2.5"
  sensitive   = true
}