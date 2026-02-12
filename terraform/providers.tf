terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "6.31.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "2.7.1"
    }
  }
}

provider "aws" {
  region = "us-east-1"

  access_key = var.use_localstack ? "test" : null
  secret_key = var.use_localstack ? "test" : null

  s3_use_path_style           = var.use_localstack
  skip_credentials_validation = var.use_localstack
  skip_metadata_api_check     = var.use_localstack
  skip_requesting_account_id  = var.use_localstack

  dynamic "endpoints" {
    for_each = var.use_localstack ? [1] : []
    content {
      lambda = "http://localhost:4566"
      iam    = "http://localhost:4566"
      logs   = "http://localhost:4566"
      s3     = "http://localhost:4566"
    }
  }
}
