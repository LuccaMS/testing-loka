resource "aws_lambda_function" "ingestion_lambda" {
  function_name = "ingestion-lambda"
  role          = aws_iam_role.lambda_role.arn
  runtime       = "python3.12"
  handler       = "main.handler"

  timeout = 30

  memory_size = 1024

  s3_bucket = "hot-reload"

  #the key HAS to be an absolute path
  s3_key    = "/home/lucca/testing-loka/ingestion_lambda_function/dist" # <-- CHANGE THIS to your absolute path

  environment {
    variables = {
      GEMINI_API_KEY = var.gemini_api_key
      QDRANT_HOST    = "qdrant" # Change back to "qdrant" - it works!
      QDRANT_PORT    = "6333"
      MODEL_ID       = "gemini-embedding-001"
      ENVIRONMENT    = "production"
    }
  }
}

resource "aws_lambda_function_url" "ingestion_lambda" {
  function_name      = aws_lambda_function.ingestion_lambda.function_name
  authorization_type = "NONE"

  cors {
    allow_origins = ["*"]
    allow_methods = ["*"]
    allow_headers = ["*"]
  }
}

output "ingestion_lambda_function_url" {
  description = "ingestion ambda function url"
  value       = aws_lambda_function_url.ingestion_lambda.function_url
}
