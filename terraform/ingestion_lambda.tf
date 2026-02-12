resource "aws_lambda_function" "ingestion_lambda" {
  function_name = "ingestion-lambda"
  role          = aws_iam_role.lambda_role.arn
  runtime       = "python3.12"
  handler       = "main.handler"

  timeout = 30

  filename         = "../ingestion_lambda_function/lambda.zip"
  source_code_hash = filebase64sha256("../ingestion_lambda_function/lambda.zip")

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
