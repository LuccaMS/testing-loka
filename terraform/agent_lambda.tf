resource "aws_lambda_function" "python_lambda" {
  function_name = "agent-lambda"
  role          = aws_iam_role.lambda_role.arn
  runtime       = "python3.12"
  handler       = "handler.handler"

  timeout = 30

  filename         = "../agent_lambda_function/lambda.zip"
  source_code_hash = filebase64sha256("../agent_lambda_function/lambda.zip")

  environment {
    variables = {
      GEMINI_API_KEY     = var.gemini_api_key
      QDRANT_HOST        = "qdrant" # Change back to "qdrant" - it works!
      QDRANT_PORT        = "6333"
      MODEL_ID           = "gemini-2.5-flash-lite"
      EMBEDDING_MODEL_ID = "gemini-embedding-001"
      ENVIRONMENT        = "production"
    }
  }
}

resource "aws_lambda_function_url" "example" {
  function_name      = aws_lambda_function.python_lambda.function_name
  authorization_type = "NONE"

  cors {
    allow_origins = ["*"]
    allow_methods = ["*"]
    allow_headers = ["*"]
  }
}

output "funcion_url" {
  description = "lambda function url"
  value       = aws_lambda_function_url.example.function_url
}
