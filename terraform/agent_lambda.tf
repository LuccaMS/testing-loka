resource "aws_lambda_function" "python_lambda" {
  function_name = "agent-lambda"
  role          = aws_iam_role.lambda_role.arn
  runtime       = "python3.12"
  handler       = "handler.handler"
  timeout       = 300

  memory_size = 1024

  #using hot reload for localstack
  s3_bucket = "hot-reload"
  
  #the key HAS to be an absolute path
  s3_key    = "/home/lucca/testing-loka/agent_lambda_function/dist" # <-- CHANGE THIS to your absolute path

  environment {
    variables = {
      GEMINI_API_KEY     = var.gemini_api_key
      QDRANT_HOST         = "qdrant"
      QDRANT_PORT         = "6333"
      MODEL_ID           = "gemini-2.5-flash-lite"
      EMBEDDING_MODEL_ID = "gemini-embedding-001"
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
