# Quickstart Guide

## Prerequisites

To run this project, you need:

- A Linux environment (WSL or native)
- Terraform installed
- Docker installed

## Setup Instructions

### 1. Start Local Services

Run the following command to create LocalStack and Qdrant:
```bash
docker compose up -d
```

### 2. Create Gemini API Key

This project uses Gemini API (which has free quota available).

Create your API key here: https://aistudio.google.com/api-keys

### 3. Build Lambda Functions

Before running Terraform, you need to build the lambda functions.

Navigate to the `agent_lambda_function` folder and run:
```bash
chmod +x ./build.sh
sudo ./build.sh
```

Then navigate to the `ingestion_lambda_function` folder and run:
```bash
chmod +x ./build.sh
sudo ./build.sh
```

This step is required because the lambdas will consume these built artifacts.

#### You MUST run with sudo, otherwise the hot reload of the lambdas won't work.

### 4. Initialize Terraform

Navigate to the terraform folder and run:
```bash
terraform init
terraform apply
```

You will be prompted for your API key. After entering it, Terraform will show which resources it will create. Type `yes` to confirm.

### 5. Expected Output

After Terraform finishes, you should see output similar to:
```
Apply complete! Resources: 5 added, 0 changed, 0 destroyed.

Outputs:
function_url = "http://78l3096tz6hyzc91grco9v9nqj7xxxt6.lambda-url.us-east-1.localhost.localstack.cloud:4566/"
ingestion_lambda_function_url = "http://6k4fkpmmw8q04oaaifr9ar6ruz5k9ehk.lambda-url.us-east-1.localhost.localstack.cloud:4566/"
```

## Lambda Functions

Both lambdas use the Magnum library and are FastAPI ASGI apps. You have access to the `/docs` and `/redoc` endpoints for API documentation.

### function_url

The main agent lambda that:
- Performs RAG (Retrieval Augmented Generation)
- Uses a regression model to predict alanine_aminotransferase
- Performs analysis on CSV files

### ingestion_lambda_function_url

The ingestion lambda that:
- Manages the vector database
- Accepts markdown document uploads for RAG operations