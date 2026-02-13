"""
Configuration Module
Centralized configuration management for the RAG agent.
"""
import os

# API Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Qdrant Configuration
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)

# Model Configuration
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "gemini-embedding-001")
MODEL_ID = os.environ.get("MODEL_ID", "gemini-2.0-flash-lite")
COLLECTION_NAME = "medical_docs"
VECTOR_SIZE = 768

# Model Files
XGBOOST_MODEL_PATH = "model_alt.json"
