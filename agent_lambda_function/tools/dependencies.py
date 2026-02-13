"""
Dependency Management
Centralized access to shared resources (clients, models).
"""
from typing import Optional, Callable
from qdrant_client import AsyncQdrantClient
from google import genai
import xgboost as xgb

# Global instances (initialized during app startup)
_qdrant_client: Optional[AsyncQdrantClient] = None
_genai_client: Optional[genai.Client] = None
_xgboost_model: Optional[xgb.XGBRegressor] = None


def set_qdrant_client(client: AsyncQdrantClient):
    """Set the global Qdrant client instance."""
    global _qdrant_client
    _qdrant_client = client


def get_qdrant_client() -> Optional[AsyncQdrantClient]:
    """Get the global Qdrant client instance."""
    return _qdrant_client


def set_genai_client(client: genai.Client):
    """Set the global GenAI client instance."""
    global _genai_client
    _genai_client = client


def get_genai_client() -> Optional[genai.Client]:
    """Get the global GenAI client instance."""
    return _genai_client


def set_xgboost_model(model: xgb.XGBRegressor):
    """Set the global XGBoost model instance."""
    global _xgboost_model
    _xgboost_model = model


def get_xgboost_model() -> Optional[xgb.XGBRegressor]:
    """Get the global XGBoost model instance."""
    return _xgboost_model


def get_embedding_function() -> Callable[[str], list]:
    """
    Returns a function that generates embeddings using the GenAI client.
    This function can be called synchronously within an async context using to_thread.
    """
    from google.genai import types
    from config import EMBEDDING_MODEL_ID, VECTOR_SIZE
    import logging
    
    logger = logging.getLogger("rag-agent.embedding")
    
    def generate_embedding(text: str) -> list:
        """Generate embedding for given text."""
        client = get_genai_client()
        
        if not client:
            raise ValueError("GenAI Client not initialized")
        
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL_ID,
                contents=[text],
                config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE)
            )
            
            if hasattr(result, 'embeddings') and result.embeddings:
                return result.embeddings[0].values
            return []
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise e
    
    return generate_embedding