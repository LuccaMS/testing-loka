"""
Medical RAG Agent API
FastAPI application with LangGraph agent for medical Q&A and predictions.
"""
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

import xgboost as xgb
from qdrant_client import AsyncQdrantClient
from google import genai

# Local imports
from config import (
    GEMINI_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    XGBOOST_MODEL_PATH
)
from tools.dependencies import (
    set_qdrant_client,
    set_genai_client,
    set_xgboost_model,
    get_qdrant_client,
    get_genai_client
)
from agent import get_agent

# ------------------------
# Logging Configuration
# ------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-agent")

# ------------------------
# Application Lifecycle
# ------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize connections and models on startup, cleanup on shutdown."""
    logger.info("🚀 Initializing Medical RAG Agent...")
    
    # Initialize Gemini Client for embeddings
    if GEMINI_API_KEY:
        try:
            genai_client = genai.Client(api_key=GEMINI_API_KEY)
            set_genai_client(genai_client)
            logger.info("✓ Gemini Native Client initialized")
        except Exception as e:
            logger.error(f"✗ Gemini initialization failed: {e}")
            raise
    else:
        logger.error("✗ GEMINI_API_KEY not provided")
        raise ValueError("GEMINI_API_KEY is required")
    
    # Initialize Qdrant Vector Database
    try:
        qdrant_client = AsyncQdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
        )
        # Test connection
        await qdrant_client.get_collections()
        set_qdrant_client(qdrant_client)
        logger.info("✓ Connected to Qdrant")
    except Exception as e:
        logger.error(f"✗ Qdrant connection failed: {e}")
        raise
    
    # Load XGBoost Model
    try:
        xgboost_model = xgb.XGBRegressor()
        xgboost_model.load_model(XGBOOST_MODEL_PATH)
        set_xgboost_model(xgboost_model)
        logger.info(f"✓ XGBoost model loaded from {XGBOOST_MODEL_PATH}")
    except Exception as e:
        logger.error(f"✗ Failed to load XGBoost model: {e}")
        raise
    
    logger.info("✓ All services initialized successfully")
    
    # Application is now ready to handle requests
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down...")
    qdrant_client = get_qdrant_client()
    if qdrant_client:
        await qdrant_client.close()
        logger.info("✓ Qdrant connection closed")

# ------------------------
# FastAPI Application
# ------------------------

app = FastAPI(
    title="Medical RAG Agent",
    description="AI-powered medical assistant with RAG and predictive capabilities",
    version="2.0.0",
    lifespan=lifespan
)

# ------------------------
# Request/Response Models
# ------------------------

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What are the symptoms recorded for patient PT-12345?"
            }
        }


class Citation(BaseModel):
    """Citation information for RAG results."""
    id: int
    file_name: str
    section: str
    patient_id: str
    clinician_id: str
    score: float | None = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    tool_calls: List[str] = []
    citations: List[Citation] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Based on the medical records [1], the patient...",
                "tool_calls": ["search_medical_records"],
                "citations": [
                    {
                        "id": 1,
                        "file_name": "discharge_summary.pdf",
                        "section": "Chief Complaint",
                        "patient_id": "PT-12345",
                        "clinician_id": "CL-9876",
                        "score": 0.89
                    }
                ]
            }
        }

# ------------------------
# API Endpoints
# ------------------------

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Medical RAG Agent",
        "version": "2.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check with service status."""
    qdrant_client = get_qdrant_client()
    genai_client = get_genai_client()
    
    return {
        "status": "healthy",
        "services": {
            "qdrant": "connected" if qdrant_client else "disconnected",
            "genai": "connected" if genai_client else "disconnected",
            "xgboost": "loaded"
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for interacting with the medical AI agent.
    
    Args:
        request: ChatRequest with user message
        
    Returns:
        ChatResponse with agent's reply, tool calls, and citations
        
    Raises:
        HTTPException: If services not initialized or agent error
    """
    # Verify services are initialized
    if not get_qdrant_client() or not get_genai_client():
        raise HTTPException(
            status_code=503,
            detail="Services not fully initialized"
        )
    
    # Initialize agent
    try:
        agent = get_agent()
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise HTTPException(status_code=500, detail="Agent initialization failed")
    
    # Prepare agent input
    inputs = {"messages": [HumanMessage(content=request.message)]}
    
    try:
        # Invoke agent
        result = await agent.ainvoke(inputs)
        
        logger.info(f"Agent result: {len(result['messages'])} messages")
        
        # Extract final message
        final_message = result["messages"][-1].content
        
        # Handle Gemini's list-based content format
        if isinstance(final_message, list):
            final_message = final_message[0].get("text", "No response generated")
        
        # Extract tool usage information
        tool_usage = []
        citations = []
        
        for msg in result["messages"]:
            # Track tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_usage.append(tool_call['name'])
            
            # Extract citations from tool results
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                # Try to parse JSON responses from search_medical_records
                import json
                try:
                    content_json = json.loads(msg.content)
                    if 'citations' in content_json:
                        for citation_data in content_json['citations']:
                            citations.append(Citation(**citation_data))
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass  # Not a JSON response or no citations
        
        return ChatResponse(
            response=str(final_message),
            tool_calls=tool_usage,
            citations=citations
        )
        
    except Exception as e:
        logger.error(f"Agent execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

# ------------------------
# AWS Lambda Handler
# ------------------------

handler = Mangum(app, lifespan="on")