import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel

# Qdrant
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue
)

# Gemini Native SDK (for Embeddings)
from google import genai
from google.genai import types

# LangGraph & LangChain (For the Agent Logic)
#from langgraph.prebuilt import create_react_agent # This is the modern replacement
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import joblib

from utils import predict_alanine_aminotransferase


# ------------------------
# Configuration
# ------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-agent")

# Env Vars
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)

# Config to match Ingestion Lambda
EMBEDDING_MODEL_ID = os.environ.get("EMBEDDING_MODEL_ID", "gemini-embedding-001") 
COLLECTION_NAME = "medical_docs"
VECTOR_SIZE = 768

# Global Clients
qdrant_client: Optional[AsyncQdrantClient] = None
genai_client: Optional[genai.Client] = None

# ------------------------
# Helpers
# ------------------------

def generate_embedding_sync(text: str) -> List[float]:
    """
    Generates embedding using the native google-genai SDK.
    Run this in a thread to keep FastAPI async.
    """
    if not genai_client:
        raise ValueError("GenAI Client not initialized")
    
    try:
        # Exact same logic as ingestion lambda
        result = genai_client.models.embed_content(
            model=EMBEDDING_MODEL_ID,
            contents=[text],
            config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE)
        )
        if hasattr(result, 'embeddings') and result.embeddings:
            return result.embeddings[0].values
        return []
    except Exception as e:
        logger.error(f"Native Embedding Error: {e}")
        raise e

# ------------------------
# 1. The RAG Tool
# ------------------------

@tool
async def search_medical_records(
    query: str, 
    patient_id: Optional[str] = None, 
    clinician_id: Optional[str] = None
):
    """
    Search the medical vector database for relevant documents.
    ALWAYS use this tool when answering questions about medical history, diagnosis, or patient details.
    
    Args:
        query: The search query string (e.g., "symptoms of heart attack").
        patient_id: Optional. The specific ID of the patient (e.g., "PT-12345").
        clinician_id: Optional. The ID of the doctor/clinician (e.g., "CL-9876").
    """
    global qdrant_client
    
    if not qdrant_client:
        return "Error: Database connection not available."

    logger.info(f"🔍 Searching: '{query}' | Patient: {patient_id}")

    try:
        # 1. Generate Embedding (Native SDK in Thread)
        query_vector = await asyncio.to_thread(generate_embedding_sync, query)

        if not query_vector:
            return "Error: Failed to generate embedding for query."

        # 2. Build Filters
        must_filters = []
        if patient_id:
            must_filters.append(FieldCondition(key="patient_id", match=MatchValue(value=patient_id)))
        if clinician_id:
            must_filters.append(FieldCondition(key="clinician_id", match=MatchValue(value=clinician_id)))

        search_filter = Filter(must=must_filters) if must_filters else None

        # 3. Search Qdrant
        search_result = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5, 
            query_filter=search_filter,
            with_payload=True
        )

        # 4. Format Results
        if not search_result.points:
            return "No relevant medical records found matching criteria."

        formatted_hits = []
        for hit in search_result.points:
            payload = hit.payload
            formatted_hits.append(
                f"Source: {payload.get('file_name')} (Section: {payload.get('section')})\n"
                f"Patient: {payload.get('patient_id')} | Clinician: {payload.get('clinician_id')}\n"
                f"Content: {payload.get('text')}\n"
                "---"
            )
        
        return "\n".join(formatted_hits)

    except Exception as e:
        logger.error(f"Search tool error: {str(e)}")
        return f"Error occurred during search: {str(e)}"

# ------------------------
# 2. The Agent Setup
# ------------------------

def get_agent():
    """Initializes the LangGraph ReAct agent."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing")

    # LangChain wrapper for Gemini Chat (works well for logic)
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("MODEL_ID", "gemini-2.5-flash-lite") ,
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    tools = [
        search_medical_records, 
        predict_alanine_aminotransferase
    ]
    
    system_prompt = (
        "You are an expert Medical AI Assistant with access to two specific tools:\n\n"
        "1. 'search_medical_records': Use this to find information in the patient's history, "
        "doctor notes, or medical documents. Use this for qualitative questions.\n\n"
        "2. 'predict_alanine_aminotransferase': Use this when a user asks for a 'prediction', "
        "'expected value', or 'forecast' of Alanine Aminotransferase (ALT) levels.\n\n"
        "Guidelines for the Prediction Tool:\n"
        "- If the user says 'woman' or 'lady', map sex to 'Female'. If 'man' or 'guy', map to 'Male'.\n"
        "- If the user says 'athlete' or 'gym', map exercise_frequency to 'High'.\n"
        "- If the user says 'city' or 'downtown', set urban to 1.\n"
        "- If the user DOES NOT provide a value (like BMI or Smoker), leave it blank so the tool uses its clinical defaults.\n"
        "- Always provide the final result clearly with units (U/L)."
    )

    agent_executor = create_agent(
        model=llm, 
        tools=tools, 
        system_prompt=system_prompt
    )
    
    return agent_executor

# ------------------------
# 3. FastAPI & Lifespan
# ------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize connections on startup."""
    global qdrant_client, genai_client
    logger.info("Initializing RAG Lambda...")
    
    # Init Gemini Native Client (for Embeddings)
    if GEMINI_API_KEY:
        try:
            genai_client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("✓ Gemini Native Client initialized")
        except Exception as e:
            logger.error(f"✗ Gemini Init Failed: {e}")

    # Init Qdrant
    try:
        qdrant_client = AsyncQdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
        )
        await qdrant_client.get_collections()
        logger.info("✓ Connected to Qdrant")
    except Exception as e:
        logger.error(f"✗ Qdrant Connection Failed: {e}")

    # --- Load XGBoost ---
    try:
        # Load your saved model
        xgboost_model = joblib.load("model.pkl")
        logger.info("✓ XGBoost model.pkl loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load model.pkl: {e}")

    yield

    if qdrant_client:
        await qdrant_client.close()

app = FastAPI(
    title="Medical RAG Agent",
    lifespan=lifespan
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    tool_calls: List[str] = []

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not qdrant_client or not genai_client:
        raise HTTPException(status_code=503, detail="Services not initialized")

    agent = get_agent()
    inputs = {"messages": [HumanMessage(content=request.message)]}
    
    try:
        # LangGraph invocation
        result = await agent.ainvoke(inputs)
        
        # Result is a dict with 'messages' list. Last one is the AI response.
        final_message = result["messages"][-1].content
        
        # Debug: Check for tool usage
        tool_usage = []
        for m in result["messages"]:
            if hasattr(m, 'tool_calls') and m.tool_calls:
                for t in m.tool_calls:
                    tool_usage.append(t['name'])

        return ChatResponse(response=str(final_message), tool_calls=tool_usage)

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

handler = Mangum(app, lifespan="on")