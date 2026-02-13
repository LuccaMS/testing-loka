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
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Gemini Native SDK (for Embeddings)
from google import genai
from google.genai import types

import xgboost as xgb

# LangChain (For the Agent Logic)
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

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
MODEL_ID = os.environ.get("MODEL_ID", "gemini-2.0-flash-lite")
COLLECTION_NAME = "medical_docs"
VECTOR_SIZE = 768

# Global Clients
qdrant_client: Optional[AsyncQdrantClient] = None
genai_client: Optional[genai.Client] = None
xgboost_model = None

# ------------------------
# Helpers
# ------------------------

def generate_embedding_sync(text: str) -> List[float]:
    """Generates embedding using the native google-genai SDK."""
    if not genai_client:
        raise ValueError("GenAI Client not initialized")
    
    try:
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
# Tools
# ------------------------

# Mapping dictionaries
MAPPINGS = {
    'sex': {'Female': 0, 'Male': 1},
    "smoker": {"No": 0, "Yes": 1},
    "diagnosis_code": {"D1": 1.0, "D2": 2.0, "D3": 3.0, "D4": 4.0, "D5": 5.0},
    "exercise_frequency": {"Low": 0.0, "Moderate": 1.0, "High": 2.0},
    "diet_quality": {"Poor": 0.0, "Average": 1.0, "Good": 2.0},
    "income_bracket": {"Low": 0.0, "Middle": 1.0, "High": 2.0},
    "education_level": {"Primary": 0.0, "Secondary": 1.0, "Tertiary": 2.0},
    "urban": {"No": 0.0, "Yes": 1.0},
    "readmitted": {"No": 0.0, "Yes": 1.0}
}

@tool
def predict_alanine_aminotransferase(
    age: float = 53.0,
    sex: str = "Female",
    bmi: float = 26.9,
    smoker: str = "No",
    diagnosis_code: str = "D5",
    medication_count: int = 3,
    days_hospitalized: int = 5,
    readmitted: str = "No",
    last_lab_glucose: float = 100.1,
    exercise_frequency: str = "Moderate",
    diet_quality: str = "Average",
    income_bracket: str = "Middle",
    education_level: str = "Secondary",
    urban: str = "Yes",
    albumin_globulin_ratio: float = 0.5037,
) -> str:
    """
    Predicts the Alanine Aminotransferase (ALT) levels for a patient using an XGBoost model.
    Use this tool when users ask for predictions, forecasts, or expected liver enzyme values.
    """
    global xgboost_model
    
    if xgboost_model is None:
        return "Error: Prediction model is not currently loaded in the system."

    try:
        # Transform literals to numeric values
        data_dict = {
            'age': age,
            'sex': MAPPINGS["sex"].get(sex, 0),
            'bmi': bmi,
            'smoker': MAPPINGS["smoker"].get(smoker, 0),
            'diagnosis_code': MAPPINGS["diagnosis_code"].get(diagnosis_code, 5.0),
            'medication_count': medication_count,
            'days_hospitalized': days_hospitalized,
            'readmitted':  MAPPINGS["readmitted"].get(readmitted, 0),
            'last_lab_glucose': last_lab_glucose,
            'exercise_frequency': MAPPINGS["exercise_frequency"].get(exercise_frequency, 1.0),
            'diet_quality': MAPPINGS["diet_quality"].get(diet_quality, 1.0),
            'income_bracket': MAPPINGS["income_bracket"].get(income_bracket, 1.0),
            'education_level': MAPPINGS["education_level"].get(education_level, 1.0),
            'urban': MAPPINGS["urban"].get(urban, 1.0),
            'albumin_globulin_ratio': albumin_globulin_ratio
        }

        # Convert to DataFrame
        df = pd.DataFrame([data_dict])
        
        # Ensure correct column order
        column_order = [
            'age', 'sex', 'bmi', 'smoker', 'diagnosis_code', 'medication_count',
            'days_hospitalized', 'readmitted', 'last_lab_glucose', 'exercise_frequency',
            'diet_quality', 'income_bracket', 'education_level', 'urban', 'albumin_globulin_ratio'
        ]
        df = df[column_order]

        # Predict
        prediction = xgboost_model.predict(df)[0]

        return (
            f"Based on the clinical parameters provided, the predicted Alanine Aminotransferase (ALT) "
            f"level is {float(prediction):.2f} U/L."
        )

    except Exception as e:
        logger.error(f"XGBoost Prediction Error: {e}", exc_info=True)
        return f"I encountered an error while trying to calculate the prediction: {str(e)}"

@tool
async def search_medical_records(
    query: str,
    patient_id: Optional[str] = None,
    clinician_id: Optional[str] = None
) -> str:
    """
    Search the medical vector database for relevant documents.
    ALWAYS use this tool when answering questions about medical history, diagnosis, or patient details.
    
    Args:
        query: The search query string
        patient_id: Optional patient ID filter
        clinician_id: Optional clinician ID filter
    
    Returns:
        JSON string with search results and citations
    """
    global qdrant_client
    
    if not qdrant_client:
        return '{"error": "Database connection not available.", "citations": []}'

    logger.info(f"🔍 Searching: '{query}' | Patient: {patient_id}")

    try:
        # Generate Embedding
        query_vector = await asyncio.to_thread(generate_embedding_sync, query)

        if not query_vector:
            return '{"error": "Failed to generate embedding for query.", "citations": []}'

        # Build Filters
        must_filters = []
        if patient_id:
            must_filters.append(FieldCondition(key="patient_id", match=MatchValue(value=patient_id)))
        if clinician_id:
            must_filters.append(FieldCondition(key="clinician_id", match=MatchValue(value=clinician_id)))

        search_filter = Filter(must=must_filters) if must_filters else None

        # Search Qdrant
        search_result = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5,
            query_filter=search_filter,
            with_payload=True
        )

        # Format Results with Citations
        if not search_result.points:
            return '{"content": "No relevant medical records found matching criteria.", "citations": []}'

        formatted_content = []
        citations = []
        
        for idx, hit in enumerate(search_result.points, start=1):
            payload = hit.payload
            
            # Build citation
            citation = {
                "id": idx,
                "file_name": payload.get('file_name', 'Unknown'),
                "section": payload.get('section', 'N/A'),
                "patient_id": payload.get('patient_id', 'N/A'),
                "clinician_id": payload.get('clinician_id', 'N/A'),
                "score": float(hit.score) if hasattr(hit, 'score') else None
            }
            citations.append(citation)
            
            # Format content with citation markers
            content_piece = (
                f"[{idx}] {payload.get('text', '')}\n"
                f"(Source: {payload.get('file_name')} - {payload.get('section')})"
            )
            formatted_content.append(content_piece)
        
        import json
        result = {
            "content": "\n\n".join(formatted_content),
            "citations": citations
        }
        
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Search tool error: {str(e)}", exc_info=True)
        import json
        return json.dumps({
            "error": f"Error occurred during search: {str(e)}",
            "citations": []
        })

# ------------------------
# Agent
# ------------------------

def get_agent():
    """Initializes the LangChain agent."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing")

    llm = ChatGoogleGenerativeAI(
        model=MODEL_ID,
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    tools = [
        search_medical_records,
        predict_alanine_aminotransferase
    ]
    
    system_prompt = '''You are an expert Medical AI Assistant with access to two specific tools.

AVAILABLE TOOLS:
1. search_medical_records - Returns JSON with "content" and "citations"
2. predict_alanine_aminotransferase - Predicts ALT levels

When using search results, reference citations like [1], [2], etc.
Always provide sources at the end of your response.
'''

    agent_executor = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )
    
    return agent_executor

# ------------------------
# Lifecycle
# ------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize connections on startup."""
    global qdrant_client, genai_client, xgboost_model
    logger.info("Initializing RAG Lambda...")
    
    # Init Gemini Native Client
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

    # Load XGBoost
    try:
        xgboost_model = xgb.XGBRegressor()
        xgboost_model.load_model("model_alt.json")
        logger.info("✓ XGBoost model loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")

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
        # LangGraph invocation (exactly as in your original)
        result = await agent.ainvoke(inputs)
        
        logger.info(f"Agent completed with {len(result['messages'])} messages")
        
        # Result is a dict with 'messages' list. Last one is the AI response.
        final_message = result["messages"][-1].content

        # Handle Gemini's list format
        if isinstance(final_message, list):
            final_message = final_message[0].get("text", "empty")
            
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