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

import xgboost as xgb

# LangGraph & LangChain (For the Agent Logic)
#from langgraph.prebuilt import create_react_agent # This is the modern replacement
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import joblib

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
xgboost_model = None

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

from langchain_core.tools import tool
from typing import Literal
import pandas as pd

#The default values on the pydantic object were extract on  "notebooks/feature_engineering.ipynb". 
#These values are the most frequent ones, so if the person do not give this information we use
#the most frequent as default

# Mapping dictionaries based on the feature engineering logic
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
    sex: Literal["Male", "Female"] = "Female",
    bmi: float = 26.9,
    smoker: Literal["Yes", "No"] = "No",
    diagnosis_code: Literal["D1", "D2", "D3", "D4", "D5"] = "D5",
    medication_count: int = 3,
    days_hospitalized: int = 5,
    readmitted: Literal["No", "Yes"] = "No",
    last_lab_glucose: float = 100.1,
    exercise_frequency: Literal["Low", "Moderate", "High"] = "Moderate",
    diet_quality: Literal["Poor", "Average", "Good"] = "Average",
    income_bracket: Literal["Low", "Middle", "High"] = "Middle",
    education_level: Literal["Primary", "Secondary", "Tertiary"] = "Secondary",
    urban: Literal["No", "Yes"] = "Yes",
    albumin_globulin_ratio: float = 0.5037,
) -> float:
    """
    Predicts the Alanine Aminotransferase (ALT) levels for a patient using an XGBoost model.
    Use this tool when users ask for predictions, forecasts, or expected liver enzyme values.
    """
    global xgboost_model # Loaded in lifespan
    
    if xgboost_model is None:
        return "Error: Prediction model is not currently loaded in the system."

    try:
        # 1. Transform literals to numeric values used in training
        data_dict = {
            'age': age,
            'sex': MAPPINGS["sex"][sex],
            'bmi': bmi,
            'smoker': MAPPINGS["smoker"][smoker],
            'diagnosis_code': MAPPINGS["diagnosis_code"][diagnosis_code],
            'medication_count': medication_count,
            'days_hospitalized': days_hospitalized,
            'readmitted':  MAPPINGS["readmitted"][readmitted],
            'last_lab_glucose': last_lab_glucose,
            'exercise_frequency': MAPPINGS["exercise_frequency"][exercise_frequency],
            'diet_quality': MAPPINGS["diet_quality"][diet_quality],
            'income_bracket': MAPPINGS["income_bracket"][income_bracket],
            'education_level': MAPPINGS["education_level"][education_level],
            'urban': MAPPINGS["urban"][urban],
            'albumin_globulin_ratio': albumin_globulin_ratio
        }

        # 2. Convert to DataFrame
        # IMPORTANT: This order MUST match X = data.drop(...) from your training script
        df = pd.DataFrame([data_dict])
        
        # Ensure the columns are in the exact order the model expects
        column_order = [
            'age', 'sex', 'bmi', 'smoker', 'diagnosis_code', 'medication_count',
            'days_hospitalized', 'readmitted', 'last_lab_glucose', 'exercise_frequency',
            'diet_quality', 'income_bracket', 'education_level', 'urban', 'albumin_globulin_ratio'
        ]
        df = df[column_order]

        # 3. Predict
        prediction = xgboost_model.predict(df)[0]

        return (
            f"Based on the clinical parameters provided, the predicted Alanine Aminotransferase (ALT) "
            f"level is {float(prediction):.2f} U/L."
        )

    except Exception as e:
        logger.error(f"XGBoost Prediction Error: {e}")
        return f"I encountered an error while trying to calculate the prediction: {str(e)}"

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

def get_agent():
    """Initializes the LangGraph ReAct agent."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing")

    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("MODEL_ID", "gemini-2.0-flash-lite") ,
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    tools = [
        search_medical_records, 
        predict_alanine_aminotransferase
    ]
    
    system_prompt = ('''
        <system_prompt>
        <role>
            You are an expert Medical AI Assistant with access to two specific tools.
        </role>

        <available_tools>
            <tool>
            <name>search_medical_records</name>
            <description>
                Use this to find information in the patient's history, doctor notes, or medical documents. Use this for qualitative questions.
            </description>
            </tool>
            
            <tool>
            <name>predict_alanine_aminotransferase</name>
            <description>
                Use this to predict Alanine Aminotransferase (ALT) levels.
            </description>
            <critical_rule>
                MUST ALWAYS provide a prediction when requested by the user, EVEN WITHOUT COMPLETE INFORMATION. Never refuse to predict due to missing data.
            </critical_rule>
            </tool>
        </available_tools>

        <prediction_tool_guidelines>
            <feature_mapping>
            <mapping>
                <user_terms>woman, lady</user_terms>
                <feature>Female</feature>
            </mapping>
            <mapping>
                <user_terms>man, guy</user_terms>
                <feature>Male</feature>
            </mapping>
            <mapping>
                <user_terms>athlete, gym</user_terms>
                <feature>High exercise_frequency</feature>
            </mapping>
            <mapping>
                <user_terms>city, downtown</user_terms>
                <feature>urban = 1</feature>
            </mapping>
            </feature_mapping>

            <missing_data_handling>
            <rule>
                If the user does not provide a value (like BMI or smoker status), leave it blank so the tool uses its clinical defaults.
            </rule>
            <rule>
                ALWAYS predict ALT when requested, regardless of missing information.
            </rule>
            <rule>
                When providing predictions with incomplete data, clearly inform the user: "This is a prediction based on available data. However, this prediction has limited information which may impact accuracy. Providing additional information, especially BMI, would significantly improve the prediction's reliability."
            </rule>
            </missing_data_handling>

            <output_requirements>
            <requirement>
                Always provide the final result clearly with units (U/L).
            </requirement>
            <requirement>
                Include a disclaimer about data completeness and its potential impact on accuracy when information is limited.
            </requirement>
            </output_requirements>
        </prediction_tool_guidelines>
        </system_prompt>'''
    )

    agent_executor = create_agent(
        model=llm, 
        tools=tools, 
        system_prompt=system_prompt
    )
    
    return agent_executor

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
        global xgboost_model
        xgboost_model = xgb.XGBRegressor()
        xgboost_model.load_model("model_alt.json")  # ✅ Matches your filename
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

        print(result["messages"])
        
        # Result is a dict with 'messages' list. Last one is the AI response.
        final_message = result["messages"][-1].content

        #usually this is not needed, but gemini  "gemini-3-flash-preview" add signatures to tool usage
        if isinstance(final_message, list):
            final_message = final_message.get("text", "empty")
            
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