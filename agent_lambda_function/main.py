"""
Medical RAG Agent API - Modular Version
FastAPI application with LangChain agent for medical Q&A and predictions.
"""
import os
import logging
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from mangum import Mangum

# Third-party imports
from qdrant_client import AsyncQdrantClient
from google import genai
import xgboost as xgb
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports
from config import (
    GEMINI_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    EMBEDDING_MODEL_ID,
    MODEL_ID,
    COLLECTION_NAME,
    VECTOR_SIZE,
    XGBOOST_MODEL_PATH
)
from models import ChatRequest, ChatResponse, Citation
import tools as tool_module

# ------------------------
# Logging Configuration
# ------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-agent")

# ------------------------
# Global Clients
# ------------------------

qdrant_client: Optional[AsyncQdrantClient] = None
genai_client: Optional[genai.Client] = None
xgboost_model = None

# ------------------------
# Agent Configuration
# ------------------------

SYSTEM_PROMPT = ('''
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
                Include a disclaimer about data completeness and its potential impact on accuracy when information is limited.
            </requirement>
            </output_requirements>
        </prediction_tool_guidelines>

        <citation>
            <rule>
                Reference sources in your response using [1], [2], etc. 
            </rule>
            <rule>
                The number of the reference must be the same as the idx returned by the tool. 
            </rule>
        </citation>

        </system_prompt>'''
)

def get_agent():
    """Initialize the LangChain agent."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing")

    llm = ChatGoogleGenerativeAI(
        model=MODEL_ID,
        temperature=0,
        google_api_key=GEMINI_API_KEY
    )

    tools = [
        tool_module.search_medical_records,
        tool_module.predict_alanine_aminotransferase
    ]

    agent_executor = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
    )
    
    return agent_executor

# ------------------------
# Application Lifecycle
# ------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize connections on startup, cleanup on shutdown."""
    global qdrant_client, genai_client, xgboost_model
    
    logger.info("🚀 Initializing Medical RAG Agent...")
    
    # Init Gemini Native Client
    if GEMINI_API_KEY:
        try:
            genai_client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("✓ Gemini Native Client initialized")
        except Exception as e:
            logger.error(f"✗ Gemini Init Failed: {e}")
            raise
    else:
        raise ValueError("GEMINI_API_KEY is required")

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
        raise

    # Load XGBoost
    try:
        xgboost_model = xgb.XGBRegressor()
        xgboost_model.load_model(XGBOOST_MODEL_PATH)
        logger.info(f"✓ XGBoost model loaded from {XGBOOST_MODEL_PATH}")
    except Exception as e:
        logger.error(f"✗ Failed to load XGBoost model: {e}")
        raise
    
    # Initialize tools with dependencies
    tool_module.init_tools(
        qdrant_client=qdrant_client,
        genai_client=genai_client,
        xgboost_model=xgboost_model,
        config={
            'EMBEDDING_MODEL_ID': EMBEDDING_MODEL_ID,
            'VECTOR_SIZE': VECTOR_SIZE,
            'COLLECTION_NAME': COLLECTION_NAME
        }
    )
    
    logger.info("✓ All services initialized successfully")

    yield

    # Cleanup
    logger.info("🛑 Shutting down...")
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
    return {
        "status": "healthy",
        "services": {
            "qdrant": "connected" if qdrant_client else "disconnected",
            "genai": "connected" if genai_client else "disconnected",
            "xgboost": "loaded" if xgboost_model else "not loaded"
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
    if not qdrant_client or not genai_client:
        raise HTTPException(status_code=503, detail="Services not initialized")

    # Get agent
    try:
        agent = get_agent()
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent initialization failed: {str(e)}")

    # Prepare input
    inputs = {"messages": [HumanMessage(content=request.message)]}
    
    try:
        # Invoke agent (using your working pattern)
        logger.info(f"💬 Processing: {request.message[:80]}...")
        result = await agent.ainvoke(inputs)
        
        logger.info(f"✓ Agent completed with {len(result['messages'])} messages")
        
        # Extract final message
        final_message = result["messages"][-1].content

        # Handle Gemini's list format
        if isinstance(final_message, list):
            final_message = final_message[0].get("text", "No response generated")
        
        # Extract tool usage
        tool_usage = []
        citations = []
        
        for msg in result["messages"]:
            # Track tool calls
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for t in msg.tool_calls:
                    tool_usage.append(t['name'])
                    logger.info(f"🔧 Tool used: {t['name']}")
            
            # Extract citations from search_medical_records responses
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                try:
                    # Try to parse JSON from tool responses
                    content_json = json.loads(msg.content)
                    if 'citations' in content_json and content_json['citations']:
                        for citation_data in content_json['citations']:
                            citations.append(Citation(**citation_data))
                        logger.info(f"📚 Extracted {len(content_json['citations'])} citations")
                except (json.JSONDecodeError, TypeError, KeyError):
                    pass  # Not a JSON response or no citations

        return ChatResponse(
            response=str(final_message),
            tool_calls=tool_usage,
            citations=citations
        )

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------
# AWS Lambda Handler
# ------------------------

handler = Mangum(app, lifespan="on")