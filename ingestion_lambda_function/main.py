import os
import logging
import uuid
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status, UploadFile, File, Depends
from pydantic import BaseModel, Field
from mangum import Mangum

from google import genai
from google.genai import types

# Use Async Client for non-blocking I/O in FastAPI/Lambda
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from utils import process_markdown_content

# ------------------------
# Config & Globals
# ------------------------

COLLECTION_NAME = "medical_docs"
VECTOR_SIZE = 768

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical-api")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_ID = os.environ.get("MODEL_ID", "gemini-embedding-001")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)

# Global Client placeholders
global_clients = {
    "qdrant": None,
    "gemini": None
}

# ------------------------
# Lifespan (Startup/Shutdown)
# ------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles connection initialization on Lambda Cold Start.
    """
    logger.info("🚀 Starting up Medical Ingestion API...")
    
    # 1. Initialize Gemini Client
    if GEMINI_API_KEY:
        try:
            # Initialize Gemini Client
            global_clients["gemini"] = genai.Client(api_key=GEMINI_API_KEY)
            logger.info("✓ Gemini client initialized")
        except Exception as e:
            logger.error(f"✗ Failed to initialize Gemini: {e}")
    else:
        logger.warning("! GEMINI_API_KEY not found")

    # 2. Initialize Qdrant Client (Async)
    try:
        client = AsyncQdrantClient(
            host=QDRANT_HOST, 
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
        )
        # Quick check
        await client.get_collections()
        global_clients["qdrant"] = client
        logger.info(f"✓ Connected to Qdrant at {QDRANT_HOST}")
        
        # Ensure Collection Exists
        await ensure_collection(client)
        
    except Exception as e:
        logger.error(f"✗ Failed to connect to Qdrant: {e}")

    yield
    
    # Shutdown logic
    if global_clients["qdrant"]:
        await global_clients["qdrant"].close()
        logger.info("Qdrant connection closed")

# ------------------------
# Models
# ------------------------

class HealthResponse(BaseModel):
    status: str
    collection: str
    services: Dict[str, bool]

class IngestResponse(BaseModel):
    results: List[Dict[str, Any]]

class SearchResult(BaseModel):
    score: float
    text: str
    document_id: str
    section: Optional[str]
    file_name: Optional[str]
    patient_id: Optional[str] = None
    clinician_id: Optional[str] = None

# ------------------------
# FastAPI App
# ------------------------

app = FastAPI(
    title="Medical Document Ingestion API",
    version="1.2.0",
    lifespan=lifespan
)

# ------------------------
# Dependencies & Helpers
# ------------------------

async def get_qdrant_client() -> AsyncQdrantClient:
    """Dependency to get the Qdrant client."""
    client = global_clients.get("qdrant")
    if not client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant service not available"
        )
    return client

def get_gemini_client():
    """Dependency to get the Gemini client."""
    client = global_clients.get("gemini")
    if not client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini service not available"
        )
    return client

async def ensure_collection(client: AsyncQdrantClient):
    """Ensures Qdrant collection exists."""
    try:
        await client.get_collection(COLLECTION_NAME)
    except Exception:
        logger.info(f"Creating collection '{COLLECTION_NAME}'...")
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

def generate_embeddings(client: genai.Client, texts: List[str]) -> List[List[float]]:
    """Batch embedding using Gemini."""
    try:
        result = client.models.embed_content(
            model=MODEL_ID,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE)
        )
        if hasattr(result, 'embeddings'):
            return [emb.values for emb in result.embeddings]
        return []
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

def generate_deterministic_uuid(input_str: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, input_str))

# ------------------------
# Routes
# ------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    q_client = global_clients.get("qdrant")
    g_client = global_clients.get("gemini")
    
    q_status = False
    if q_client:
        try:
            await q_client.get_collection(COLLECTION_NAME)
            q_status = True
        except:
            pass

    return {
        "status": "healthy" if (q_status and g_client) else "degraded",
        "collection": COLLECTION_NAME,
        "services": {
            "qdrant": q_status,
            "gemini": g_client is not None
        }
    }

@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest_markdowns(
    files: List[UploadFile] = File(...),
    q_client: AsyncQdrantClient = Depends(get_qdrant_client),
    g_client: genai.Client = Depends(get_gemini_client)
):
    results = []

    for file in files:
        try:
            # 1. Read and Process (CPU bound -> ThreadPool)
            content_bytes = await file.read()
            raw_content = content_bytes.decode("utf-8")
            
            processed = await asyncio.to_thread(
                process_markdown_content, raw_content, file.filename
            )
            
            file_hash = processed["file_hash"]
            doc_id = processed["document_id"]

            # 2. Check for Duplicates
            existing_count = await q_client.count(
                collection_name=COLLECTION_NAME,
                count_filter=Filter(
                    must=[FieldCondition(key="file_hash", match=MatchValue(value=file_hash))]
                )
            )
            
            if existing_count.count > 0:
                results.append({
                    "file": file.filename, 
                    "status": "skipped", 
                    "reason": "Duplicate SHA256"
                })
                continue

            # 3. Generate Embeddings (Network/CPU bound -> ThreadPool)
            texts = [c["content"] for c in processed["chunks"]]
            if not texts:
                results.append({"file": file.filename, "status": "skipped", "reason": "Empty content"})
                continue

            embeddings = await asyncio.to_thread(generate_embeddings, g_client, texts)

            # 4. Prepare Points
            points = []
            for i, (chunk, vector) in enumerate(zip(processed["chunks"], embeddings)):
                point_id = generate_deterministic_uuid(f"{file_hash}_{i}")
                
                payload = {
                    "file_hash": file_hash,
                    "document_id": doc_id,
                    "text": chunk["content"],
                    "chunk_index": chunk["chunk_index"],
                    "section": chunk["metadata"].get("Section", "General"),
                    "patient_id": chunk["metadata"].get("patient_id"),
                    "clinician_id": chunk["metadata"].get("clinician_id"),
                    "file_name": file.filename
                }
                
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))

            # 5. Upsert
            await q_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )

            results.append({
                "file": file.filename, 
                "doc_id": doc_id, 
                "status": "success", 
                "chunks": len(points)
            })

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}", exc_info=True)
            results.append({"file": file.filename, "status": "error", "message": str(e)})

    return {"results": results}

@app.get("/search", tags=["Search"])
async def search_documents(
    query: str, 
    limit: int = 5,
    q_client: AsyncQdrantClient = Depends(get_qdrant_client),
    g_client: genai.Client = Depends(get_gemini_client)
):
    # 1. Embed Query
    query_embeddings = await asyncio.to_thread(generate_embeddings, g_client, [query])
    if not query_embeddings:
        raise HTTPException(status_code=500, detail="Failed to embed query")
    
    query_vector = query_embeddings[0]

    # 2. Search Qdrant (Async) using query_points
    response = await q_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    )

    # 3. Format
    results = [
        SearchResult(
            score=hit.score,
            text=hit.payload.get("text"),
            document_id=hit.payload.get("document_id"),
            section=hit.payload.get("section"),
            file_name=hit.payload.get("file_name"),
            patient_id=hit.payload.get("patient_id"),
            clinician_id=hit.payload.get("clinician_id")
        )
        for hit in response.points
    ]

    return {"query": query, "results": results}

@app.delete("/admin/reset", tags=["Admin"])
async def reset_collection(q_client: AsyncQdrantClient = Depends(get_qdrant_client)):
    """Deletes and recreates the collection (useful for dev/testing)."""
    try:
        await q_client.delete_collection(COLLECTION_NAME)
        await ensure_collection(q_client)
        return {"message": "Collection reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------
# Lambda Handler
# ------------------------

handler = Mangum(app, lifespan="on")