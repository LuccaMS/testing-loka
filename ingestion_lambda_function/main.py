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
MAX_FILES_PER_REQUEST = 5

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
    summary: Dict[str, int] = Field(default_factory=dict)

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
    version="1.3.0",
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

def validate_files(files: List[UploadFile], max_files: int = MAX_FILES_PER_REQUEST) -> None:
    """Validates uploaded files for count and extension."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )
    
    if len(files) > max_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {max_files} files allowed per request. Received {len(files)} files."
        )
    
    for file in files:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File with missing filename detected"
            )
        if not file.filename.lower().endswith('.md'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only .md (Markdown) files are allowed. Invalid file: {file.filename}"
            )

async def process_single_file(
    file: UploadFile,
    q_client: AsyncQdrantClient,
    g_client: genai.Client
) -> Dict[str, Any]:
    """Process a single file: read, chunk, embed, and upsert to Qdrant."""
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
            return {
                "file": file.filename, 
                "status": "skipped", 
                "reason": "Duplicate SHA256"
            }

        # 3. Generate Embeddings (Network/CPU bound -> ThreadPool)
        texts = [c["content"] for c in processed["chunks"]]
        if not texts:
            return {
                "file": file.filename, 
                "status": "skipped", 
                "reason": "Empty content"
            }

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

        return {
            "file": file.filename, 
            "doc_id": doc_id, 
            "status": "success", 
            "chunks": len(points)
        }

    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}", exc_info=True)
        return {
            "file": file.filename, 
            "status": "error", 
            "message": str(e)
        }

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
    files: List[UploadFile] = File(..., description=f"Upload 1-{MAX_FILES_PER_REQUEST} markdown (.md) files"),
    q_client: AsyncQdrantClient = Depends(get_qdrant_client),
    g_client: genai.Client = Depends(get_gemini_client)
):
    """
    Ingest multiple markdown files (up to 5) into the vector database.
    
    - Files are processed concurrently for better performance
    - Only .md files are accepted
    - Duplicate files (based on SHA256 hash) are automatically skipped
    """
    # Validate files before processing
    validate_files(files)
    
    # Process all files concurrently
    tasks = [
        process_single_file(file, q_client, g_client)
        for file in files
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Generate summary statistics
    summary = {
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "error": sum(1 for r in results if r["status"] == "error"),
        "total_chunks": sum(r.get("chunks", 0) for r in results if r["status"] == "success")
    }
    
    logger.info(f"Batch ingestion complete: {summary}")
    
    return {
        "results": results,
        "summary": summary
    }

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