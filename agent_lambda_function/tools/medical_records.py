"""
Medical Records Search Tool
Provides vector database search functionality with citation tracking.
"""
import logging
from typing import Optional, List, Dict, Any
from langchain_core.tools import tool

logger = logging.getLogger("rag-agent.tools.medical_records")


def format_search_results(search_results: List[Any]) -> Dict[str, Any]:
    """
    Format search results with citations for the RAG tool.
    
    Args:
        search_results: List of search result points from Qdrant
        
    Returns:
        Dict containing formatted content and citations
    """
    if not search_results:
        return {
            "content": "No relevant medical records found matching criteria.",
            "citations": []
        }
    
    formatted_content = []
    citations = []
    
    for idx, hit in enumerate(search_results, start=1):
        payload = hit.payload
        
        # Build citation entry
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
    
    return {
        "content": "\n\n".join(formatted_content),
        "citations": citations
    }


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
        query: The search query string (e.g., "symptoms of heart attack").
        patient_id: Optional. The specific ID of the patient (e.g., "PT-12345").
        clinician_id: Optional. The ID of the doctor/clinician (e.g., "CL-9876").
        
    Returns:
        JSON string containing search results and citations
    """
    # Import here to avoid circular dependencies
    from tools.dependencies import get_qdrant_client, get_embedding_function
    
    qdrant_client = get_qdrant_client()
    generate_embedding = get_embedding_function()
    
    if not qdrant_client:
        return '{"error": "Database connection not available.", "citations": []}'
    
    logger.info(f"🔍 Searching: '{query}' | Patient: {patient_id} | Clinician: {clinician_id}")
    
    try:
        # Import Qdrant models
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        import asyncio
        import json
        from config import COLLECTION_NAME
        
        # 1. Generate Embedding
        query_vector = await asyncio.to_thread(generate_embedding, query)
        
        if not query_vector:
            return '{"error": "Failed to generate embedding for query.", "citations": []}'
        
        # 2. Build Filters
        must_filters = []
        if patient_id:
            must_filters.append(
                FieldCondition(key="patient_id", match=MatchValue(value=patient_id))
            )
        if clinician_id:
            must_filters.append(
                FieldCondition(key="clinician_id", match=MatchValue(value=clinician_id))
            )
        
        search_filter = Filter(must=must_filters) if must_filters else None
        
        # 3. Search Qdrant
        search_result = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5,
            query_filter=search_filter,
            with_payload=True,
            with_vectors=False
        )
        
        # 4. Format Results with Citations
        result = format_search_results(search_result.points)
        
        # Return as JSON string for the agent
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Search tool error: {str(e)}", exc_info=True)
        return json.dumps({
            "error": f"Error occurred during search: {str(e)}",
            "citations": []
        })