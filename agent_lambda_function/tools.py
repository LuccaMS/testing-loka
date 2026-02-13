"""
Agent Tools
Contains all tools available to the medical AI agent.
"""
import logging
import asyncio
import json
from typing import Optional, List, Literal
import pandas as pd

from langchain_core.tools import tool
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger("rag-agent.tools")

# These will be set by main.py during initialization
_qdrant_client = None
_genai_client = None
_xgboost_model = None
_config = {}

def init_tools(qdrant_client, genai_client, xgboost_model, config):
    """Initialize tool dependencies (called from main.py)."""
    global _qdrant_client, _genai_client, _xgboost_model, _config
    _qdrant_client = qdrant_client
    _genai_client = genai_client
    _xgboost_model = xgboost_model
    _config = config
    logger.info("✓ Tools initialized with dependencies")


def generate_embedding_sync(text: str) -> List[float]:
    """Generate embedding using the native google-genai SDK."""
    from google.genai import types
    
    if not _genai_client:
        raise ValueError("GenAI Client not initialized")
    
    try:
        result = _genai_client.models.embed_content(
            model=_config['EMBEDDING_MODEL_ID'],
            contents=[text],
            config=types.EmbedContentConfig(output_dimensionality=_config['VECTOR_SIZE'])
        )
        if hasattr(result, 'embeddings') and result.embeddings:
            return result.embeddings[0].values
        return []
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise e


# ============================================================================
# SEARCH MEDICAL RECORDS TOOL
# ============================================================================

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
        query: The search query string (e.g., "symptoms of heart attack")
        patient_id: Optional patient ID filter (e.g., "PT-12345")
        clinician_id: Optional clinician ID filter (e.g., "CL-9876")
    
    Returns:
        JSON string with search results and citations
    """
    if not _qdrant_client:
        return json.dumps({"error": "Database connection not available.", "citations": []})

    logger.info(f"🔍 Searching: '{query}' | Patient: {patient_id}")

    try:
        # Generate Embedding
        query_vector = await asyncio.to_thread(generate_embedding_sync, query)

        if not query_vector:
            return json.dumps({"error": "Failed to generate embedding for query.", "citations": []})

        # Build Filters
        must_filters = []
        if patient_id:
            must_filters.append(FieldCondition(key="patient_id", match=MatchValue(value=patient_id)))
        if clinician_id:
            must_filters.append(FieldCondition(key="clinician_id", match=MatchValue(value=clinician_id)))

        search_filter = Filter(must=must_filters) if must_filters else None

        # Search Qdrant
        search_result = await _qdrant_client.query_points(
            collection_name=_config['COLLECTION_NAME'],
            query=query_vector,
            limit=5,
            query_filter=search_filter,
            with_payload=True
        )

        # Format Results with Citations
        if not search_result.points:
            return json.dumps({
                "content": "No relevant medical records found matching criteria.",
                "citations": []
            })

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
        
        result = {
            "content": "\n\n".join(formatted_content),
            "citations": citations
        }
        
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Search tool error: {str(e)}", exc_info=True)
        return json.dumps({
            "error": f"Error occurred during search: {str(e)}",
            "citations": []
        })


# ============================================================================
# PREDICT ALT TOOL
# ============================================================================

# Mapping dictionaries
MAPPINGS = {
    'sex': {'Female': 0, 'Male': 1},
    'smoker': {'No': 0, 'Yes': 1},
    'diagnosis_code': {'D1': 1.0, 'D2': 2.0, 'D3': 3.0, 'D4': 4.0, 'D5': 5.0},
    'exercise_frequency': {'Low': 0.0, 'Moderate': 1.0, 'High': 2.0},
    'diet_quality': {'Poor': 0.0, 'Average': 1.0, 'Good': 2.0},
    'income_bracket': {'Low': 0.0, 'Middle': 1.0, 'High': 2.0},
    'education_level': {'Primary': 0.0, 'Secondary': 1.0, 'Tertiary': 2.0},
    'urban': {'No': 0.0, 'Yes': 1.0},
    'readmitted': {'No': 0.0, 'Yes': 1.0}
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
) -> str:
    """
    Predicts the Alanine Aminotransferase (ALT) levels for a patient using an XGBoost model.
    Use this tool when users ask for predictions, forecasts, or expected liver enzyme values.
    
    Args:
        age: Patient age in years
        sex: Patient biological sex (Male/Female)
        bmi: Body Mass Index
        smoker: Smoking status (Yes/No)
        diagnosis_code: Primary diagnosis code (D1-D5)
        medication_count: Number of current medications
        days_hospitalized: Days spent in hospital
        readmitted: Whether patient was readmitted (Yes/No)
        last_lab_glucose: Most recent glucose reading
        exercise_frequency: Level of physical activity (Low/Moderate/High)
        diet_quality: Overall diet quality (Poor/Average/Good)
        income_bracket: Socioeconomic status (Low/Middle/High)
        education_level: Highest education completed (Primary/Secondary/Tertiary)
        urban: Urban vs rural residence (Yes/No)
        albumin_globulin_ratio: Lab test ratio
        
    Returns:
        Prediction result with ALT level in U/L
    """
    if _xgboost_model is None:
        return "Error: Prediction model is not currently loaded in the system."

    try:
        # Transform literals to numeric values
        data_dict = {
            'age': age,
            'sex': MAPPINGS['sex'].get(sex, 0),
            'bmi': bmi,
            'smoker': MAPPINGS['smoker'].get(smoker, 0),
            'diagnosis_code': MAPPINGS['diagnosis_code'].get(diagnosis_code, 5.0),
            'medication_count': medication_count,
            'days_hospitalized': days_hospitalized,
            'readmitted': MAPPINGS['readmitted'].get(readmitted, 0),
            'last_lab_glucose': last_lab_glucose,
            'exercise_frequency': MAPPINGS['exercise_frequency'].get(exercise_frequency, 1.0),
            'diet_quality': MAPPINGS['diet_quality'].get(diet_quality, 1.0),
            'income_bracket': MAPPINGS['income_bracket'].get(income_bracket, 1.0),
            'education_level': MAPPINGS['education_level'].get(education_level, 1.0),
            'urban': MAPPINGS['urban'].get(urban, 1.0),
            'albumin_globulin_ratio': albumin_globulin_ratio
        }

        # Convert to DataFrame with correct column order
        column_order = [
            'age', 'sex', 'bmi', 'smoker', 'diagnosis_code', 'medication_count',
            'days_hospitalized', 'readmitted', 'last_lab_glucose', 'exercise_frequency',
            'diet_quality', 'income_bracket', 'education_level', 'urban', 'albumin_globulin_ratio'
        ]
        df = pd.DataFrame([data_dict])[column_order]

        # Predict
        prediction = _xgboost_model.predict(df)[0]

        return (
            f"Based on the clinical parameters provided, the predicted Alanine Aminotransferase (ALT) "
            f"level is {float(prediction):.2f} U/L."
        )

    except Exception as e:
        logger.error(f"XGBoost Prediction Error: {e}", exc_info=True)
        return f"I encountered an error while trying to calculate the prediction: {str(e)}"