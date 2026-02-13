"""
API Models
Pydantic models for request/response validation.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., description="User message to send to the agent")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What are the symptoms for patient PT-12345?"
            }
        }


class Citation(BaseModel):
    """Citation information for RAG results."""
    id: int = Field(..., description="Citation reference number")
    file_name: Optional[str]  = Field(..., description="Source document filename")
    section: Optional[str]  = Field(..., description="Document section")
    patient_id: Optional[str]  = Field(..., description="Associated patient ID")
    clinician_id: Optional[str] = Field(..., description="Associated clinician ID")
    score: Optional[float] = Field(None, description="Relevance score (0-1)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "file_name": "discharge_summary.pdf",
                "section": "Chief Complaint",
                "patient_id": "PT-12345",
                "clinician_id": "CL-9876",
                "score": 0.89
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str = Field(..., description="Agent's text response")
    tool_calls: List[str] = Field(default=[], description="List of tools used")
    citations: List[Citation] = Field(default=[], description="Source citations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Based on the discharge summary [1], the patient...",
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