import re
import hashlib
from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter

def clean_value(value: str) -> str:
    """
    Removes markdown syntax (bold, stars) from extracted metadata.
    Allows underscores to support IDs like 'doc_123'.
    """
    if not value:
        return "Unknown"
    # Remove stars, hashes, and colons. Keep underscores.
    cleaned = re.sub(r"[\*#:]", "", value).strip()
    return cleaned if cleaned else "Unknown"

def compute_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def extract_metadata(content: str) -> Dict[str, str]:
    """
    Extracts structured metadata using robust regex patterns.
    Handles headers with Colons/Hyphens and fields inside lists.
    """
    metadata = {
        "document_id": "Unknown",
        "patient_id": "Unknown",
        "clinician_id": "Unknown",
        "date_created": "Unknown"
    }
    
    # 1. DOCUMENT ID
    # Look for "**Document ID:** 123"
    doc_id_match = re.search(r"\*\*Document ID:?\*\*\s*([^\n]+)", content, re.IGNORECASE)
    if doc_id_match:
        metadata["document_id"] = clean_value(doc_id_match.group(1))

    # Fallback: Header 1 "Report - 123" or "Report: 123"
    if metadata["document_id"] == "Unknown":
        h1_match = re.search(r"^#\s*[^:\-\n]+[:\-]\s*([^\n]+)", content, re.MULTILINE)
        if h1_match:
            metadata["document_id"] = clean_value(h1_match.group(1))

    # 2. OTHER FIELDS (Patient, Clinician, Date)
    # Matches "**Key:** Value" inside lists or standalone lines
    fields = {
        "patient_id": r"\*\*Patient ID:?\*\*\s*([^\n]+)",
        "clinician_id": r"\*\*Clinician ID:?\*\*\s*([^\n]+)",
        "date_created": r"\*\*Date Created:?\*\*\s*([^\n]+)"
    }
    
    for key, pattern in fields.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            metadata[key] = clean_value(match.group(1))

    return metadata

def process_markdown_content(content: str, filename: str) -> Dict[str, Any]:
    """
    Splits markdown into chunks and attaches metadata/hashes.
    """
    file_hash = compute_sha256(content)
    global_meta = extract_metadata(content)

    headers_to_split_on = [("#", "Title"), ("##", "Section")]
    
    try:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        splits = splitter.split_text(content)
    except Exception:
        # Fallback if splitting fails
        return {
            "file_hash": file_hash,
            "document_id": global_meta["document_id"],
            "chunks": [{
                "content": content[:2000],
                "chunk_index": 0,
                "metadata": global_meta,
                "file_hash": file_hash
            }]
        }

    chunks = []
    for i, split in enumerate(splits):
        chunk_meta = global_meta.copy()
        chunk_meta.update(split.metadata)
        
        chunks.append({
            "content": split.page_content,
            "chunk_index": i,
            "metadata": chunk_meta,
            "file_hash": file_hash
        })

    return {
        "file_hash": file_hash,
        "document_id": global_meta["document_id"],
        "chunks": chunks
    }