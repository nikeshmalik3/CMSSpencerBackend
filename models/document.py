from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    EXCEL = "excel"
    WORD = "word"
    AUTOCAD = "autocad"
    IMAGE = "image"
    TEXT = "text"
    UNKNOWN = "unknown"

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentMetadata(BaseModel):
    filename: str
    size_bytes: int
    mime_type: str
    pages: Optional[int] = None
    sheets: Optional[int] = None  # For Excel
    layers: Optional[List[str]] = None  # For AutoCAD
    dimensions: Optional[Dict[str, float]] = None  # For images/CAD
    
class ProcessedData(BaseModel):
    text_content: Optional[str] = None
    tables: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[str]] = None  # Base64 encoded
    structured_data: Optional[Dict[str, Any]] = None
    embeddings_generated: bool = False
    embedding_ids: List[str] = Field(default_factory=list)

class Document(BaseModel):
    id: str = Field(description="MongoDB ObjectId as string")
    user_id: int
    project_id: Optional[int] = None
    file_path: str
    original_name: str
    document_type: DocumentType
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: DocumentMetadata
    processed_data: Optional[ProcessedData] = None
    processing_errors: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": 3392,
                "file_path": "/storage/documents/blueprint_123.pdf",
                "original_name": "Site Blueprint Rev3.pdf",
                "document_type": "pdf",
                "status": "completed",
                "metadata": {
                    "filename": "Site Blueprint Rev3.pdf",
                    "size_bytes": 2548760,
                    "mime_type": "application/pdf",
                    "pages": 12
                },
                "tags": ["blueprint", "site-plan", "revision-3"]
            }
        }

class DocumentProcessingRequest(BaseModel):
    file_path: str
    document_type: Optional[DocumentType] = None
    project_id: Optional[int] = None
    extract_tables: bool = True
    extract_images: bool = True
    generate_embeddings: bool = True
    ocr_if_needed: bool = True