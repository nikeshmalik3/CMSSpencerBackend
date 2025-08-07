"""Document processing routes"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from typing import Dict, Any, Optional, List
import os
import uuid
from datetime import datetime
import logging

from agents.document_processor import document_processor
from storage import mongodb_connector
from middleware.auth import get_current_user
from config.settings import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Upload directory
UPLOAD_DIR = config.UPLOAD_DIR or "/tmp/spencer_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", response_model=Dict[str, Any])
async def upload_document(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    extract_tables: bool = Form(True),
    extract_images: bool = Form(True),
    generate_embeddings: bool = Form(True),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Upload and process a document"""
    try:
        # Validate file type
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in document_processor.supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}"
            )
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_ext}")
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document
        process_request = {
            "file_path": file_path,
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"],
                "project_id": project_id
            },
            "parameters": {
                "extract_tables": extract_tables,
                "extract_images": extract_images,
                "generate_embeddings": generate_embeddings
            }
        }
        
        result = await document_processor.process(process_request)
        
        return {
            "message": "Document uploaded and processed successfully",
            "document": result["data"],
            "processing_time": result["metadata"]["processing_time"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload error: {e}", exc_info=True)
        # Clean up file if processing failed
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="Document processing failed")

@router.get("/{document_id}", response_model=Dict[str, Any])
async def get_document(
    document_id: str,
    include_content: bool = False,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get document details"""
    try:
        document = await mongodb_connector.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check access permissions
        if (document["user_id"] != current_user["user_id"] and 
            current_user["role"] not in ["admin", "God"]):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Remove sensitive data if not including content
        if not include_content:
            document.pop("processed_data", None)
        
        return {
            "document": document
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve document")

@router.get("/", response_model=Dict[str, Any])
async def list_documents(
    project_id: Optional[str] = None,
    document_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List user's documents"""
    try:
        filters = {"user_id": current_user["user_id"]}
        
        if project_id:
            filters["project_id"] = project_id
        
        if document_type:
            filters["document_type"] = document_type
        
        documents = await mongodb_connector.find_documents(
            filters=filters,
            limit=limit,
            skip=offset
        )
        
        # Remove processed data from list view
        for doc in documents:
            doc.pop("processed_data", None)
        
        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"List documents error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list documents")

@router.post("/{document_id}/extract", response_model=Dict[str, Any])
async def extract_from_document(
    document_id: str,
    extraction_request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Extract specific information from a document"""
    try:
        # Get document
        document = await mongodb_connector.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check access
        if (document["user_id"] != current_user["user_id"] and 
            current_user["role"] not in ["admin", "God"]):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Use semantic search to find relevant content
        from agents.semantic_search import semantic_search
        
        search_request = {
            "query": extraction_request.get("query", ""),
            "filters": {"document_id": document_id},
            "context": {"user_id": current_user["user_id"]},
            "parameters": {"limit": 5}
        }
        
        search_result = await semantic_search.process(search_request)
        
        return {
            "document_id": document_id,
            "extraction_query": extraction_request.get("query"),
            "results": search_result["data"]["results"],
            "confidence": search_result["data"]["confidence"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Extraction failed")

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a document"""
    try:
        # Get document to verify ownership
        document = await mongodb_connector.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check ownership
        if (document["user_id"] != current_user["user_id"] and 
            current_user["role"] != "God"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete file if it exists
        if os.path.exists(document["file_path"]):
            os.remove(document["file_path"])
        
        # Delete from database
        await mongodb_connector.delete_document(document_id)
        
        # Remove from vector store
        from agents.semantic_search import semantic_search
        await semantic_search._remove_document_embeddings(document_id)
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete document")

@router.post("/{document_id}/reprocess", response_model=Dict[str, Any])
async def reprocess_document(
    document_id: str,
    parameters: Dict[str, Any] = {},
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Reprocess an existing document with new parameters"""
    try:
        # Get document
        document = await mongodb_connector.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check access
        if (document["user_id"] != current_user["user_id"] and 
            current_user["role"] not in ["admin", "God"]):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Reprocess
        process_request = {
            "document_id": document_id,
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"]
            },
            "parameters": parameters
        }
        
        result = await document_processor.process(process_request)
        
        return {
            "message": "Document reprocessed successfully",
            "document": result["data"],
            "processing_time": result["metadata"]["processing_time"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document reprocess error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Reprocessing failed")