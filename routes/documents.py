"""
Document upload and processing routes with new file storage structure
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
import io

from services.file_storage_service import file_storage, FileType, FileAction
from agents.document_processor import document_processor
from storage import mongodb_connector
from models.api_models import ErrorResponse
from auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None),
    client_id: Optional[int] = Form(None),
    tags: Optional[List[str]] = Form(None),
    extract_text: bool = Form(True),
    current_user: Dict = Depends(get_current_user)
):
    """
    Upload and process a document using the new file storage structure
    Files are stored in: /users/{user_id}/{date}/uploads/{type}/{filename}
    """
    try:
        user_id = current_user.get("id", 0)
        
        # Read file content
        file_content = await file.read()
        
        # Determine file type
        file_extension = file.filename.split('.')[-1].lower()
        
        # Map to storage file type
        if file_extension in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'svg']:
            storage_file_type = FileType.IMAGE
        elif file_extension in ['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx']:
            storage_file_type = FileType.DOCUMENT
        elif file_extension in ['mp3', 'wav', 'm4a', 'ogg']:
            storage_file_type = FileType.AUDIO
        elif file_extension in ['mp4', 'avi', 'mov', 'wmv']:
            storage_file_type = FileType.VIDEO
        elif file_extension in ['json', 'xml', 'yaml', 'csv']:
            storage_file_type = FileType.DATA
        else:
            storage_file_type = FileType.DOCUMENT
        
        # Save to new file storage
        storage_result = file_storage.save_upload(
            user_id=user_id,
            file_content=file_content,
            filename=file.filename,
            file_type=storage_file_type,
            metadata={
                "project_id": project_id,
                "client_id": client_id,
                "tags": tags,
                "content_type": file.content_type,
                "upload_timestamp": datetime.now().isoformat()
            }
        )
        
        # Process document if requested
        processing_result = None
        if extract_text:
            # Create temporary file for document processor
            temp_path = f"/tmp/{storage_result['stored_name']}"
            with open(temp_path, 'wb') as f:
                f.write(file_content)
            
            # Process with document processor agent
            processing_request = {
                "file_path": temp_path,
                "context": {
                    "user_id": user_id,
                    "project_id": project_id,
                    "client_id": client_id
                },
                "parameters": {
                    "generate_embeddings": True,
                    "ocr_if_needed": True,
                    "extract_tables": True
                }
            }
            
            processing_result = await document_processor.process(processing_request)
            
            # Save processed data
            if processing_result and processing_result.get("data"):
                processed_info = file_storage.save_processed(
                    user_id=user_id,
                    original_filename=file.filename,
                    processed_content=processing_result["data"],
                    process_type="extracted",
                    output_format="json"
                )
                
                # Update storage result with processing info
                storage_result["processed_path"] = processed_info["storage_path"]
        
        # Save metadata to MongoDB
        document_record = {
            "user_id": user_id,
            "client_id": client_id,
            "project_id": project_id,
            "original_name": file.filename,
            "stored_name": storage_result["stored_name"],
            "storage_path": storage_result["storage_path"],
            "file_type": storage_file_type.value,
            "file_size": storage_result["file_size"],
            "mime_type": storage_result["mime_type"],
            "file_hash": storage_result["file_hash"],
            "upload_date": storage_result["upload_date"],
            "tags": tags,
            "processed": extract_text,
            "processed_data": processing_result["data"] if processing_result else None
        }
        
        doc_id = await mongodb_connector.save_document(document_record)
        
        return {
            "success": True,
            "document_id": str(doc_id),
            "filename": file.filename,
            "storage_path": storage_result["storage_path"],
            "file_size": storage_result["file_size"],
            "file_hash": storage_result["file_hash"],
            "processed": extract_text,
            "message": f"Document uploaded successfully to user storage: /users/{user_id}/{datetime.now().strftime('%Y-%m-%d')}/uploads/{storage_file_type.value}/"
        }
        
    except Exception as e:
        logger.error(f"Document upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}")
async def get_document(
    document_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get document details and metadata"""
    try:
        user_id = current_user.get("id", 0)
        
        # Get document from MongoDB
        doc = await mongodb_connector.get_document(document_id)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check user access
        if doc.get("user_id") != user_id and not current_user.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "document_id": str(doc["_id"]),
            "filename": doc.get("original_name"),
            "storage_path": doc.get("storage_path"),
            "file_type": doc.get("file_type"),
            "file_size": doc.get("file_size"),
            "upload_date": doc.get("upload_date"),
            "tags": doc.get("tags"),
            "processed": doc.get("processed"),
            "processed_data": doc.get("processed_data")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Download original document file"""
    try:
        user_id = current_user.get("id", 0)
        
        # Get document from MongoDB
        doc = await mongodb_connector.get_document(document_id)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check user access
        if doc.get("user_id") != user_id and not current_user.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Get file from storage
        file_content, metadata = file_storage.get_file(
            user_id=user_id,
            storage_path=doc.get("storage_path")
        )
        
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type=metadata.get("mime_type", "application/octet-stream"),
            headers={
                "Content-Disposition": f"attachment; filename={doc.get('original_name')}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def list_user_documents(
    date: Optional[str] = None,
    file_type: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: Dict = Depends(get_current_user)
):
    """List documents for current user"""
    try:
        user_id = current_user.get("id", 0)
        
        # Get files from storage
        files = file_storage.get_user_files(
            user_id=user_id,
            date=date,
            file_type=FileType(file_type) if file_type else None
        )
        
        # Apply additional filters
        if project_id:
            files = [f for f in files if f.get("custom_metadata", {}).get("project_id") == project_id]
        
        # Apply pagination
        total = len(files)
        files = files[offset:offset + limit]
        
        return {
            "files": files,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"List documents error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Delete a document"""
    try:
        user_id = current_user.get("id", 0)
        
        # Get document from MongoDB
        doc = await mongodb_connector.get_document(document_id)
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check user access
        if doc.get("user_id") != user_id and not current_user.get("is_admin"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete from file storage
        deleted = file_storage.delete_file(
            user_id=user_id,
            storage_path=doc.get("storage_path")
        )
        
        # Delete from MongoDB
        await mongodb_connector.delete_document(document_id)
        
        return {
            "success": True,
            "message": "Document deleted successfully",
            "document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/storage")
async def get_storage_stats(
    current_user: Dict = Depends(get_current_user)
):
    """Get storage statistics for current user"""
    try:
        user_id = current_user.get("id", 0)
        
        stats = file_storage.get_storage_stats(user_id=user_id)
        
        return {
            "user_id": user_id,
            "total_size_mb": stats["total_size_mb"],
            "file_count": stats["file_count"],
            "type_breakdown": stats["type_breakdown"],
            "storage_path": f"/users/{user_id}/"
        }
        
    except Exception as e:
        logger.error(f"Storage stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-report")
async def generate_report(
    report_type: str = Form(...),
    project_id: Optional[str] = Form(None),
    client_id: Optional[int] = Form(None),
    date_from: Optional[str] = Form(None),
    date_to: Optional[str] = Form(None),
    current_user: Dict = Depends(get_current_user)
):
    """Generate and save a report to user's output folder"""
    try:
        user_id = current_user.get("id", 0)
        
        # Generate report content (placeholder - integrate with analytics agent)
        report_content = {
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "user_id": user_id,
            "project_id": project_id,
            "client_id": client_id,
            "date_range": {
                "from": date_from,
                "to": date_to
            },
            "data": {
                # Report data would be generated here
                "placeholder": "Report data would be generated by analytics agent"
            }
        }
        
        # Save to output folder
        output_info = file_storage.save_output(
            user_id=user_id,
            content=report_content,
            output_type="reports",
            filename=f"{report_type}_report.json",
            client_id=client_id,
            project_id=project_id
        )
        
        return {
            "success": True,
            "report_path": output_info["storage_path"],
            "filename": output_info["filename"],
            "shared": output_info["shared"],
            "message": f"Report generated and saved to: {output_info['storage_path']}"
        }
        
    except Exception as e:
        logger.error(f"Generate report error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))