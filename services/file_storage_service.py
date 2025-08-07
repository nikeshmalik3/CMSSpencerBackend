"""
File Storage Service
Handles all file operations using Spencer File Manager API
Structure: Documents organized by type (pdfs, excel, word, autocad, images)
Uses remote file storage at http://157.180.62.92:8002
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import json
import mimetypes
from enum import Enum
import aiohttp
import base64
import logging

logger = logging.getLogger(__name__)

class FileAction(Enum):
    """File action types for organizing storage"""
    UPLOAD = "uploads"
    PROCESSED = "processed"
    OUTPUT = "outputs"

class FileType(Enum):
    """File types for categorization"""
    DOCUMENT = "documents"
    IMAGE = "images"
    AUDIO = "audio"
    VIDEO = "video"
    DATA = "data"
    REPORT = "reports"
    EXPORT = "exports"
    DOWNLOAD = "downloads"

class FileStorageService:
    """
    Service for managing file storage using Spencer File Manager API
    Files are stored on VPS at http://157.180.62.92:8002
    """
    
    def __init__(self):
        """
        Initialize file storage service with Spencer File Manager API
        """
        self.api_url = "http://157.180.62.92:8002"
        self.session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
    
    def _get_category_subcategory(self, file_type: FileType, extension: str) -> Tuple[str, str]:
        """
        Map file type to Spencer storage category and subcategory
        
        Args:
            file_type: Type of file
            extension: File extension
        
        Returns:
            Tuple of (category, subcategory)
        """
        # Map to Spencer File Manager structure
        mapping = {
            FileType.DOCUMENT: {
                ".pdf": ("documents", "pdfs"),
                ".xlsx": ("documents", "excel"),
                ".xls": ("documents", "excel"),
                ".docx": ("documents", "word"),
                ".doc": ("documents", "word"),
                ".dwg": ("documents", "autocad"),
                ".dxf": ("documents", "autocad"),
                "default": ("documents", "pdfs")
            },
            FileType.IMAGE: ("documents", "images"),
            FileType.AUDIO: ("audio", "transcriptions"),
            FileType.DATA: ("uploads", "processed"),
            FileType.REPORT: ("projects", "construction"),
            FileType.EXPORT: ("uploads", "processed")
        }
        
        if file_type == FileType.DOCUMENT and extension in mapping[FileType.DOCUMENT]:
            return mapping[FileType.DOCUMENT][extension]
        elif file_type == FileType.DOCUMENT:
            return mapping[FileType.DOCUMENT]["default"]
        elif file_type in mapping:
            return mapping[file_type]
        else:
            return ("uploads", "pending")
    
    def _detect_file_type(self, filename: str) -> FileType:
        """Detect file type based on extension"""
        ext = Path(filename).suffix.lower()
        
        document_exts = ['.pdf', '.docx', '.doc', '.xlsx', '.xls', '.dwg', '.dxf']
        image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
        audio_exts = ['.mp3', '.wav', '.m4a', '.ogg']
        video_exts = ['.mp4', '.avi', '.mov', '.mkv']
        data_exts = ['.csv', '.json', '.xml', '.txt']
        
        if ext in document_exts:
            return FileType.DOCUMENT
        elif ext in image_exts:
            return FileType.IMAGE
        elif ext in audio_exts:
            return FileType.AUDIO
        elif ext in video_exts:
            return FileType.VIDEO
        elif ext in data_exts:
            return FileType.DATA
        else:
            return FileType.DOCUMENT
    
    async def save_to_spencer(self,
                             file_content: bytes,
                             filename: str,
                             file_type: Optional[FileType] = None,
                             user_id: Optional[int] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Save file to Spencer File Manager API
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            file_type: Type of file (auto-detected if not provided)
            user_id: Optional user ID for metadata
            metadata: Optional metadata to store with file
        
        Returns:
            Dictionary with file information and storage path
        """
        await self._ensure_session()
        
        try:
            # Auto-detect file type if not provided
            if file_type is None:
                file_type = self._detect_file_type(filename)
            
            # Get extension
            extension = Path(filename).suffix.lower()
            
            # Get Spencer storage category and subcategory
            category, subcategory = self._get_category_subcategory(file_type, extension)
            
            # Encode content as base64
            content_b64 = base64.b64encode(file_content).decode('utf-8')
            
            # Prepare payload
            payload = {
                "filename": filename,
                "content": content_b64,
                "category": category,
                "subcategory": subcategory
            }
            
            # Add metadata if provided
            if metadata:
                payload["metadata"] = metadata
            
            # Send to Spencer File Manager API
            async with self.session.post(
                f"{self.api_url}/api/v1/files/save",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        logger.info(f"✅ Saved to Spencer storage: {filename}")
                        
                        # Add additional info
                        result["filebrowser_url"] = f"http://157.180.62.92:8090/files/{result['path']}"
                        result["user_id"] = user_id
                        result["file_type"] = file_type.value
                        
                        return result
                    else:
                        logger.error(f"Spencer API error: {result.get('error')}")
                        raise Exception(f"Failed to save file: {result.get('error')}")
                else:
                    error_text = await response.text()
                    logger.error(f"Spencer API HTTP error {response.status}: {error_text}")
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Failed to save to Spencer storage: {e}")
            raise
    
    async def save_upload(self, 
                         user_id: int,
                         file_content: bytes,
                         filename: str,
                         file_type: Optional[FileType] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Save uploaded file to Spencer File Manager API
        
        Args:
            user_id: User ID
            file_content: File content as bytes
            filename: Original filename
            file_type: Type of file (auto-detected if not provided)
            metadata: Optional metadata to store with file
        
        Returns:
            Dictionary with file information and storage path
        """
        # Add user info to metadata
        if metadata is None:
            metadata = {}
        metadata["user_id"] = user_id
        metadata["upload_date"] = datetime.now().isoformat()
        metadata["action"] = "upload"
        
        return await self.save_to_spencer(
            file_content=file_content,
            filename=filename,
            file_type=file_type,
            user_id=user_id,
            metadata=metadata
        )
    
    async def save_processed(self,
                            user_id: int,
                            original_filename: str,
                            processed_content: Any,
                            process_type: str,
                            output_format: str = "json") -> Dict[str, Any]:
        """
        Save processed file data to Spencer File Manager
        
        Args:
            user_id: User ID
            original_filename: Original filename that was processed
            processed_content: Processed data
            process_type: Type of processing (ocr, extracted, analysis, transcript)
            output_format: Format to save (json, txt, csv)
        
        Returns:
            Dictionary with processed file information
        """
        # Generate filename
        base_name = Path(original_filename).stem
        processed_filename = f"{base_name}_{process_type}.{output_format}"
        
        # Convert content to bytes
        if output_format == "json":
            content_bytes = json.dumps(processed_content, indent=2).encode('utf-8')
        else:
            content_bytes = str(processed_content).encode('utf-8')
        
        # Metadata
        metadata = {
            "user_id": user_id,
            "original_file": original_filename,
            "process_type": process_type,
            "processed_date": datetime.now().isoformat(),
            "action": "processed"
        }
        
        # Determine file type based on output
        if process_type == "transcript":
            file_type = FileType.AUDIO
        elif output_format in ["csv", "json", "txt"]:
            file_type = FileType.DATA
        else:
            file_type = FileType.DOCUMENT
        
        return await self.save_to_spencer(
            file_content=content_bytes,
            filename=processed_filename,
            file_type=file_type,
            user_id=user_id,
            metadata=metadata
        )
    
    async def save_output(self,
                         user_id: int,
                         content: Any,
                         output_type: str,
                         filename: str,
                         client_id: Optional[int] = None,
                         project_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Save generated output file to Spencer File Manager
        
        Args:
            user_id: User ID
            content: Output content
            output_type: Type of output (report, export, download)
            filename: Output filename
            client_id: Optional client ID for shared outputs
            project_id: Optional project ID for shared outputs
        
        Returns:
            Dictionary with output file information
        """
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(filename).stem
        extension = Path(filename).suffix
        final_filename = f"{base_name}_{timestamp}{extension}"
        
        # Convert content to bytes
        if isinstance(content, (dict, list)):
            content_bytes = json.dumps(content, indent=2).encode('utf-8')
        elif isinstance(content, bytes):
            content_bytes = content
        else:
            content_bytes = str(content).encode('utf-8')
        
        # Metadata
        metadata = {
            "user_id": user_id,
            "output_type": output_type,
            "creation_date": datetime.now().isoformat(),
            "action": "output"
        }
        
        if client_id:
            metadata["client_id"] = client_id
        if project_id:
            metadata["project_id"] = project_id
        
        # Determine file type
        if output_type == "report":
            file_type = FileType.REPORT
        elif output_type == "export":
            file_type = FileType.EXPORT
        elif output_type == "download":
            file_type = FileType.DOWNLOAD
        else:
            file_type = FileType.DATA
        
        return await self.save_to_spencer(
            file_content=content_bytes,
            filename=final_filename,
            file_type=file_type,
            user_id=user_id,
            metadata=metadata
        )
    
    async def list_files(self, path: str = "") -> Dict[str, Any]:
        """
        List files in Spencer storage
        
        Args:
            path: Path to list (e.g., "documents/pdfs")
        
        Returns:
            Dictionary with file listing
        """
        await self._ensure_session()
        
        try:
            params = {"path": path} if path else {}
            
            async with self.session.get(
                f"{self.api_url}/api/v1/files",
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to list files: HTTP {response.status}")
                    return {"success": False, "error": f"HTTP {response.status}"}
                    
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_health(self) -> Dict[str, Any]:
        """Check Spencer File Manager health"""
        await self._ensure_session()
        
        try:
            async with self.session.get(
                f"{self.api_url}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status}"}
                    
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

# Global instance
file_storage = FileStorageService()