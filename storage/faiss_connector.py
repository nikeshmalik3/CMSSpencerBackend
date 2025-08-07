# FAISS is replaced by Vector Service at port 8001
# This is a stub implementation that uses Vector Service instead

import numpy as np
import os
import logging
from typing import List, Tuple, Dict, Any, Optional
import aiohttp
from config.settings import config

logger = logging.getLogger(__name__)

class FAISSConnector:
    """Vector database connector using VPS Vector Service instead of FAISS"""
    
    def __init__(self):
        self.vector_service_url = config.VECTOR_SERVICE_URL
        self.dimension = 1024  # mxbai-embed-large dimension
        self.session = None
        
    async def initialize(self):
        """Initialize connection to Vector Service"""
        try:
            self.session = aiohttp.ClientSession()
            logger.info(f"Vector Service connector initialized at {self.vector_service_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Vector Service: {e}")
            # Don't raise - allow app to start without vector search
    
    async def load_index(self):
        """Compatibility method - Vector Service handles this"""
        pass
    
    async def save_index(self):
        """Compatibility method - Vector Service handles this"""
        pass
    
    async def add_embeddings(
        self, 
        embeddings: List[List[float]], 
        ids: List[str], 
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """Add embeddings to Vector Service"""
        try:
            if not self.session:
                return False
                
            async with self.session.post(
                f"{self.vector_service_url}/add",
                json={
                    "embeddings": embeddings,
                    "ids": ids,
                    "metadata": metadata
                }
            ) as response:
                return response.status == 200
        except:
            return False
    
    async def search(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search using Vector Service"""
        try:
            if not self.session:
                return []
                
            async with self.session.post(
                f"{self.vector_service_url}/search",
                json={
                    "embedding": query_embedding,
                    "k": k,
                    "filters": filters
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("results", [])
                return []
        except:
            return []
    
    async def update_embedding(
        self, 
        doc_id: str, 
        new_embedding: List[float],
        new_metadata: Dict[str, Any]
    ) -> bool:
        """Update embedding in Vector Service"""
        return await self.add_embeddings([new_embedding], [doc_id], [new_metadata])
    
    async def delete_embedding(self, doc_id: str) -> bool:
        """Delete from Vector Service"""
        try:
            if not self.session:
                return False
                
            async with self.session.delete(
                f"{self.vector_service_url}/delete/{doc_id}"
            ) as response:
                return response.status == 200
        except:
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get Vector Service stats"""
        try:
            if not self.session:
                return {"status": "disconnected"}
                
            async with self.session.get(
                f"{self.vector_service_url}/stats"
            ) as response:
                if response.status == 200:
                    return await response.json()
                return {"status": "error"}
        except:
            return {"status": "unavailable"}
    
    async def rebuild_index(self):
        """Vector Service handles this"""
        pass

# Singleton instance
faiss_connector = FAISSConnector()