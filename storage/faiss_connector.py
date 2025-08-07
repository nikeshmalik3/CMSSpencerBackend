import faiss
import numpy as np
import pickle
import os
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime  # FIX: Added missing import

from config.settings import config

logger = logging.getLogger(__name__)

class FAISSConnector:
    """FAISS vector database connector for semantic search"""
    
    def __init__(self):
        self.dimension = config.EMBEDDING_DIMENSION  # 1024 for mxbai-embed-large-v1
        self.index: Optional[faiss.Index] = None
        self.id_map: Dict[int, str] = {}  # Maps FAISS internal IDs to document IDs
        self.metadata: Dict[str, Dict[str, Any]] = {}  # Stores metadata for each vector
        self.index_path = Path(config.FAISS_INDEX_PATH)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize FAISS index"""
        try:
            # Create index directory if it doesn't exist
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            # Load existing index or create new one
            await self.load_index()
            
            logger.info(f"FAISS initialized with dimension {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise
    
    async def load_index(self):
        """Load existing index from disk or create new one"""
        index_file = self.index_path / "spencer_ai.index"
        metadata_file = self.index_path / "spencer_ai_metadata.pkl"
        
        if index_file.exists() and metadata_file.exists():
            try:
                # Load in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                
                self.index = await loop.run_in_executor(
                    self.executor,
                    faiss.read_index,
                    str(index_file)
                )
                
                with open(metadata_file, 'rb') as f:
                    saved_data = await loop.run_in_executor(
                        self.executor,
                        pickle.load,
                        f
                    )
                    self.id_map = saved_data['id_map']
                    self.metadata = saved_data['metadata']
                
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create new FAISS index"""
        # Using IndexIVFPQ for memory efficiency with large scale
        # First create a flat index for training
        quantizer = faiss.IndexFlatL2(self.dimension)
        
        # Create IVF index with PQ compression
        # 4096 clusters, 64 sub-vectors for compression
        self.index = faiss.IndexIVFPQ(
            quantizer, 
            self.dimension, 
            4096,  # n_list (number of clusters)
            64,    # n_subvectors
            8      # bits per code
        )
        
        logger.info("Created new FAISS index")
    
    async def save_index(self):
        """Save index to disk"""
        try:
            index_file = self.index_path / "spencer_ai.index"
            metadata_file = self.index_path / "spencer_ai_metadata.pkl"
            
            loop = asyncio.get_event_loop()
            
            # Save index
            await loop.run_in_executor(
                self.executor,
                faiss.write_index,
                self.index,
                str(index_file)
            )
            
            # Save metadata
            save_data = {
                'id_map': self.id_map,
                'metadata': self.metadata
            }
            
            with open(metadata_file, 'wb') as f:
                await loop.run_in_executor(
                    self.executor,
                    pickle.dump,
                    save_data,
                    f
                )
            
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    async def add_embeddings(
        self, 
        embeddings: List[List[float]], 
        ids: List[str], 
        metadata: List[Dict[str, Any]]
    ) -> bool:
        """Add embeddings to the index"""
        try:
            # Convert to numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Train index if needed (for IVF indexes)
            if not self.index.is_trained:
                logger.info("Training FAISS index...")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor,
                    self.index.train,
                    embeddings_array
                )
            
            # Get current index size
            start_id = self.index.ntotal
            
            # Add vectors
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self.index.add,
                embeddings_array
            )
            
            # Update mappings
            for i, (doc_id, meta) in enumerate(zip(ids, metadata)):
                internal_id = start_id + i
                self.id_map[internal_id] = doc_id
                self.metadata[doc_id] = meta
            
            # Save index periodically
            if self.index.ntotal % 1000 == 0:
                await self.save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            return False
    
    async def search(
        self, 
        query_embedding: List[float], 
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors"""
        try:
            # Convert to numpy array
            query_array = np.array([query_embedding]).astype('float32')
            
            # Search
            loop = asyncio.get_event_loop()
            distances, indices = await loop.run_in_executor(
                self.executor,
                self.index.search,
                query_array,
                k * 3  # Get more results for filtering
            )
            
            # Convert results and apply filters
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # No result
                    continue
                
                doc_id = self.id_map.get(int(idx))
                if not doc_id:
                    continue
                
                meta = self.metadata.get(doc_id, {})
                
                # Apply filters if provided
                if filters:
                    match = all(
                        meta.get(key) == value 
                        for key, value in filters.items()
                    )
                    if not match:
                        continue
                
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + float(dist))
                
                results.append((doc_id, similarity, meta))
                
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    async def update_embedding(
        self, 
        doc_id: str, 
        new_embedding: List[float],
        new_metadata: Dict[str, Any]
    ) -> bool:
        """Update an existing embedding"""
        try:
            # Find internal ID
            internal_id = None
            for int_id, d_id in self.id_map.items():
                if d_id == doc_id:
                    internal_id = int_id
                    break
            
            if internal_id is None:
                logger.warning(f"Document {doc_id} not found for update")
                return False
            
            # FAISS doesn't support in-place updates for IVF indexes
            # We need to implement a workaround by marking as deleted
            # and adding new version
            
            # Mark old as deleted in metadata
            if doc_id in self.metadata:
                self.metadata[doc_id]['deleted'] = True
            
            # Add new version
            await self.add_embeddings(
                [new_embedding],
                [doc_id + f"_v{int(datetime.utcnow().timestamp())}"],
                [new_metadata]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating embedding: {e}")
            return False
    
    async def delete_embedding(self, doc_id: str) -> bool:
        """Mark embedding as deleted"""
        try:
            if doc_id in self.metadata:
                self.metadata[doc_id]['deleted'] = True
                await self.save_index()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting embedding: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            active_count = sum(
                1 for meta in self.metadata.values() 
                if not meta.get('deleted', False)
            )
            
            return {
                "total_vectors": self.index.ntotal,
                "active_vectors": active_count,
                "deleted_vectors": self.index.ntotal - active_count,
                "dimension": self.dimension,
                "index_type": type(self.index).__name__,
                "is_trained": self.index.is_trained
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    async def rebuild_index(self):
        """Rebuild index to remove deleted vectors"""
        try:
            logger.info("Rebuilding FAISS index...")
            
            # Collect active vectors
            active_embeddings = []
            active_ids = []
            active_metadata = []
            
            # This would require storing embeddings separately
            # For now, log warning
            logger.warning("Index rebuild requires embedding storage implementation")
            
            # Save current index
            await self.save_index()
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            raise

# Singleton instance
faiss_connector = FAISSConnector()