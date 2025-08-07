import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime
import asyncio

from agents.base_agent import BaseAgent
from storage import faiss_connector, mongodb_connector
from config.settings import config

logger = logging.getLogger(__name__)

class SemanticSearchAgent(BaseAgent):
    """
    The Memory of the system - finds relevant information through meaning, not just keywords
    """
    
    def __init__(self):
        super().__init__(
            name="semantic_search",
            description="Performs semantic search and manages knowledge base"
        )
        self.embedding_model = None
        self._init_embedding_model()
        
    def _init_embedding_model(self):
        """Initialize embedding model (mxbai-embed-large-v1)"""
        try:
            # In production, this would load the actual model
            # For now, we'll use a placeholder
            logger.info("Initializing mxbai-embed-large-v1 embedding model")
            # self.embedding_model = load_model("mxbai-embed-large-v1")
            self.embedding_model = "placeholder"  # Replace with actual model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
    
    async def validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate semantic search request"""
        if request.get("parameters", {}).get("mode") == "index":
            return "documents" in request or "content" in request
        return "query" in request
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process semantic search or indexing request
        """
        try:
            parameters = request.get("parameters", {})
            mode = parameters.get("mode", "search")
            
            if mode == "index":
                # Index new content
                return await self._index_content(request)
            else:
                # Perform search
                return await self._search_content(request)
                
        except Exception as e:
            logger.error(f"Semantic search error: {e}", exc_info=True)
            raise
    
    async def _search_content(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for semantically similar content
        """
        query = request.get("query", "")
        context = request.get("context", {})
        parameters = request.get("parameters", {})
        search_type = parameters.get("search_type", "all")
        
        # Generate embedding for query
        query_embedding = await self._generate_embedding(query)
        
        # Prepare filters based on search type
        filters = {}
        if search_type == "queries":
            filters["type"] = "query_pattern"
        elif search_type == "documents":
            filters["type"] = "document"
        elif search_type == "knowledge":
            filters["type"] = "knowledge"
        
        # Add user context filters
        if context.get("project_id"):
            filters["project_id"] = context["project_id"]
        
        # Search using YOUR VECTOR SERVICE
        search_results = await self._search_vector_service(
            query,
            k=parameters.get("limit", 10),
            filters=filters
        )
        
        # Enhance results with stored data
        enhanced_results = await self._enhance_search_results(
            search_results,
            query,
            search_type
        )
        
        # Learn from this query for future
        if parameters.get("learn", True):
            await self._learn_from_query(query, context, enhanced_results)
        
        return {
            "data": {
                "results": enhanced_results,
                "count": len(enhanced_results),
                "search_type": search_type,
                "message": self._format_search_message(enhanced_results, search_type)
            },
            "metadata": {
                "embedding_dimension": len(query_embedding),
                "filters_applied": filters
            }
        }
    
    async def _index_content(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Index new content for semantic search
        """
        documents = request.get("documents", [])
        content = request.get("content")
        context = request.get("context", {})
        
        indexed_count = 0
        
        # Handle single content
        if content:
            documents = [{
                "id": content.get("id", f"doc_{datetime.utcnow().timestamp()}"),
                "content": content.get("text", ""),
                "metadata": content.get("metadata", {})
            }]
        
        # Process documents in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            embeddings = []
            ids = []
            metadata_list = []
            
            for doc in batch:
                # Generate embedding
                embedding = await self._generate_embedding(doc["content"])
                
                # Prepare metadata
                metadata = {
                    "type": doc.get("type", "document"),
                    "user_id": context.get("user_id"),
                    "project_id": context.get("project_id"),
                    "created_at": datetime.utcnow().isoformat(),
                    **doc.get("metadata", {})
                }
                
                embeddings.append(embedding)
                ids.append(doc["id"])
                metadata_list.append(metadata)
            
            # Add to FAISS
            success = await faiss_connector.add_embeddings(
                embeddings,
                ids,
                metadata_list
            )
            
            if success:
                indexed_count += len(batch)
        
        # Save index
        await faiss_connector.save_index()
        
        return {
            "data": {
                "indexed_count": indexed_count,
                "message": f"Successfully indexed {indexed_count} documents"
            }
        }
    
    async def _search_vector_service(self, query: str, k: int = 10, filters: Dict = None) -> List[Tuple[str, float, Dict]]:
        """
        Search using YOUR VECTOR SERVICE at port 8001
        """
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                payload = {
                    "query": query,
                    "top_k": k,
                    "min_similarity": 0.001
                }
                
                # Add filters if provided
                if filters:
                    payload["filters"] = filters
                
                async with session.post(
                    "http://157.180.62.92:8001/api/v1/search",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success'):
                            # Convert to expected format
                            results = []
                            for item in data.get('results', []):
                                doc_id = item.get('metadata', {}).get('id', 'unknown')
                                similarity = item.get('similarity', 0.0)
                                metadata = item.get('metadata', {})
                                results.append((doc_id, similarity, metadata))
                            return results
                        else:
                            logger.error(f"Vector search error: {data.get('error')}")
                            return []
                    else:
                        logger.error(f"Vector service returned status {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Failed to search vector service: {e}")
            return []
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using YOUR VECTOR SERVICE
        """
        try:
            # Use YOUR vector service at port 8001 with mxbai-embed-large
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://157.180.62.92:8001/api/v1/embed",
                    json={"text": text}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('success'):
                            # REAL EMBEDDINGS FROM YOUR SERVICE!
                            return data['embedding']
                        else:
                            logger.error(f"Vector service error: {data.get('error')}")
                            return [0.0] * config.EMBEDDING_DIMENSION
                    else:
                        logger.error(f"Vector service returned status {response.status}")
                        return [0.0] * config.EMBEDDING_DIMENSION
        except Exception as e:
            logger.error(f"Failed to connect to vector service: {e}")
            # Return zero vector as fallback
            return [0.0] * config.EMBEDDING_DIMENSION
    
    async def _enhance_search_results(
        self,
        search_results: List[Tuple[str, float, Dict[str, Any]]],
        query: str,
        search_type: str
    ) -> List[Dict[str, Any]]:
        """
        Enhance search results with additional data
        """
        enhanced = []
        
        for doc_id, similarity, metadata in search_results:
            result = {
                "id": doc_id,
                "similarity": similarity,
                "metadata": metadata
            }
            
            # Add type-specific enhancements
            if search_type == "queries" and metadata.get("type") == "query_pattern":
                # Add successful API sequence
                result["api_sequence"] = metadata.get("api_sequence", [])
                result["success_rate"] = metadata.get("success_rate", 0)
                result["original_query"] = metadata.get("query", "")
                
            elif search_type == "documents":
                # Add document details
                doc = await mongodb_connector.get_document(doc_id)
                if doc:
                    result["filename"] = doc.get("original_name", "")
                    result["document_type"] = doc.get("document_type", "")
                    result["summary"] = metadata.get("summary", "")
                    
            elif search_type == "knowledge":
                # Add knowledge context
                result["category"] = metadata.get("category", "")
                result["source"] = metadata.get("source", "")
                result["confidence"] = metadata.get("confidence", 0)
            
            enhanced.append(result)
        
        return enhanced
    
    async def _learn_from_query(
        self,
        query: str,
        context: Dict[str, Any],
        results: List[Dict[str, Any]]
    ):
        """
        Learn from successful queries to improve future searches
        """
        try:
            # Only learn if we got good results
            if not results or results[0]["similarity"] < 0.7:
                return
            
            # Create query pattern entry
            query_pattern = {
                "id": f"qp_{datetime.utcnow().timestamp()}",
                "content": query,
                "metadata": {
                    "type": "query_pattern",
                    "user_id": context.get("user_id"),
                    "intent": context.get("intent"),
                    "timestamp": datetime.utcnow().isoformat(),
                    "result_count": len(results),
                    "top_similarity": results[0]["similarity"]
                }
            }
            
            # Index the successful query pattern
            await self._index_content({
                "documents": [query_pattern],
                "context": context
            })
            
        except Exception as e:
            logger.error(f"Error learning from query: {e}")
    
    def _format_search_message(
        self,
        results: List[Dict[str, Any]],
        search_type: str
    ) -> str:
        """Format search results into a message"""
        if not results:
            return "No relevant results found."
        
        count = len(results)
        
        if search_type == "queries":
            return f"Found {count} similar query patterns that might help."
        elif search_type == "documents":
            return f"Found {count} relevant documents."
        elif search_type == "knowledge":
            return f"Found {count} relevant knowledge entries."
        else:
            return f"Found {count} relevant results."
    
    async def update_knowledge(
        self,
        doc_id: str,
        new_content: str,
        new_metadata: Dict[str, Any]
    ) -> bool:
        """Update existing knowledge entry"""
        try:
            # Generate new embedding
            new_embedding = await self._generate_embedding(new_content)
            
            # Update in FAISS
            success = await faiss_connector.update_embedding(
                doc_id,
                new_embedding,
                new_metadata
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating knowledge: {e}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the semantic index"""
        try:
            stats = await faiss_connector.get_stats()
            
            # Add domain-specific stats from actual data
            try:
                # Query actual counts from vector database
                if hasattr(faiss_connector, 'metadata_store'):
                    metadata = faiss_connector.metadata_store
                    
                    # Count actual categories from metadata
                    api_count = sum(1 for m in metadata.values() if m.get('type') == 'api_endpoint')
                    query_count = sum(1 for m in metadata.values() if m.get('type') == 'query_pattern')
                    doc_count = sum(1 for m in metadata.values() if m.get('type') == 'document')
                    term_count = sum(1 for m in metadata.values() if m.get('type') == 'construction_term')
                    
                    stats["knowledge_categories"] = {
                        "api_endpoints": api_count,
                        "query_patterns": query_count,
                        "documents": doc_count,
                        "construction_terms": term_count
                    }
                else:
                    # Return actual zero counts if no metadata available
                    stats["knowledge_categories"] = {
                        "api_endpoints": 0,
                        "query_patterns": 0,
                        "documents": 0,
                        "construction_terms": 0
                    }
            except:
                # Return zeros on any error
                stats["knowledge_categories"] = {
                    "api_endpoints": 0,
                    "query_patterns": 0,
                    "documents": 0,
                    "construction_terms": 0
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}

# Create singleton instance
semantic_search = SemanticSearchAgent()