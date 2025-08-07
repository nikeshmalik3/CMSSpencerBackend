"""Semantic search routes"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from agents.semantic_search import semantic_search
from middleware.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])

@router.post("/", response_model=Dict[str, Any])
async def search(
    search_request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Perform semantic search across all indexed content"""
    try:
        # Add user context
        search_request["context"] = {
            "user_id": current_user["user_id"],
            "user_role": current_user["role"],
            "client_id": current_user.get("client_id")
        }
        
        # Set default parameters
        if "parameters" not in search_request:
            search_request["parameters"] = {}
        
        search_request["parameters"].setdefault("limit", 10)
        search_request["parameters"].setdefault("include_metadata", True)
        
        # Process search
        result = await semantic_search.process(search_request)
        
        return {
            "status": "success",
            "results": result["data"]["results"],
            "total_found": len(result["data"]["results"]),
            "confidence": result["data"]["confidence"],
            "search_time": result["metadata"].get("search_time", 0)
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed")

@router.post("/similar", response_model=Dict[str, Any])
async def find_similar(
    similarity_request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Find similar content to a given text or document"""
    try:
        # Build search request
        search_request = {
            "query": similarity_request.get("text", ""),
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"]
            },
            "parameters": {
                "mode": "similarity",
                "limit": similarity_request.get("limit", 5),
                "threshold": similarity_request.get("threshold", 0.7)
            }
        }
        
        if "document_id" in similarity_request:
            search_request["parameters"]["reference_document"] = similarity_request["document_id"]
        
        # Process search
        result = await semantic_search.process(search_request)
        
        return {
            "status": "success",
            "similar_items": result["data"]["results"],
            "count": len(result["data"]["results"])
        }
        
    except Exception as e:
        logger.error(f"Find similar error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Similarity search failed")

@router.post("/index", response_model=Dict[str, Any])
async def index_content(
    index_request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Index new content for semantic search"""
    try:
        # Validate request
        if "content" not in index_request:
            raise HTTPException(status_code=400, detail="Content is required")
        
        # Build index request
        request = {
            "action": "index",
            "content": index_request["content"],
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"]
            }
        }
        
        # Process indexing
        result = await semantic_search.process(request)
        
        return {
            "status": "success",
            "message": "Content indexed successfully",
            "content_id": result["data"].get("content_id")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Index content error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Indexing failed")

@router.delete("/index/{content_id}")
async def delete_indexed_content(
    content_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete indexed content"""
    try:
        # Build delete request
        request = {
            "action": "delete",
            "content_id": content_id,
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"]
            }
        }
        
        # Process deletion
        await semantic_search.process(request)
        
        return {"message": "Content deleted from index successfully"}
        
    except Exception as e:
        logger.error(f"Delete indexed content error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Deletion failed")

@router.get("/stats", response_model=Dict[str, Any])
async def get_search_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get search index statistics"""
    try:
        # Get stats from semantic search
        stats = await semantic_search.get_index_stats()
        
        # Filter by user if not admin
        if current_user["role"] not in ["admin", "God"]:
            stats = {
                "user_content_count": stats.get("user_stats", {}).get(
                    str(current_user["user_id"]), 0
                ),
                "total_searches": stats.get("total_searches", 0)
            }
        
        return {
            "status": "success",
            "stats": stats,
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get search stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get stats")

@router.post("/reindex", response_model=Dict[str, Any])
async def reindex_content(
    reindex_request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Reindex content (admin only)"""
    try:
        # Check permissions
        if current_user["role"] not in ["admin", "God"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Build reindex request
        request = {
            "action": "reindex",
            "scope": reindex_request.get("scope", "all"),
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"]
            }
        }
        
        # Process reindexing
        result = await semantic_search.process(request)
        
        return {
            "status": "success",
            "message": "Reindexing started",
            "details": result["data"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reindex error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Reindexing failed")

@router.post("/filters", response_model=Dict[str, Any])
async def search_with_filters(
    filtered_search: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Search with advanced filters"""
    try:
        # Build search request with filters
        search_request = {
            "query": filtered_search.get("query", ""),
            "filters": filtered_search.get("filters", {}),
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"]
            },
            "parameters": {
                "limit": filtered_search.get("limit", 10),
                "offset": filtered_search.get("offset", 0),
                "sort_by": filtered_search.get("sort_by", "relevance")
            }
        }
        
        # Add user filter for non-admin users
        if current_user["role"] not in ["admin", "God"]:
            search_request["filters"]["user_id"] = current_user["user_id"]
        
        # Process search
        result = await semantic_search.process(search_request)
        
        return {
            "status": "success",
            "results": result["data"]["results"],
            "total_found": len(result["data"]["results"]),
            "filters_applied": search_request["filters"]
        }
        
    except Exception as e:
        logger.error(f"Filtered search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Filtered search failed")