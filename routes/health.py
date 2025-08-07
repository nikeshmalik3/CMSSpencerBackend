"""Health check and system status routes"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from datetime import datetime
import psutil
import logging

from storage import mongodb_connector, redis_connector, faiss_connector
from agents import AGENTS
from middleware.auth import get_current_user
from config.settings import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])

@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Spencer AI Backend",
        "version": "1.0.0"
    }

@router.get("/ready", response_model=Dict[str, Any])
async def readiness_check():
    """Readiness check - verifies all services are operational"""
    try:
        checks = {
            "mongodb": False,
            "redis": False,
            "faiss": False,
            "agents": False
        }
        
        # Check MongoDB
        try:
            await mongodb_connector.get_conversation("test")
            checks["mongodb"] = True
        except:
            pass
        
        # Check Redis
        try:
            await redis_connector.get("test")
            checks["redis"] = True
        except:
            pass
        
        # Check FAISS
        try:
            faiss_connector.get_index_size()
            checks["faiss"] = True
        except:
            pass
        
        # Check agents
        checks["agents"] = len(AGENTS) == 8
        
        # Overall ready status
        ready = all(checks.values())
        
        return {
            "ready": ready,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Readiness check error: {e}", exc_info=True)
        return {
            "ready": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/status", response_model=Dict[str, Any])
async def system_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Detailed system status (authenticated users only)"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get service statuses
        services = {
            "mongodb": {
                "status": "operational",
                "connection_pool": mongodb_connector.get_connection_pool_stats()
                if hasattr(mongodb_connector, 'get_connection_pool_stats') else {}
            },
            "redis": {
                "status": "operational",
                "memory_usage": await redis_connector.get_memory_usage()
                if hasattr(redis_connector, 'get_memory_usage') else "unknown"
            },
            "faiss": {
                "status": "operational",
                "index_size": faiss_connector.get_index_size(),
                "embedding_dimension": config.EMBEDDING_DIMENSION
            }
        }
        
        # Get agent statuses
        agent_statuses = {}
        for agent_name, agent in AGENTS.items():
            agent_statuses[agent_name] = {
                "status": "active",
                "description": agent.description
            }
        
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "free": disk.free,
                    "percent": disk.percent
                }
            },
            "services": services,
            "agents": agent_statuses,
            "uptime": datetime.utcnow().isoformat()  # Would calculate actual uptime
        }
        
    except Exception as e:
        logger.error(f"System status error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get system status")

@router.get("/metrics", response_model=Dict[str, Any])
async def system_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """System performance metrics (admin only)"""
    try:
        # Check permissions
        if current_user["role"] not in ["admin", "God"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Collect metrics
        metrics = {
            "requests": {
                "total": 0,  # Would track actual requests
                "success": 0,
                "errors": 0,
                "average_response_time": 0
            },
            "agents": {},
            "storage": {
                "mongodb": {
                    "documents": await mongodb_connector.get_collection_stats()
                    if hasattr(mongodb_connector, 'get_collection_stats') else {}
                },
                "redis": {
                    "keys": await redis_connector.get_key_count()
                    if hasattr(redis_connector, 'get_key_count') else 0
                },
                "faiss": {
                    "vectors": faiss_connector.get_index_size()
                }
            }
        }
        
        # Get agent metrics
        for agent_name, agent in AGENTS.items():
            if hasattr(agent, 'get_metrics'):
                metrics["agents"][agent_name] = await agent.get_metrics()
            else:
                metrics["agents"][agent_name] = {
                    "requests_processed": 0,
                    "average_processing_time": 0
                }
        
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"System metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get metrics")

@router.post("/test-connection", response_model=Dict[str, Any])
async def test_connection(
    service: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Test connection to a specific service (admin only)"""
    try:
        # Check permissions
        if current_user["role"] not in ["admin", "God"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        result = {
            "service": service,
            "status": "unknown",
            "details": {}
        }
        
        if service == "mongodb":
            try:
                # Test MongoDB connection
                test_id = await mongodb_connector.save_conversation({
                    "test": True,
                    "timestamp": datetime.utcnow().isoformat()
                })
                await mongodb_connector.delete_conversation(test_id)
                result["status"] = "connected"
                result["details"]["message"] = "MongoDB connection successful"
            except Exception as e:
                result["status"] = "failed"
                result["details"]["error"] = str(e)
        
        elif service == "redis":
            try:
                # Test Redis connection
                await redis_connector.set_with_ttl("test_key", {"test": True}, 5)
                await redis_connector.delete("test_key")
                result["status"] = "connected"
                result["details"]["message"] = "Redis connection successful"
            except Exception as e:
                result["status"] = "failed"
                result["details"]["error"] = str(e)
        
        elif service == "faiss":
            try:
                # Test FAISS
                size = faiss_connector.get_index_size()
                result["status"] = "connected"
                result["details"]["index_size"] = size
                result["details"]["message"] = "FAISS index accessible"
            except Exception as e:
                result["status"] = "failed"
                result["details"]["error"] = str(e)
        
        elif service == "cms_api":
            try:
                # Test CMS API connection
                from api_client import cms_api_client
                test_result = await cms_api_client.test_connection()
                result["status"] = "connected" if test_result else "failed"
                result["details"]["message"] = "CMS API connection successful" if test_result else "CMS API connection failed"
            except Exception as e:
                result["status"] = "failed"
                result["details"]["error"] = str(e)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown service: {service}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Test connection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Connection test failed")