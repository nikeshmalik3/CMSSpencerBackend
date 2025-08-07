from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from config.settings import config
from api.cms_client import cms_client
from middleware.auth import AuthMiddleware
from middleware.rate_limit import RateLimitMiddleware
from routes import (
    conversation_router,
    agent_router,
    analytics_router,
    workflow_router,
    search_router,
    voice_router,
    compliance_router,
    health_router
)
from routes.documents import router as document_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Spencer AI Backend...")
    
    # Initialize API client
    await cms_client.initialize()
    logger.info("CMS API client initialized")
    
    # Initialize Redis connection
    from storage.redis_connector import redis_connector
    try:
        await redis_connector.initialize()
        logger.info("Redis connection initialized")
    except Exception as e:
        logger.error(f"Redis initialization failed: {e}")
        # Continue without Redis (degraded mode)
    
    # Initialize MongoDB connection
    from storage.mongodb_connector import mongodb_connector
    try:
        await mongodb_connector.initialize()
        logger.info("MongoDB connection initialized")
    except Exception as e:
        logger.error(f"MongoDB initialization failed: {e}")
        # Continue without MongoDB (degraded mode)
    
    # Initialize FAISS indexes (skip if import fails due to numpy conflict)
    try:
        from storage.faiss_connector import faiss_connector
        await faiss_connector.initialize()
        logger.info("FAISS indexes initialized")
    except ImportError as e:
        logger.warning(f"FAISS disabled due to numpy conflict: {e}")
        logger.info("Continuing without FAISS vector search")
    except Exception as e:
        logger.error(f"FAISS initialization failed: {e}")
        # Continue without FAISS (degraded mode)
    
    logger.info("Spencer AI Backend started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Spencer AI Backend...")
    
    # Close API client
    await cms_client.close()
    
    # Close Redis connection
    try:
        from storage.redis_connector import redis_connector
        await redis_connector.close()
        logger.info("Redis connection closed")
    except:
        pass
    
    # Close MongoDB connection
    try:
        from storage.mongodb_connector import mongodb_connector
        await mongodb_connector.close()
        logger.info("MongoDB connection closed")
    except:
        pass
    
    # Close FAISS (if it was initialized)
    try:
        from storage.faiss_connector import faiss_connector
        await faiss_connector.close()
        logger.info("FAISS closed")
    except (ImportError, NameError):
        pass  # FAISS wasn't loaded due to numpy conflict
    except:
        pass
    
    logger.info("Spencer AI Backend shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Spencer AI Backend",
    description="Advanced Construction Management Assistant API",
    version="3.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": request.url.path
        }
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "Spencer AI",
        "version": "3.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    checks = {
        "api": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # TODO: Add actual health checks
    # - Redis connection
    # - MongoDB connection
    # - CMS API connectivity
    # - Agent status
    
    return checks

# API Info endpoint
@app.get("/api/v1/info")
async def api_info():
    return {
        "name": "Spencer AI API",
        "version": "1.0.0",
        "capabilities": [
            "chat",
            "document_processing",
            "voice_transcription",
            "analytics",
            "workflow_automation"
        ],
        "agents": [
            "master_orchestrator",
            "api_executor",
            "semantic_search",
            "document_processor",
            "workflow_automation",
            "analytics",
            "voice_nlu",
            "compliance"
        ],
        "status": "ready"
    }

# Include routers
app.include_router(conversation_router)
app.include_router(agent_router)
app.include_router(document_router)
app.include_router(analytics_router)
app.include_router(workflow_router)
app.include_router(search_router)
app.include_router(voice_router)
app.include_router(compliance_router)
app.include_router(health_router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        workers=1 if config.DEBUG else 4,
        log_level=config.LOG_LEVEL.lower()
    )