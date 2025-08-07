"""API Routes package"""

from .conversation import router as conversation_router
from .agent import router as agent_router
from .document import router as document_router
from .analytics import router as analytics_router
from .workflow import router as workflow_router
from .search import router as search_router
from .voice import router as voice_router
from .compliance import router as compliance_router
from .health import router as health_router

__all__ = [
    "conversation_router",
    "agent_router", 
    "document_router",
    "analytics_router",
    "workflow_router",
    "search_router",
    "voice_router",
    "compliance_router",
    "health_router"
]