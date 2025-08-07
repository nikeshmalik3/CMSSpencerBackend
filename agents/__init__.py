"""Agent implementations for Spencer AI"""

from .master_orchestrator import master_orchestrator
from .api_executor import api_executor
from .semantic_search import semantic_search
from .document_processor import document_processor
from .workflow_automation import workflow_automation
from .analytics_insights import analytics_insights
from .voice_nlu import voice_nlu
from .compliance_validation import compliance_validation

# Agent mapping for routes
AGENTS = {
    "orchestrator": master_orchestrator,
    "api_executor": api_executor,
    "semantic_search": semantic_search,
    "document_processor": document_processor,
    "workflow_automation": workflow_automation,
    "analytics": analytics_insights,
    "voice_nlu": voice_nlu,
    "compliance": compliance_validation
}

__all__ = [
    "master_orchestrator",
    "api_executor",
    "semantic_search",
    "document_processor",
    "workflow_automation",
    "analytics_insights",
    "voice_nlu",
    "compliance_validation",
    "AGENTS"
]