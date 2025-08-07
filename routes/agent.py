"""Direct agent access routes for testing and specific agent functionality"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
import logging

from agents import (
    master_orchestrator,
    api_executor,
    semantic_search,
    document_processor,
    workflow_automation,
    analytics_insights,
    voice_nlu,
    compliance_validation
)
from middleware.auth import get_current_user
from config.settings import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])

# Agent mapping
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

@router.get("/", response_model=Dict[str, Any])
async def list_agents(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List available agents and their capabilities"""
    agents_info = []
    
    for agent_name, agent in AGENTS.items():
        agents_info.append({
            "name": agent_name,
            "description": agent.description,
            "status": "active",
            "capabilities": agent.get_capabilities() if hasattr(agent, 'get_capabilities') else []
        })
    
    return {
        "agents": agents_info,
        "total": len(agents_info)
    }

@router.post("/{agent_name}/execute", response_model=Dict[str, Any])
async def execute_agent(
    agent_name: str,
    request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Execute a specific agent directly
    Useful for testing or when you need specific agent functionality
    """
    try:
        # Check if agent exists
        if agent_name not in AGENTS:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        agent = AGENTS[agent_name]
        
        # Add user context
        if "context" not in request:
            request["context"] = {}
        
        request["context"].update({
            "user_id": current_user["user_id"],
            "user_role": current_user["role"],
            "client_id": current_user.get("client_id"),
            "direct_execution": True
        })
        
        # Validate request
        if not await agent.validate_request(request):
            raise HTTPException(status_code=400, detail="Invalid request for this agent")
        
        # Execute agent
        result = await agent.process(request)
        
        return {
            "agent": agent_name,
            "status": "success" if result.get("success", True) else "failed",
            "data": result.get("data", {}),
            "metadata": result.get("metadata", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Agent execution failed")

@router.get("/{agent_name}/status", response_model=Dict[str, Any])
async def get_agent_status(
    agent_name: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get agent status and metrics"""
    try:
        if agent_name not in AGENTS:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        agent = AGENTS[agent_name]
        
        # Get agent metrics if available
        metrics = await agent.get_metrics() if hasattr(agent, 'get_metrics') else {}
        
        return {
            "agent": agent_name,
            "status": "active",
            "description": agent.description,
            "metrics": metrics,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get agent status error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get agent status")

@router.post("/{agent_name}/validate", response_model=Dict[str, Any])
async def validate_agent_request(
    agent_name: str,
    request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Validate a request for a specific agent without executing it"""
    try:
        if agent_name not in AGENTS:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        agent = AGENTS[agent_name]
        
        # Add minimal context for validation
        if "context" not in request:
            request["context"] = {}
        
        request["context"]["user_id"] = current_user["user_id"]
        
        # Validate
        is_valid = await agent.validate_request(request)
        
        # Get validation details if available
        validation_details = {}
        if hasattr(agent, 'get_validation_details'):
            validation_details = agent.get_validation_details(request)
        
        return {
            "agent": agent_name,
            "valid": is_valid,
            "details": validation_details
        }
        
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Validation failed")