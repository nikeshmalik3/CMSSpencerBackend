"""Workflow automation routes"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from agents.workflow_automation import workflow_automation
from storage import mongodb_connector
from middleware.auth import get_current_user
from models import WorkflowStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workflows", tags=["workflows"])

@router.post("/execute", response_model=Dict[str, Any])
async def execute_workflow(
    request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Execute a workflow by name or natural language"""
    try:
        # Add user context
        request["context"] = {
            "user_id": current_user["user_id"],
            "user_role": current_user["role"],
            "client_id": current_user.get("client_id")
        }
        
        # Process through workflow agent
        result = await workflow_automation.process(request)
        
        return {
            "status": "success",
            "workflow": result["data"],
            "message": result["data"].get("message", "Workflow started")
        }
        
    except Exception as e:
        logger.error(f"Workflow execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Workflow execution failed")

@router.get("/templates", response_model=Dict[str, Any])
async def list_workflow_templates(
    category: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List available workflow templates"""
    try:
        # Get built-in templates
        templates = []
        for name, template in workflow_automation.workflow_templates.items():
            template_dict = template.dict()
            template_dict["id"] = name
            template_dict["type"] = "built-in"
            
            if not category or template.category == category:
                templates.append(template_dict)
        
        # Get custom templates from database
        filters = {"created_by": current_user["user_id"]}
        if category:
            filters["category"] = category
        
        custom_templates = await mongodb_connector.get_workflows_by_category(
            category or "all"
        )
        
        for template in custom_templates:
            if template.get("created_by") == current_user["user_id"]:
                template["type"] = "custom"
                templates.append(template)
        
        return {
            "templates": templates,
            "total": len(templates),
            "categories": ["order_management", "reporting", "approval", "custom"]
        }
        
    except Exception as e:
        logger.error(f"List templates error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list templates")

@router.get("/executions", response_model=Dict[str, Any])
async def list_workflow_executions(
    status: Optional[WorkflowStatus] = None,
    limit: int = 20,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List workflow executions"""
    try:
        filters = {"user_id": current_user["user_id"]}
        if status:
            filters["status"] = status.value
        
        executions = await mongodb_connector.find_workflow_executions(
            filters=filters,
            limit=limit,
            skip=offset
        )
        
        return {
            "executions": executions,
            "total": len(executions),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"List executions error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list executions")

@router.get("/executions/{execution_id}", response_model=Dict[str, Any])
async def get_workflow_execution(
    execution_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get workflow execution details"""
    try:
        execution = await mongodb_connector.get_workflow_execution(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        # Check access
        if (execution["user_id"] != current_user["user_id"] and 
            current_user["role"] not in ["admin", "God"]):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "execution": execution
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get execution")

@router.post("/templates", response_model=Dict[str, Any])
async def create_workflow_template(
    template: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a custom workflow template"""
    try:
        # Validate template structure
        required_fields = ["name", "description", "steps"]
        for field in required_fields:
            if field not in template:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # Add metadata
        template["created_by"] = current_user["user_id"]
        template["created_at"] = datetime.utcnow().isoformat()
        template["category"] = template.get("category", "custom")
        
        # Save to database
        template_id = await mongodb_connector.save_workflow_template(template)
        
        return {
            "message": "Workflow template created successfully",
            "template_id": template_id,
            "template_name": template["name"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create template error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create template")

@router.delete("/templates/{template_id}")
async def delete_workflow_template(
    template_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a custom workflow template"""
    try:
        # Get template to verify ownership
        template = await mongodb_connector.get_workflow_template(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        # Check ownership
        if (template.get("created_by") != current_user["user_id"] and 
            current_user["role"] != "God"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete template
        await mongodb_connector.delete_workflow_template(template_id)
        
        return {"message": "Template deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete template error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete template")

@router.post("/executions/{execution_id}/cancel")
async def cancel_workflow_execution(
    execution_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Cancel a running workflow execution"""
    try:
        # Get execution
        execution = await mongodb_connector.get_workflow_execution(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        # Check ownership
        if (execution["user_id"] != current_user["user_id"] and 
            current_user["role"] not in ["admin", "God"]):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if still running
        if execution["status"] != WorkflowStatus.RUNNING.value:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel workflow in {execution['status']} state"
            )
        
        # Update status
        execution["status"] = WorkflowStatus.CANCELLED.value
        execution["cancelled_at"] = datetime.utcnow().isoformat()
        
        await mongodb_connector.save_workflow(execution)
        
        # Remove from running workflows if present
        if execution_id in workflow_automation.running_workflows:
            del workflow_automation.running_workflows[execution_id]
        
        return {"message": "Workflow cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to cancel execution")

@router.get("/executions/{execution_id}/steps", response_model=Dict[str, Any])
async def get_workflow_steps(
    execution_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get detailed step-by-step execution progress"""
    try:
        execution = await mongodb_connector.get_workflow_execution(execution_id)
        
        if not execution:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        # Check access
        if (execution["user_id"] != current_user["user_id"] and 
            current_user["role"] not in ["admin", "God"]):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Format step results
        steps = []
        for step_id, result in execution.get("step_results", {}).items():
            steps.append({
                "step_id": step_id,
                "status": "completed" if result.get("success") else "failed",
                "result": result,
                "completed_at": result.get("timestamp")
            })
        
        return {
            "execution_id": execution_id,
            "current_step": execution.get("current_step"),
            "steps": steps,
            "status": execution["status"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get workflow steps error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get steps")