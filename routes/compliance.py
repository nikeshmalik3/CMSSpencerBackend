"""Compliance and validation routes"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from agents.compliance_validation import compliance_validation
from middleware.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/compliance", tags=["compliance"])

@router.post("/validate", response_model=Dict[str, Any])
async def validate_request(
    validation_request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Validate a request before execution"""
    try:
        # Build compliance request
        request = {
            "query": validation_request.get("query", ""),
            "action": validation_request.get("action"),
            "data": validation_request.get("data"),
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"],
                "client_id": current_user.get("client_id")
            },
            "parameters": {
                "validate_only": True
            }
        }
        
        # Process validation
        result = await compliance_validation.process(request)
        
        return {
            "status": "success",
            "valid": result["data"]["valid"],
            "violations": result["data"].get("violations", []),
            "warnings": result["data"].get("warnings", []),
            "checked_at": result["data"]["checked_at"]
        }
        
    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Validation failed")

@router.post("/validate-data", response_model=Dict[str, Any])
async def validate_data(
    data_validation: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Validate data against business rules"""
    try:
        # Build request
        request = {
            "data": data_validation["data"],
            "entity_type": data_validation["entity_type"],
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"]
            }
        }
        
        # Process validation
        result = await compliance_validation.process(request)
        
        return {
            "status": "success",
            "valid": result["data"]["valid"],
            "violations": result["data"].get("violations", []),
            "warnings": result["data"].get("warnings", []),
            "entity_type": result["data"]["entity_type"]
        }
        
    except Exception as e:
        logger.error(f"Data validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Data validation failed")

@router.post("/check-permission", response_model=Dict[str, Any])
async def check_permission(
    permission_request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Check if user has permission for an action"""
    try:
        # Build request
        request = {
            "action": permission_request["action"],
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"],
                "resource": permission_request.get("resource")
            }
        }
        
        # Check permission
        result = await compliance_validation.process(request)
        
        return {
            "status": "success",
            "allowed": result["data"]["allowed"],
            "reason": result["data"].get("reason"),
            "required_role": result["data"].get("required_role"),
            "user_role": current_user["role"]
        }
        
    except Exception as e:
        logger.error(f"Permission check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Permission check failed")

@router.post("/validate-workflow", response_model=Dict[str, Any])
async def validate_workflow(
    workflow_validation: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Validate workflow compliance"""
    try:
        # Build request
        request = {
            "workflow": workflow_validation["workflow"],
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"]
            }
        }
        
        # Validate workflow
        result = await compliance_validation.process(request)
        
        return {
            "status": "success",
            "valid": result["data"]["valid"],
            "violations": result["data"].get("violations", []),
            "warnings": result["data"].get("warnings", []),
            "workflow_name": result["data"]["workflow_name"]
        }
        
    except Exception as e:
        logger.error(f"Workflow validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Workflow validation failed")

@router.get("/compliance-status", response_model=Dict[str, Any])
async def get_compliance_status(
    check_type: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get overall compliance status"""
    try:
        # Build request
        request = {
            "check_type": check_type or "general",
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"],
                "project_id": current_user.get("current_project_id")
            }
        }
        
        # Get compliance status
        result = await compliance_validation.process(request)
        
        return {
            "status": "success",
            "compliance_score": result["data"]["compliance_score"],
            "compliance_status": result["data"]["status"],
            "results": result["data"]["results"],
            "checked_at": result["data"]["checked_at"]
        }
        
    except Exception as e:
        logger.error(f"Get compliance status error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get compliance status")

@router.get("/rules", response_model=Dict[str, Any])
async def get_validation_rules(
    entity_type: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get validation rules for entity types"""
    try:
        # Get rules based on user role
        if current_user["role"] in ["admin", "God"]:
            # Return all rules
            if entity_type:
                rules = compliance_validation.validation_rules.get(entity_type, [])
            else:
                rules = compliance_validation.validation_rules
        else:
            # Return limited rules
            rules = {
                "message": "Contact admin for detailed validation rules"
            }
        
        return {
            "status": "success",
            "rules": rules,
            "entity_types": list(compliance_validation.validation_rules.keys())
        }
        
    except Exception as e:
        logger.error(f"Get rules error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get rules")

@router.get("/business-rules", response_model=Dict[str, Any])
async def get_business_rules(
    rule_category: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get business rules and thresholds"""
    try:
        # Get rules based on category
        if rule_category:
            rules = compliance_validation.business_rules.get(rule_category, {})
        else:
            rules = compliance_validation.business_rules
        
        # Filter sensitive information for non-admin users
        if current_user["role"] not in ["admin", "God"]:
            # Show only relevant limits
            filtered_rules = {
                "approval_limits": {
                    current_user["role"]: rules.get("approval_limits", {}).get(
                        current_user["role"], 0
                    )
                }
            }
            rules = filtered_rules
        
        return {
            "status": "success",
            "rules": rules,
            "categories": list(compliance_validation.business_rules.keys())
        }
        
    except Exception as e:
        logger.error(f"Get business rules error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get business rules")

@router.get("/audit-log", response_model=Dict[str, Any])
async def get_audit_log(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 50,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get audit log entries (admin only)"""
    try:
        # Check permissions
        if current_user["role"] not in ["admin", "God"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Build filters
        filters = {}
        if start_date:
            filters["timestamp"] = {"$gte": start_date}
        if end_date:
            filters.setdefault("timestamp", {})["$lte"] = end_date
        if action:
            filters["action"] = action
        
        # Get audit entries from MongoDB
        from storage import mongodb_connector
        audit_entries = await mongodb_connector.find_audit_logs(
            filters=filters,
            limit=limit
        )
        
        return {
            "status": "success",
            "audit_entries": audit_entries,
            "total": len(audit_entries)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get audit log error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get audit log")

@router.post("/safety-check", response_model=Dict[str, Any])
async def safety_compliance_check(
    safety_request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Check safety compliance for activities"""
    try:
        # Build request
        request = {
            "check_type": "safety",
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"],
                "location": safety_request.get("location"),
                "activity_type": safety_request.get("activity_type"),
                "project_id": safety_request.get("project_id")
            }
        }
        
        # Check safety compliance
        result = await compliance_validation.process(request)
        
        # Extract safety-specific results
        safety_results = result["data"]["results"].get("safety_compliance", {})
        
        return {
            "status": "success",
            "compliant": len(safety_results.get("violations", [])) == 0,
            "violations": safety_results.get("violations", []),
            "requirements": compliance_validation.safety_regulations.get(
                safety_request.get("activity_type", "general"), 
                []
            )
        }
        
    except Exception as e:
        logger.error(f"Safety check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Safety check failed")