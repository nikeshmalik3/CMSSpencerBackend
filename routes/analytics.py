"""Analytics and reporting routes"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

from agents.analytics_insights import analytics_insights
from storage import mongodb_connector
from middleware.auth import get_current_user
from models import TimeGranularity, MetricType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

@router.post("/query", response_model=Dict[str, Any])
async def analytics_query(
    request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Execute natural language analytics query"""
    try:
        # Add user context
        request["context"] = {
            "user_id": current_user["user_id"],
            "user_role": current_user["role"],
            "client_id": current_user.get("client_id")
        }
        
        # Process through analytics agent
        result = await analytics_insights.process(request)
        
        return {
            "status": "success",
            "data": result["data"],
            "metadata": result.get("metadata", {})
        }
        
    except Exception as e:
        logger.error(f"Analytics query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analytics query failed")

@router.get("/reports", response_model=Dict[str, Any])
async def list_reports(
    report_type: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List available analytics reports"""
    try:
        filters = {"user_id": current_user["user_id"]}
        
        if report_type:
            filters["report_type"] = report_type
            
        if project_id:
            filters["project_id"] = project_id
        
        reports = await mongodb_connector.find_analytics_reports(
            filters=filters,
            limit=limit,
            skip=offset
        )
        
        return {
            "reports": reports,
            "total": len(reports),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"List reports error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list reports")

@router.post("/reports/{report_type}", response_model=Dict[str, Any])
async def generate_report(
    report_type: str,
    parameters: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate a specific report type"""
    try:
        # Validate report type
        valid_types = ["project_dashboard", "weekly_summary", "executive_report", "custom"]
        if report_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid report type. Must be one of: {valid_types}"
            )
        
        # Build request
        request = {
            "report_type": report_type,
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"],
                "project_id": parameters.get("project_id")
            },
            "parameters": parameters
        }
        
        # Generate report
        result = await analytics_insights.process(request)
        
        return {
            "status": "success",
            "report": result["data"]["report"],
            "report_id": result["data"]["report_id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generate report error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Report generation failed")

@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics(
    metrics: List[str] = Query(...),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    granularity: TimeGranularity = TimeGranularity.DAILY,
    project_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get specific metrics"""
    try:
        # Build time range
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
        if not end_date:
            end_date = datetime.utcnow().isoformat()
        
        # Build request
        request = {
            "metrics": metrics,
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"],
                "project_id": project_id
            },
            "parameters": {
                "period": {
                    "start": start_date,
                    "end": end_date,
                    "name": f"{granularity.value} metrics"
                },
                "granularity": granularity
            }
        }
        
        # Get metrics
        result = await analytics_insights.process(request)
        
        return {
            "status": "success",
            "metrics": result["data"]["metrics"],
            "period": result["data"]["period"]
        }
        
    except Exception as e:
        logger.error(f"Get metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get metrics")

@router.get("/kpis", response_model=Dict[str, Any])
async def get_kpis(
    project_id: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get current KPIs"""
    try:
        # Get all KPIs
        request = {
            "query": "show all kpis",
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"],
                "project_id": project_id
            }
        }
        
        result = await analytics_insights.process(request)
        
        return {
            "status": "success",
            "kpis": result["data"].get("kpis", {}),
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get KPIs error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get KPIs")

@router.post("/forecast", response_model=Dict[str, Any])
async def generate_forecast(
    forecast_request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate forecast for metrics"""
    try:
        # Build request
        request = {
            "query": f"forecast {', '.join(forecast_request.get('metrics', []))}",
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"],
                "project_id": forecast_request.get("project_id")
            },
            "parameters": forecast_request.get("parameters", {})
        }
        
        # Generate forecast
        result = await analytics_insights.process(request)
        
        return {
            "status": "success",
            "forecasts": result["data"].get("forecasts", {}),
            "forecast_period": result["data"].get("forecast_period", "30 days")
        }
        
    except Exception as e:
        logger.error(f"Generate forecast error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Forecast generation failed")

@router.get("/insights", response_model=Dict[str, Any])
async def get_insights(
    project_id: Optional[str] = None,
    insight_type: Optional[str] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get AI-generated insights"""
    try:
        # Generate insights based on recent data
        request = {
            "query": "generate insights",
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"],
                "project_id": project_id
            },
            "parameters": {
                "insight_type": insight_type or "general"
            }
        }
        
        result = await analytics_insights.process(request)
        
        # Extract insights from various response formats
        insights = []
        if "key_insights" in result["data"]:
            insights = result["data"]["key_insights"]
        elif "insights" in result["data"]:
            insights = result["data"]["insights"]
        elif "report" in result["data"] and "insights" in result["data"]["report"]:
            insights = result["data"]["report"]["insights"]
        
        return {
            "status": "success",
            "insights": insights,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Get insights error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate insights")

@router.get("/reports/{report_id}", response_model=Dict[str, Any])
async def get_report(
    report_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get a specific report"""
    try:
        report = await mongodb_connector.get_analytics_report(report_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        # Check access
        if (report["user_id"] != current_user["user_id"] and 
            current_user["role"] not in ["admin", "God"]):
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "status": "success",
            "report": report
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get report error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get report")