"""API request/response models"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class ConversationRequest(BaseModel):
    """Request model for conversation endpoint"""
    message: str = Field(..., description="User message to process")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    project_id: Optional[str] = Field(None, description="Associated project ID")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional parameters")

class ConversationResponse(BaseModel):
    """Response model for conversation endpoint"""
    conversation_id: str
    message: str
    agents_used: List[str]
    confidence: float
    suggestions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchRequest(BaseModel):
    """Request model for search endpoint"""
    query: str
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)

class AnalyticsRequest(BaseModel):
    """Request model for analytics endpoint"""
    query: Optional[str] = None
    metrics: Optional[List[str]] = None
    report_type: Optional[str] = None
    time_range: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)

class WorkflowRequest(BaseModel):
    """Request model for workflow endpoint"""
    workflow_name: Optional[str] = None
    workflow_id: Optional[str] = None
    query: Optional[str] = None
    variables: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ValidationRequest(BaseModel):
    """Request model for validation endpoint"""
    query: Optional[str] = None
    action: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    entity_type: Optional[str] = None

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    timestamp: datetime
    path: str
    details: Optional[Dict[str, Any]] = None