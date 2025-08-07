from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepType(str, Enum):
    API_CALL = "api_call"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"
    WAIT = "wait"
    TRANSFORM = "transform"
    NOTIFICATION = "notification"

class WorkflowStep(BaseModel):
    id: str
    name: str
    type: StepType
    config: Dict[str, Any]
    next_steps: List[str] = Field(default_factory=list)
    error_handler: Optional[str] = None
    retry_config: Optional[Dict[str, Any]] = None

class WorkflowTemplate(BaseModel):
    id: str = Field(description="MongoDB ObjectId as string")
    name: str
    description: str
    category: str  # e.g., "order_management", "reporting", "approvals"
    created_by: int
    steps: List[WorkflowStep]
    variables: Dict[str, Any] = Field(default_factory=dict)
    trigger_conditions: Optional[Dict[str, Any]] = None
    usage_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Order Approval Workflow",
                "description": "Automated order approval based on amount",
                "category": "order_management",
                "steps": [
                    {
                        "id": "step1",
                        "name": "Check Order Amount",
                        "type": "api_call",
                        "config": {
                            "endpoint": "/api/orders/{order_id}",
                            "method": "GET"
                        },
                        "next_steps": ["step2"]
                    }
                ]
            }
        }

class WorkflowExecution(BaseModel):
    id: str = Field(description="MongoDB ObjectId as string")
    workflow_id: str
    workflow_name: str
    user_id: int
    status: WorkflowStatus
    current_step: Optional[str] = None
    execution_data: Dict[str, Any] = Field(default_factory=dict)
    step_results: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
class WorkflowTrigger(BaseModel):
    workflow_id: str
    input_data: Dict[str, Any]
    user_id: int
    context: Optional[Dict[str, Any]] = None