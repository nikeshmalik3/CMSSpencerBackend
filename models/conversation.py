from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None

class ConversationContext(BaseModel):
    user_id: int
    session_id: str
    company_id: int
    project_id: Optional[int] = None
    user_role: str
    preferences: Dict[str, Any] = Field(default_factory=dict)

class AgentExecution(BaseModel):
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class Conversation(BaseModel):
    id: str = Field(description="MongoDB ObjectId as string")
    session_id: str
    user_id: int
    messages: List[Message]
    context: ConversationContext
    agent_executions: List[AgentExecution] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "uuid-string",
                "user_id": 3392,
                "messages": [
                    {
                        "role": "user",
                        "content": "Show me pending orders",
                        "timestamp": "2025-01-04T10:00:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "I found 5 pending orders...",
                        "timestamp": "2025-01-04T10:00:02Z"
                    }
                ],
                "context": {
                    "user_id": 3392,
                    "session_id": "uuid-string",
                    "company_id": 5,
                    "user_role": "God"
                }
            }
        }

class ConversationSummary(BaseModel):
    """Summary for listing conversations"""
    id: str
    session_id: str
    user_id: int
    message_count: int
    last_message: str
    last_activity: datetime
    created_at: datetime