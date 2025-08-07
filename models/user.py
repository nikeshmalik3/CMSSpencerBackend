from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Any, Optional
from datetime import datetime

class UserRole(BaseModel):
    id: int
    code: str
    description: Optional[str] = None
    permission_editor: bool = False

class UserPreferences(BaseModel):
    language: str = "en"
    timezone: str = "UTC"
    notification_enabled: bool = True
    ui_theme: str = "light"
    default_project_filter: Optional[int] = None
    custom_settings: Dict[str, Any] = Field(default_factory=dict)

class User(BaseModel):
    id: int
    email: EmailStr
    name: str
    company_id: Optional[int] = None
    client_id: int
    roles: List[UserRole]
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    # Spencer AI specific fields
    ai_interactions_count: int = 0
    favorite_workflows: List[str] = Field(default_factory=list)
    saved_queries: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "id": 3392,
                "email": "nikesh.m@companiesms.co.uk",
                "name": "Nikesh Malik",
                "company_id": None,
                "client_id": 5,
                "roles": [
                    {
                        "id": 4,
                        "code": "God",
                        "permission_editor": True
                    }
                ],
                "preferences": {
                    "language": "en",
                    "timezone": "Europe/London"
                }
            }
        }

class UserSession(BaseModel):
    """Active user session tracking"""
    session_id: str
    user_id: int
    token: str
    client_id: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None