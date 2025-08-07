from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from enum import Enum

class MetricType(str, Enum):
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    PERCENTAGE = "percentage"
    TREND = "trend"
    DISTRIBUTION = "distribution"

class TimeGranularity(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class KPIMetric(BaseModel):
    name: str
    value: float
    unit: Optional[str] = None
    change_percentage: Optional[float] = None
    trend: Optional[List[float]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AnalyticsQuery(BaseModel):
    metric_type: MetricType
    entity_type: str  # e.g., "orders", "projects", "users"
    filters: Dict[str, Any] = Field(default_factory=dict)
    group_by: Optional[List[str]] = None
    time_range: Optional[Dict[str, datetime]] = None
    granularity: Optional[TimeGranularity] = None
    limit: int = 100

class AnalyticsReport(BaseModel):
    id: str = Field(description="MongoDB ObjectId as string")
    name: str
    description: str
    user_id: int
    query: AnalyticsQuery
    results: Dict[str, Any]
    visualizations: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
class DashboardWidget(BaseModel):
    id: str
    type: str  # "chart", "metric", "table", "map"
    title: str
    query: AnalyticsQuery
    config: Dict[str, Any]  # Chart config, colors, etc.
    position: Dict[str, int]  # x, y, width, height
    refresh_interval: Optional[int] = None  # seconds

class Dashboard(BaseModel):
    id: str = Field(description="MongoDB ObjectId as string")
    name: str
    user_id: int
    widgets: List[DashboardWidget]
    layout: str = "grid"  # or "freeform"
    shared_with: List[int] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ProjectAnalytics(BaseModel):
    project_id: int
    budget_utilization: float
    schedule_adherence: float
    productivity_score: float
    quality_metrics: Dict[str, float]
    resource_utilization: Dict[str, float]
    risk_indicators: List[Dict[str, Any]]
    calculated_at: datetime = Field(default_factory=datetime.utcnow)