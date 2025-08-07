from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime
import uuid
from enum import Enum

from api.cms_client import cms_client

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status states"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"

class AgentPriority(Enum):
    """Agent execution priority"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

class BaseAgent(ABC):
    """Base class for all Spencer AI agents"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.agent_id = str(uuid.uuid4())
        self.status = AgentStatus.IDLE
        self.priority = AgentPriority.NORMAL
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0,
            "last_request_time": None
        }
        self._processing_lock = asyncio.Lock()
        
    @abstractmethod
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request and return results
        
        Args:
            request: Dictionary containing:
                - query: User query or input
                - context: Conversation context
                - parameters: Agent-specific parameters
                
        Returns:
            Dictionary containing:
                - success: Boolean indicating success
                - data: Result data
                - error: Error message if any
                - metadata: Additional metadata
        """
        pass
    
    @abstractmethod
    async def validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate if the request can be processed by this agent"""
        pass
    
    async def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method with error handling and metrics"""
        start_time = datetime.utcnow()
        
        try:
            # Acquire lock to prevent concurrent processing
            async with self._processing_lock:
                self.status = AgentStatus.PROCESSING
                
                # Log request
                logger.info(f"Agent {self.name} processing request: {request.get('query', '')[:100]}...")
                
                # Validate request
                if not await self.validate_request(request):
                    raise ValueError("Invalid request format")
                
                # Process request
                result = await self.process(request)
                
                # Update metrics
                self._update_metrics(True, start_time)
                
                # Set status back to idle
                self.status = AgentStatus.IDLE
                
                return {
                    "success": True,
                    "agent": self.name,
                    "data": result.get("data", {}),
                    "metadata": {
                        "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                        "agent_id": self.agent_id,
                        **result.get("metadata", {})
                    }
                }
                
        except Exception as e:
            logger.error(f"Agent {self.name} error: {e}", exc_info=True)
            
            # Update metrics
            self._update_metrics(False, start_time)
            
            # Set error status
            self.status = AgentStatus.ERROR
            
            return {
                "success": False,
                "agent": self.name,
                "error": str(e),
                "metadata": {
                    "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                    "agent_id": self.agent_id
                }
            }
    
    def _update_metrics(self, success: bool, start_time: datetime):
        """Update agent metrics"""
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        self.metrics["total_requests"] += 1
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Update average response time
        total = self.metrics["total_requests"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        self.metrics["last_request_time"] = datetime.utcnow().isoformat()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        return {
            "agent_name": self.name,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "priority": self.priority.value,
            **self.metrics
        }
    
    def set_priority(self, priority: AgentPriority):
        """Set agent priority"""
        self.priority = priority
    
    def enable(self):
        """Enable the agent"""
        if self.status == AgentStatus.DISABLED:
            self.status = AgentStatus.IDLE
            logger.info(f"Agent {self.name} enabled")
    
    def disable(self):
        """Disable the agent"""
        self.status = AgentStatus.DISABLED
        logger.info(f"Agent {self.name} disabled")
    
    async def health_check(self) -> bool:
        """Check if agent is healthy"""
        return self.status in [AgentStatus.IDLE, AgentStatus.PROCESSING]
    
    @staticmethod
    async def call_cms_api(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
        """Helper method to call CMS API"""
        async with cms_client:
            if method == "GET":
                return await cms_client.get(endpoint)
            elif method == "POST":
                return await cms_client.post(endpoint, data or {})
            elif method == "PUT":
                return await cms_client.put(endpoint, data or {})
            elif method == "DELETE":
                return await cms_client.delete(endpoint)
            else:
                raise ValueError(f"Unsupported method: {method}")