import redis.asyncio as redis
import json
import logging
from typing import Any, Dict, Optional, List
from datetime import timedelta

from config.settings import config

logger = logging.getLogger(__name__)

class RedisConnector:
    """Async Redis connector for caching and session management"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.pool: Optional[redis.ConnectionPool] = None
        
    async def initialize(self):
        """Initialize Redis connection pool"""
        try:
            # Use SSL connection if configured
            connection_kwargs = {
                "host": config.REDIS_HOST,
                "port": config.REDIS_PORT,
                "db": config.REDIS_DB,
                "password": config.REDIS_PASSWORD,
                "max_connections": config.REDIS_POOL_SIZE,
                "decode_responses": True
            }
            
            if config.REDIS_USE_SSL:
                connection_kwargs["connection_class"] = redis.SSLConnection
                connection_kwargs["ssl_cert_reqs"] = "none"  # Skip cert verification for self-signed
            
            self.pool = redis.ConnectionPool(**connection_kwargs)
            
            self.redis_client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.redis_client.ping()
            logger.info(f"Redis connection established successfully (SSL: {config.REDIS_USE_SSL})")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            await self.pool.disconnect()
            logger.info("Redis connection closed")
    
    # Session Management
    async def set_session(self, session_id: str, data: Dict[str, Any], ttl: int = None) -> bool:
        """Store session data"""
        try:
            key = f"session:{session_id}"
            ttl = ttl or config.CACHE_TTL_SESSION
            
            await self.redis_client.setex(
                key,
                timedelta(seconds=ttl),
                json.dumps(data)
            )
            return True
        except Exception as e:
            logger.error(f"Error setting session {session_id}: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data"""
        try:
            key = f"session:{session_id}"
            data = await self.redis_client.get(key)
            
            if data:
                # Refresh TTL on access
                await self.redis_client.expire(key, config.CACHE_TTL_SESSION)
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session data"""
        try:
            key = f"session:{session_id}"
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    # Conversation Caching
    async def cache_conversation(self, conversation_id: str, data: Dict[str, Any]) -> bool:
        """Cache active conversation"""
        try:
            key = f"conversation:{conversation_id}"
            await self.redis_client.setex(
                key,
                timedelta(seconds=config.CACHE_TTL_CONVERSATION),
                json.dumps(data)
            )
            return True
        except Exception as e:
            logger.error(f"Error caching conversation {conversation_id}: {e}")
            return False
    
    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get cached conversation"""
        try:
            key = f"conversation:{conversation_id}"
            data = await self.redis_client.get(key)
            
            if data:
                # Refresh TTL on access
                await self.redis_client.expire(key, config.CACHE_TTL_CONVERSATION)
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting conversation {conversation_id}: {e}")
            return None
    
    # API Response Caching
    async def cache_api_response(self, endpoint: str, params: Dict[str, Any], response: Dict[str, Any]) -> bool:
        """Cache API response"""
        try:
            # Create cache key from endpoint and params
            param_str = json.dumps(params, sort_keys=True)
            key = f"api:{endpoint}:{hash(param_str)}"
            
            await self.redis_client.setex(
                key,
                timedelta(seconds=config.CACHE_TTL_API),
                json.dumps(response)
            )
            return True
        except Exception as e:
            logger.error(f"Error caching API response for {endpoint}: {e}")
            return False
    
    async def get_cached_api_response(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached API response"""
        try:
            param_str = json.dumps(params, sort_keys=True)
            key = f"api:{endpoint}:{hash(param_str)}"
            
            data = await self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting cached API response for {endpoint}: {e}")
            return None
    
    # Agent Message Queue
    async def enqueue_agent_message(self, agent_name: str, message: Dict[str, Any]) -> bool:
        """Add message to agent queue"""
        try:
            queue_key = f"queue:agent:{agent_name}"
            await self.redis_client.rpush(queue_key, json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Error enqueueing message for {agent_name}: {e}")
            return False
    
    async def dequeue_agent_message(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get message from agent queue"""
        try:
            queue_key = f"queue:agent:{agent_name}"
            data = await self.redis_client.lpop(queue_key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error dequeueing message for {agent_name}: {e}")
            return None
    
    # Real-time Metrics
    async def increment_metric(self, metric_name: str, value: int = 1) -> bool:
        """Increment a metric counter"""
        try:
            key = f"metrics:{metric_name}"
            await self.redis_client.incrby(key, value)
            return True
        except Exception as e:
            logger.error(f"Error incrementing metric {metric_name}: {e}")
            return False
    
    async def get_metric(self, metric_name: str) -> int:
        """Get metric value"""
        try:
            key = f"metrics:{metric_name}"
            value = await self.redis_client.get(key)
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Error getting metric {metric_name}: {e}")
            return 0
    
    # Rate Limiting Support
    async def check_rate_limit(self, user_id: str, limit: int = 100, window: int = 60) -> bool:
        """Check if user is within rate limit"""
        try:
            key = f"rate_limit:{user_id}"
            current = await self.redis_client.incr(key)
            
            if current == 1:
                await self.redis_client.expire(key, window)
            
            return current <= limit
        except Exception as e:
            logger.error(f"Error checking rate limit for {user_id}: {e}")
            return True  # Allow on error
    
    # Utility Methods
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        return await self.redis_client.exists(key) > 0
    
    async def set_with_ttl(self, key: str, value: Any, ttl: int) -> bool:
        """Set value with TTL"""
        try:
            await self.redis_client.setex(key, timedelta(seconds=ttl), json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Error setting {key}: {e}")
            return False
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value"""
        try:
            data = await self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Error getting {key}: {e}")
            return None

# Singleton instance
redis_connector = RedisConnector()