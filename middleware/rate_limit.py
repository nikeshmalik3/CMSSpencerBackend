"""
Rate Limiting Middleware
Implements proper rate limiting with Redis support
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict
import asyncio
from datetime import datetime
import logging
import redis.asyncio as redis

from config.settings import config

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests"""
    
    def __init__(self, app):
        super().__init__(app)
        self.requests_limit = config.RATE_LIMIT_REQUESTS  # From .env (default 100)
        self.period_seconds = config.RATE_LIMIT_PERIOD    # From .env (default 60)
        self.user_requests = defaultdict(list)  # Fallback in-memory storage
        self.redis_client = None
        self.use_redis = True
        self._start_cleanup_task()
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection for distributed rate limiting"""
        asyncio.create_task(self._connect_redis())
    
    async def _connect_redis(self):
        """Connect to Redis asynchronously"""
        try:
            self.redis_client = await redis.from_url(
                config.get_redis_url(),
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connected for rate limiting")
        except Exception as e:
            logger.warning(f"Redis not available for rate limiting, using in-memory: {e}")
            self.use_redis = False
            self.redis_client = None
    
    def _start_cleanup_task(self):
        """Start background task to clean old request entries"""
        asyncio.create_task(self._cleanup_old_entries())
    
    async def _cleanup_old_entries(self):
        """Remove old request timestamps from in-memory storage"""
        while True:
            await asyncio.sleep(60)  # Clean every minute
            current_time = time.time()
            
            # Clean entries older than the rate limit period
            for user_id in list(self.user_requests.keys()):
                self.user_requests[user_id] = [
                    ts for ts in self.user_requests[user_id]
                    if current_time - ts < self.period_seconds
                ]
                
                # Remove user if no recent requests
                if not self.user_requests[user_id]:
                    del self.user_requests[user_id]
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request"""
        # Try to get user ID from state (if authenticated)
        if hasattr(request.state, "user_id") and request.state.user_id:
            return f"user:{request.state.user_id}"
        
        # Fallback to IP address
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for proxy headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            client_ip = real_ip
        
        return f"ip:{client_ip}"
    
    async def _check_rate_limit_redis(self, client_id: str) -> tuple[bool, int]:
        """Check rate limit using Redis"""
        if not self.redis_client:
            return await self._check_rate_limit_memory(client_id)
        
        try:
            key = f"rate_limit:{client_id}"
            
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.period_seconds)
            
            results = await pipe.execute()
            current_count = results[0]
            
            remaining = max(0, self.requests_limit - current_count)
            is_allowed = current_count <= self.requests_limit
            
            return is_allowed, remaining
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to in-memory
            return await self._check_rate_limit_memory(client_id)
    
    async def _check_rate_limit_memory(self, client_id: str) -> tuple[bool, int]:
        """Check rate limit using in-memory storage"""
        current_time = time.time()
        request_times = self.user_requests[client_id]
        
        # Remove timestamps older than the period
        request_times = [
            ts for ts in request_times 
            if current_time - ts < self.period_seconds
        ]
        
        remaining = max(0, self.requests_limit - len(request_times))
        
        if len(request_times) >= self.requests_limit:
            self.user_requests[client_id] = request_times
            return False, remaining
        
        # Add current request timestamp
        request_times.append(current_time)
        self.user_requests[client_id] = request_times
        
        return True, remaining - 1
    
    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiter"""
        
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if self.use_redis and self.redis_client:
            is_allowed, remaining = await self._check_rate_limit_redis(client_id)
        else:
            is_allowed, remaining = await self._check_rate_limit_memory(client_id)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for {client_id}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_limit} requests per {self.period_seconds} seconds",
                    "retry_after": self.period_seconds,
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers={
                    "Retry-After": str(self.period_seconds),
                    "X-RateLimit-Limit": str(self.requests_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + self.period_seconds)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.period_seconds)
        
        return response