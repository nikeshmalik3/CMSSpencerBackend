from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Any, Optional
import jwt
import logging
from datetime import datetime

from config.settings import config

logger = logging.getLogger(__name__)

# Security scheme for FastAPI docs
security = HTTPBearer()

class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware for handling authentication"""
    
    # Endpoints that don't require authentication
    EXEMPT_PATHS = [
        "/",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/info"
    ]
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"error": "Authorization header missing"}
            )
        
        # Validate Bearer token format
        try:
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer":
                raise ValueError("Invalid authentication scheme")
        except ValueError:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid authorization header format"}
            )
        
        # Decode and validate JWT token
        try:
            # Note: In production, verify with proper RSA public key
            # For now, we'll do basic validation
            payload = jwt.decode(token, options={"verify_signature": False})
            
            # Check token expiration
            exp = payload.get("exp", 0)
            if datetime.utcnow().timestamp() > exp:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Token expired"}
                )
            
            # Add user info to request state
            request.state.user_id = payload.get("sub")
            request.state.client_id = payload.get("aud")
            request.state.token = token
            
        except jwt.DecodeError as e:
            logger.error(f"JWT decode error: {e}")
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid token"}
            )
        except Exception as e:
            logger.error(f"Auth error: {e}")
            return JSONResponse(
                status_code=401,
                content={"error": "Authentication failed"}
            )
        
        # Continue processing
        response = await call_next(request)
        return response

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Dependency function to get the current authenticated user
    Used in route handlers
    """
    token = credentials.credentials
    
    try:
        # Decode JWT token (without signature verification for now)
        payload = jwt.decode(token, options={"verify_signature": False})
        
        # Check token expiration
        exp = payload.get("exp", 0)
        if datetime.utcnow().timestamp() > exp:
            raise HTTPException(
                status_code=401,
                detail="Token expired"
            )
        
        # Return user info from token
        return {
            "user_id": payload.get("sub", config.USER_ID),
            "client_id": payload.get("aud", config.CLIENT_ID),
            "role": config.USER_ROLE,  # This should come from token in production
            "name": config.USER_NAME    # This should come from token in production
        }
        
    except jwt.DecodeError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Authentication failed"
        )

async def verify_websocket_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify token for WebSocket connections
    """
    try:
        # Decode JWT token
        payload = jwt.decode(token, options={"verify_signature": False})
        
        # Check token expiration
        exp = payload.get("exp", 0)
        if datetime.utcnow().timestamp() > exp:
            return None
        
        return {
            "user_id": payload.get("sub", config.USER_ID),
            "client_id": payload.get("aud", config.CLIENT_ID),
            "role": config.USER_ROLE,
            "name": config.USER_NAME
        }
    except Exception:
        return None