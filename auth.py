"""
Simple authentication module for API routes
"""

from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Optional
import jwt
import logging

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# Mock function - replace with actual implementation
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict:
    """
    Get current user from bearer token
    This is a placeholder - integrate with your actual auth system
    """
    try:
        # For now, return the user from the test data
        # In production, decode JWT and get user from database
        return {
            "id": 3392,
            "email": "nikesh.m@companiesms.co.uk",
            "name": "Nikesh Malik",
            "client_id": 5,
            "is_admin": True,
            "roles": ["God"]
        }
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )