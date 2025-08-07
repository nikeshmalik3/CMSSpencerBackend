"""Conversation routes - main Spencer AI interaction endpoint"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from models import Conversation, Message, ConversationRequest, ConversationResponse
from agents.master_orchestrator import master_orchestrator
from storage import mongodb_connector, redis_connector
from middleware.auth import get_current_user
from config.settings import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

@router.post("/", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Main Spencer AI interaction endpoint
    Processes natural language queries through the Master Orchestrator
    """
    try:
        # Create or get conversation
        conversation_id = request.conversation_id
        
        if conversation_id:
            # Get existing conversation
            conversation_data = await redis_connector.get_conversation(conversation_id)
            if not conversation_data:
                # Try MongoDB
                conversation_data = await mongodb_connector.get_conversation(conversation_id)
                
            if not conversation_data:
                raise HTTPException(status_code=404, detail="Conversation not found")
                
            conversation = Conversation(**conversation_data)
        else:
            # Create new conversation
            conversation = Conversation(
                id="",
                user_id=current_user["user_id"],
                title=request.message[:50] + "..." if len(request.message) > 50 else request.message,
                project_id=request.project_id
            )
            
            # Save to MongoDB
            conv_id = await mongodb_connector.save_conversation(conversation.dict())
            conversation.id = conv_id
            
            # Cache in Redis
            await redis_connector.save_conversation(conv_id, conversation.dict())
        
        # Add user message
        user_message = Message(
            role="user",
            content=request.message,
            timestamp=datetime.utcnow()
        )
        conversation.messages.append(user_message)
        
        # Build context
        context = {
            "user_id": current_user["user_id"],
            "user_role": current_user["role"],
            "client_id": current_user.get("client_id"),
            "project_id": request.project_id or conversation.project_id,
            "conversation_id": conversation.id,
            "user_name": current_user.get("name", "User")
        }
        
        # Process through Master Orchestrator
        orchestrator_request = {
            "query": request.message,
            "conversation": conversation,
            "context": context,
            "parameters": request.parameters or {}
        }
        
        response = await master_orchestrator.process(orchestrator_request)
        
        # Add assistant response
        assistant_message = Message(
            role="assistant",
            content=response["data"]["response"],
            timestamp=datetime.utcnow(),
            metadata={
                "agents_used": response["data"]["agents_used"],
                "confidence": response["data"]["confidence"]
            }
        )
        conversation.messages.append(assistant_message)
        conversation.updated_at = datetime.utcnow()
        
        # Save updated conversation
        await mongodb_connector.save_conversation(conversation.dict())
        await redis_connector.save_conversation(conversation.id, conversation.dict(), ttl=3600)
        
        # Format response
        return ConversationResponse(
            conversation_id=conversation.id,
            message=response["data"]["response"],
            agents_used=response["data"]["agents_used"],
            confidence=response["data"]["confidence"],
            suggestions=response["data"].get("suggestions", []),
            metadata=response.get("metadata", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{conversation_id}", response_model=Conversation)
async def get_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get conversation history"""
    try:
        # Try Redis first
        conversation_data = await redis_connector.get_conversation(conversation_id)
        
        if not conversation_data:
            # Fallback to MongoDB
            conversation_data = await mongodb_connector.get_conversation(conversation_id)
            
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Verify ownership
        if conversation_data["user_id"] != current_user["user_id"] and current_user["role"] != "God":
            raise HTTPException(status_code=403, detail="Access denied")
        
        return Conversation(**conversation_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/", response_model=Dict[str, Any])
async def list_conversations(
    project_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List user's conversations"""
    try:
        filters = {"user_id": current_user["user_id"]}
        if project_id:
            filters["project_id"] = project_id
        
        conversations = await mongodb_connector.find_conversations(
            filters=filters,
            limit=limit,
            skip=offset
        )
        
        return {
            "conversations": conversations,
            "total": len(conversations),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"List conversations error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a conversation"""
    try:
        # Get conversation to verify ownership
        conversation_data = await mongodb_connector.get_conversation(conversation_id)
        
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Verify ownership
        if conversation_data["user_id"] != current_user["user_id"] and current_user["role"] != "God":
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete from both stores
        await mongodb_connector.delete_conversation(conversation_id)
        await redis_connector.delete(f"conversation:{conversation_id}")
        
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete conversation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{conversation_id}/feedback")
async def add_feedback(
    conversation_id: str,
    feedback: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Add feedback to a conversation"""
    try:
        # Get conversation
        conversation_data = await mongodb_connector.get_conversation(conversation_id)
        
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Verify ownership
        if conversation_data["user_id"] != current_user["user_id"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Add feedback to last assistant message
        if conversation_data["messages"]:
            for msg in reversed(conversation_data["messages"]):
                if msg["role"] == "assistant":
                    msg["feedback"] = {
                        "rating": feedback.get("rating"),
                        "comment": feedback.get("comment"),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    break
        
        # Save updated conversation
        await mongodb_connector.save_conversation(conversation_data)
        
        return {"message": "Feedback added successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add feedback error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")