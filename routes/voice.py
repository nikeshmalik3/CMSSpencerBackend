"""Voice and NLU routes"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, WebSocket
from typing import Dict, Any, Optional
import base64
import logging
from datetime import datetime

from agents.voice_nlu import voice_nlu
from middleware.auth import get_current_user, verify_websocket_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])

@router.post("/transcribe", response_model=Dict[str, Any])
async def transcribe_audio(
    audio: UploadFile = File(...),
    noise_level: str = "normal",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Transcribe audio file to text"""
    try:
        # Read audio file
        audio_content = await audio.read()
        audio_base64 = base64.b64encode(audio_content).decode()
        
        # Detect audio format from filename
        audio_format = audio.filename.split('.')[-1].lower()
        if audio_format not in ['wav', 'mp3', 'ogg', 'flac', 'm4a']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {audio_format}"
            )
        
        # Build request
        request = {
            "audio_data": audio_base64,
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"]
            },
            "parameters": {
                "format": audio_format,
                "noise_level": noise_level,
                "duration": len(audio_content) / (16000 * 2)  # Approximate duration
            }
        }
        
        # Process through voice agent
        result = await voice_nlu.process(request)
        
        return {
            "status": "success",
            "transcription": result["data"]["original_text"],
            "confidence": result["data"]["confidence"],
            "metadata": result["metadata"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Transcription failed")

@router.post("/understand", response_model=Dict[str, Any])
async def understand_text(
    nlu_request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Process text for natural language understanding"""
    try:
        # Build request
        request = {
            "text": nlu_request.get("text", ""),
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"],
                "conversation_id": nlu_request.get("conversation_id")
            },
            "parameters": nlu_request.get("parameters", {})
        }
        
        # Process through voice/NLU agent
        result = await voice_nlu.process(request)
        
        return {
            "status": "success",
            "understanding": {
                "intent": result["data"]["intent"],
                "entities": result["data"]["entities"],
                "confidence": result["data"]["confidence"],
                "interpretation": result["data"]["interpretation"],
                "suggested_actions": result["data"]["suggested_actions"]
            },
            "requires_clarification": result["data"]["requires_clarification"]
        }
        
    except Exception as e:
        logger.error(f"NLU error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Understanding failed")

@router.post("/voice-command", response_model=Dict[str, Any])
async def process_voice_command(
    audio: UploadFile = File(...),
    execute: bool = True,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Process voice command and optionally execute action"""
    try:
        # First transcribe
        audio_content = await audio.read()
        audio_base64 = base64.b64encode(audio_content).decode()
        
        # Process voice input
        voice_request = {
            "audio_data": audio_base64,
            "context": {
                "user_id": current_user["user_id"],
                "user_role": current_user["role"]
            },
            "parameters": {
                "format": audio.filename.split('.')[-1].lower()
            }
        }
        
        voice_result = await voice_nlu.process(voice_request)
        
        response = {
            "transcription": voice_result["data"]["original_text"],
            "intent": voice_result["data"]["intent"],
            "entities": voice_result["data"]["entities"],
            "confidence": voice_result["data"]["confidence"]
        }
        
        # Execute command if requested and confidence is high
        if execute and voice_result["data"]["confidence"] > 0.7:
            # Import master orchestrator for execution
            from agents.master_orchestrator import master_orchestrator
            
            # Execute through orchestrator
            exec_request = {
                "query": voice_result["data"]["structured_query"],
                "context": {
                    "user_id": current_user["user_id"],
                    "user_role": current_user["role"],
                    "source": "voice"
                },
                "nlu_data": voice_result["data"]
            }
            
            exec_result = await master_orchestrator.process(exec_request)
            
            response["execution"] = {
                "status": "executed",
                "result": exec_result["data"]["response"],
                "agents_used": exec_result["data"]["agents_used"]
            }
        else:
            response["execution"] = {
                "status": "not_executed",
                "reason": "Low confidence" if voice_result["data"]["confidence"] <= 0.7 else "Execution disabled"
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Voice command error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Voice command processing failed")

@router.websocket("/stream")
async def stream_transcription(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming and transcription"""
    await websocket.accept()
    
    try:
        # Verify authentication
        token = websocket.headers.get("Authorization", "").replace("Bearer ", "")
        user = await verify_websocket_token(token)
        
        if not user:
            await websocket.close(code=1008, reason="Unauthorized")
            return
        
        # Create context
        context = {
            "user_id": user["user_id"],
            "user_role": user["role"]
        }
        
        # Audio buffer for streaming
        audio_buffer = bytearray()
        
        while True:
            # Receive audio chunk
            data = await websocket.receive_bytes()
            
            if not data:
                break
            
            audio_buffer.extend(data)
            
            # Process when we have enough audio (e.g., 1 second)
            if len(audio_buffer) >= 16000 * 2:  # 16kHz, 16-bit audio
                # Create async generator for streaming
                async def audio_generator():
                    yield bytes(audio_buffer)
                    audio_buffer.clear()
                
                # Process streaming audio
                async for result in voice_nlu.process_streaming_audio(
                    audio_generator(),
                    context
                ):
                    await websocket.send_json(result)
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.close(code=1011, reason="Internal error")

@router.get("/intents", response_model=Dict[str, Any])
async def list_supported_intents(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """List supported voice intents"""
    try:
        intents = []
        
        for intent_name, intent_config in voice_nlu.intent_patterns.items():
            intents.append({
                "name": intent_name,
                "patterns": intent_config["patterns"],
                "required_entities": intent_config["required_entities"],
                "optional_entities": intent_config.get("optional_entities", []),
                "examples": [
                    "Create an order for 50 tons of concrete",
                    "Check status of order #12345",
                    "Schedule safety inspection for tomorrow"
                ] if intent_name == "create_order" else []
            })
        
        return {
            "intents": intents,
            "total": len(intents)
        }
        
    except Exception as e:
        logger.error(f"List intents error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list intents")

@router.post("/train-vocabulary", response_model=Dict[str, Any])
async def train_custom_vocabulary(
    vocabulary: Dict[str, List[str]],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Add custom vocabulary for better recognition (admin only)"""
    try:
        # Check permissions
        if current_user["role"] not in ["admin", "God"]:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Update vocabulary
        for category, terms in vocabulary.items():
            if category in voice_nlu.construction_vocabulary:
                voice_nlu.construction_vocabulary[category].extend(terms)
            else:
                voice_nlu.construction_vocabulary[category] = terms
        
        # Update STT engine keywords
        all_keywords = []
        for terms in voice_nlu.construction_vocabulary.values():
            all_keywords.extend(terms)
        
        voice_nlu.stt_engines["config"]["deepgram"]["keywords"] = all_keywords[:100]  # Deepgram limit
        
        return {
            "message": "Vocabulary updated successfully",
            "categories_updated": list(vocabulary.keys()),
            "total_terms": sum(len(terms) for terms in vocabulary.values())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Train vocabulary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update vocabulary")