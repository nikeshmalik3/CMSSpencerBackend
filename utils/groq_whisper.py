"""
Groq Whisper Integration - Ultra-fast speech-to-text
Using official Groq SDK for production-ready voice processing
"""

import os
import logging
from typing import Optional, Dict, Any, BinaryIO
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from groq import Groq
from config.settings import config

logger = logging.getLogger(__name__)

class GroqWhisperClient:
    """Client for Groq's Whisper API - 216x faster than real-time"""
    
    def __init__(self):
        self.api_key = config.GROQ_API_KEY
        self.model = config.GROQ_WHISPER_MODEL  # whisper-large-v3-turbo
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        
        # Thread pool for sync operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"Groq Whisper initialized with model: {self.model}")
    
    async def transcribe_audio(
        self,
        audio_file: BinaryIO,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "verbose_json"
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text using Groq Whisper
        
        Args:
            audio_file: Audio file object (mp3, wav, etc.)
            language: Optional language code (e.g., 'en', 'es')
            prompt: Optional context prompt (max 224 tokens)
            response_format: json, text, or verbose_json
            
        Returns:
            Transcription result with text and metadata
        """
        try:
            # Run synchronous Groq API call in thread pool
            loop = asyncio.get_event_loop()
            
            def _transcribe():
                """Synchronous transcription call"""
                params = {
                    "file": audio_file,
                    "model": self.model,
                    "response_format": response_format
                }
                
                # Add optional parameters
                if language:
                    params["language"] = language
                if prompt:
                    params["prompt"] = prompt
                
                # Add timestamp granularities for detailed output
                if response_format == "verbose_json":
                    params["timestamp_granularities"] = ["word", "segment"]
                
                return self.client.audio.transcriptions.create(**params)
            
            # Execute in thread pool
            result = await loop.run_in_executor(self.executor, _transcribe)
            
            # Parse response based on format
            if response_format == "text":
                return {
                    "success": True,
                    "text": result,
                    "format": "text"
                }
            else:
                # For json and verbose_json formats
                return {
                    "success": True,
                    "text": result.text,
                    "language": result.language if hasattr(result, 'language') else language,
                    "duration": result.duration if hasattr(result, 'duration') else None,
                    "segments": result.segments if hasattr(result, 'segments') else None,
                    "words": result.words if hasattr(result, 'words') else None,
                    "metadata": {
                        "avg_logprob": result.avg_logprob if hasattr(result, 'avg_logprob') else None,
                        "compression_ratio": result.compression_ratio if hasattr(result, 'compression_ratio') else None,
                        "no_speech_prob": result.no_speech_prob if hasattr(result, 'no_speech_prob') else None
                    }
                }
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    async def translate_audio(
        self,
        audio_file: BinaryIO,
        prompt: Optional[str] = None,
        response_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Translate audio to English using Groq Whisper
        
        Args:
            audio_file: Audio file object
            prompt: Optional context prompt
            response_format: json or text
            
        Returns:
            Translation result
        """
        try:
            loop = asyncio.get_event_loop()
            
            def _translate():
                """Synchronous translation call"""
                params = {
                    "file": audio_file,
                    "model": self.model,
                    "response_format": response_format
                }
                
                if prompt:
                    params["prompt"] = prompt
                
                return self.client.audio.translations.create(**params)
            
            result = await loop.run_in_executor(self.executor, _translate)
            
            if response_format == "text":
                return {
                    "success": True,
                    "text": result,
                    "language": "en"
                }
            else:
                return {
                    "success": True,
                    "text": result.text,
                    "language": "en",
                    "source_language": result.language if hasattr(result, 'language') else "unknown"
                }
                
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    async def transcribe_file_path(
        self,
        file_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convenience method to transcribe from file path
        
        Args:
            file_path: Path to audio file
            language: Optional language code
            prompt: Optional context prompt
            
        Returns:
            Transcription result
        """
        try:
            path = Path(file_path)
            
            # Check file exists and size
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            # Check file size (max 25MB)
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > 25:
                raise ValueError(f"File too large: {file_size_mb:.1f}MB (max 25MB)")
            
            # Open and transcribe
            with open(file_path, "rb") as audio_file:
                return await self.transcribe_audio(
                    audio_file,
                    language=language,
                    prompt=prompt
                )
                
        except Exception as e:
            logger.error(f"Failed to transcribe file {file_path}: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    async def process_streaming_audio(
        self,
        audio_stream: AsyncIterator[bytes],
        language: Optional[str] = None
    ) -> AsyncIterator[str]:
        """
        Process streaming audio (collect chunks and transcribe)
        Note: Groq doesn't support real-time streaming, so we collect chunks
        
        Args:
            audio_stream: Async iterator of audio bytes
            language: Optional language code
            
        Yields:
            Transcribed text chunks
        """
        # Collect audio chunks (e.g., every 5 seconds)
        chunk_duration = 5  # seconds
        chunk_size = 16000 * 2 * chunk_duration  # 16kHz, 16-bit mono
        
        audio_buffer = bytearray()
        
        async for audio_chunk in audio_stream:
            audio_buffer.extend(audio_chunk)
            
            # When buffer is large enough, transcribe
            if len(audio_buffer) >= chunk_size:
                # Create temporary file-like object
                from io import BytesIO
                audio_file = BytesIO(bytes(audio_buffer))
                audio_file.name = "stream.wav"  # Groq needs a filename
                
                # Transcribe chunk
                result = await self.transcribe_audio(
                    audio_file,
                    language=language,
                    response_format="text"
                )
                
                if result["success"] and result["text"]:
                    yield result["text"]
                
                # Clear buffer
                audio_buffer.clear()
        
        # Process remaining audio
        if audio_buffer:
            from io import BytesIO
            audio_file = BytesIO(bytes(audio_buffer))
            audio_file.name = "stream.wav"
            
            result = await self.transcribe_audio(
                audio_file,
                language=language,
                response_format="text"
            )
            
            if result["success"] and result["text"]:
                yield result["text"]
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
            "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
            "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
            "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
            "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
            "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
            "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
            "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
            "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
            "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]
    
    def close(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=False)
        logger.info("Groq Whisper client closed")

# Global instance
groq_whisper = GroqWhisperClient()