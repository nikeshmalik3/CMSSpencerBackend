"""
Embedding Service - Uses YOUR Ollama server with mxbai-embed-large
Connects to your VPS Ollama instance
"""

import logging
from typing import List, Optional
import numpy as np
import aiohttp
import json
from config.settings import config

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using Ollama mxbai-embed-large on VPS"""
    
    def __init__(self):
        # YOUR OLLAMA SERVER ON VPS
        self.ollama_url = config.OLLAMA_URL if hasattr(config, 'OLLAMA_URL') else "http://157.180.62.92:11434"
        self.model_name = "mxbai-embed-large"  # Your hosted model
        self.dimension = 1024  # mxbai-embed-large dimension
        self.session = None
        
    async def initialize(self):
        """Initialize connection to Ollama server"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Test connection to Ollama
            async with self.session.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model_name, "prompt": "test"}
            ) as response:
                if response.status == 200:
                    logger.info(f"Connected to Ollama at {self.ollama_url} with model {self.model_name}")
                    return True
                else:
                    logger.error(f"Ollama server returned status {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            return False
    
    async def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding using Ollama API"""
        try:
            if not self.session:
                await self.initialize()
            
            async with self.session.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    embedding = np.array(data['embedding'])
                    return embedding
                else:
                    logger.error(f"Ollama embedding failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def generate_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = await self.generate_embedding(text)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                logger.warning(f"Failed to generate embedding for text: {text[:50]}...")
                embeddings.append(np.zeros(self.dimension))  # Fallback zero vector
        
        return np.array(embeddings) if embeddings else None
    
    async def is_available(self) -> bool:
        """Check if Ollama service is available"""
        return await self.initialize()
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

# Singleton instance
embedding_service = EmbeddingService()