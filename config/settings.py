import os
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Configuration
    API_BASE_URL = os.getenv("CMS_API_BASE_URL", "https://test.companiesms.co.uk")
    API_VERSION = "v1"
    API_BEARER_TOKEN = os.getenv("CMS_API_TOKEN")  # Required from .env
    
    # For backwards compatibility
    BEARER_TOKEN = API_BEARER_TOKEN
    
    # User Configuration (from auth response)
    USER_ID = 3392
    CLIENT_ID = 5
    USER_ROLE = "God"
    
    # Server Configuration
    HOST = os.getenv("SPENCER_HOST", "0.0.0.0")
    PORT = int(os.getenv("SPENCER_PORT", "8888"))
    DEBUG = os.getenv("SPENCER_DEBUG", "False").lower() == "true"
    
    # Redis Configuration
    REDIS_URL = os.getenv("REDIS_URL")  # Full URL with password
    REDIS_HOST = os.getenv("REDIS_HOST", "157.180.62.92")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "5599"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")  # From .env
    REDIS_USE_SSL = False  # No SSL for internal VPS
    REDIS_POOL_SIZE = 50
    
    # MongoDB Configuration
    MONGODB_URI = os.getenv("MONGODB_URI")  # Required from .env
    MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "spencer_ai")
    MONGODB_MAX_POOL_SIZE = 100
    MONGODB_MIN_POOL_SIZE = 10
    
    # Vector Service Configuration (Replaces FAISS)
    VECTOR_SERVICE_URL = os.getenv("VECTOR_SERVICE_URL", "http://157.180.62.92:8001")
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/data/spencer-ai-storage/faiss_indexes")
    EMBEDDING_MODEL = "mxbai-embed-large"
    EMBEDDING_DIMENSION = 1024
    
    # Ollama Configuration (for embeddings)
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://157.180.62.92:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mxbai-embed-large")
    
    # OpenRouter Configuration (for Gemini 2.5 Flash)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Required from .env
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    AI_MODEL = os.getenv("AI_MODEL", "google/gemini-flash-2.5")
    
    # Groq Configuration (Whisper Speech-to-Text)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Required from .env
    GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
    GROQ_TIMEOUT = 30
    GROQ_COST_PER_HOUR = 0.04
    
    # Spencer File Manager Configuration
    SPENCER_FILE_MANAGER_URL = os.getenv("SPENCER_FILE_MANAGER_URL", "http://157.180.62.92:8002")
    FILEBROWSER_URL = os.getenv("FILEBROWSER_URL", "http://157.180.62.92:8090")
    
    # Document Processing Configuration
    DOCUMENT_PROCESSING = {
        "pdf_to_markdown": True,
        "ocr_provider": "gemini",
        "max_pages_per_request": 1500,
        "cost_per_1000_pages": 0.167
    }
    
    # File Storage Configuration
    FILE_STORAGE_BASE_PATH = os.getenv("FILE_STORAGE_PATH", "/data/spencer-ai-storage")
    UPLOAD_DIR = os.path.join(FILE_STORAGE_BASE_PATH, "uploads")
    MAX_FILE_SIZE_MB = 100
    ALLOWED_EXTENSIONS = [
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx",
        "jpg", "jpeg", "png", "gif", "bmp", "svg",
        "mp3", "wav", "m4a", "ogg",
        "mp4", "avi", "mov",
        "json", "xml", "yaml", "csv", "txt",
        "dwg", "dxf"
    ]
    
    # Storage Retention Policies (in days)
    RETENTION_POLICIES = {
        "uploads/temp": 1,
        "outputs/downloads": 7,
        "cache": 0.042,
        "logs/access": 30,
        "logs/error": 90,
        "logs/audit": 2555,
        "backup/daily": 7,
        "backup/weekly": 28,
        "backup/monthly": 365
    }
    
    # Request Settings
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "60"))  # seconds
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS Settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_CREDENTIALS = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"
    CORS_METHODS = os.getenv("CORS_METHODS", "*").split(",") if os.getenv("CORS_METHODS") != "*" else ["*"]
    CORS_HEADERS = os.getenv("CORS_HEADERS", "*").split(",") if os.getenv("CORS_HEADERS") != "*" else ["*"]
    
    @classmethod
    def get_api_headers(cls) -> Dict[str, str]:
        """Get headers for API requests"""
        return {
            "Authorization": f"Bearer {cls.API_BEARER_TOKEN}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Spencer-AI/1.0"
        }
    
    @classmethod
    def get_redis_url(cls) -> str:
        """Get Redis connection URL"""
        if cls.REDIS_URL:
            return cls.REDIS_URL
        
        if cls.REDIS_PASSWORD:
            return f"redis://:{cls.REDIS_PASSWORD}@{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
        return f"redis://{cls.REDIS_HOST}:{cls.REDIS_PORT}/{cls.REDIS_DB}"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate required configuration"""
        required = {
            "API_BEARER_TOKEN": cls.API_BEARER_TOKEN,
            "MONGODB_URI": cls.MONGODB_URI,
            "OPENROUTER_API_KEY": cls.OPENROUTER_API_KEY,
            "GROQ_API_KEY": cls.GROQ_API_KEY,
        }
        
        missing = [key for key, value in required.items() if not value]
        
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}. Please check your .env file.")
        
        return True

# Create config instance
config = Config()

# Validate on import (only in production)
if not config.DEBUG:
    try:
        config.validate_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Please copy .env.example to .env and fill in your API keys.")