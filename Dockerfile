# Spencer AI CMS Backend - Production Dockerfile
FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install one by one to identify issues
COPY requirements.txt .

# Install core dependencies first
RUN pip install fastapi==0.109.0 uvicorn==0.27.0 python-multipart==0.0.6

# Install async dependencies
RUN pip install aiohttp==3.9.1 aiofiles==23.2.1

# Install database drivers
RUN pip install motor==3.3.2 pymongo==4.6.1 redis==5.0.1

# Install AI/ML dependencies
RUN pip install openai==1.6.1 groq==0.11.0

# Install remaining dependencies
RUN pip install PyJWT==2.8.0 python-dotenv==1.0.0 pydantic==2.5.3 email-validator==2.1.0 orjson==3.9.10

# Try to install optional dependencies (if they fail, continue)
RUN pip install tiktoken==0.5.2 || true
RUN pip install numpy==1.24.3 || true
RUN pip install PyMuPDF==1.23.8 || true
RUN pip install pymupdf4llm==0.0.9 || true
RUN pip install openpyxl==3.1.2 || true
RUN pip install python-docx==1.1.0 || true
RUN pip install ezdxf==1.1.3 || true
RUN pip install Pillow==10.1.0 || true
RUN pip install python-jose[cryptography]==3.3.0 || true
RUN pip install passlib[bcrypt]==1.7.4 || true
RUN pip install pydantic-settings==2.1.0 || true
RUN pip install structlog==24.1.0 || true
RUN pip install pyyaml==6.0.1 || true
RUN pip install uvloop==0.19.0 || true

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/logs /data/spencer-ai-storage

# Expose port
EXPOSE 8888

# Run application
CMD ["python", "main.py"]