# Spencer AI CMS Backend - Production Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for all packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    libffi-dev \
    libssl-dev \
    python3-dev \
    libmupdf-dev \
    mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /data/spencer-ai-storage

# Expose port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Run application
CMD ["python", "main.py"]