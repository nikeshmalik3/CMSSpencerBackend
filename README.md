# 🚀 Spencer AI CMS Backend

Production backend for Spencer AI - The intelligent CMS assistant with 11 specialized agents.

**Live API:** https://spencerai-backend.cmsdeskai.com

## 📋 Features

- **11 Specialized AI Agents** for different tasks
- **Voice Processing** with Groq Whisper (216x real-time)
- **Document OCR** with Gemini 2.5 Flash (PDFs, Excel, Word, CAD)
- **Real-time Analytics** from CMS data
- **Semantic Search** using FAISS/Vector Service
- **52 CMS API Endpoints** integrated
- **Rate Limiting** (100 req/60s configurable)
- **MongoDB & Redis** for data persistence

## 🌐 API Endpoints

Base URL: `https://spencerai-backend.cmsdeskai.com`

### Core APIs
- `POST /api/v1/conversation/message` - Chat with AI
- `POST /api/v1/documents/upload` - Process documents
- `POST /api/v1/voice/transcribe` - Speech to text
- `POST /api/v1/analytics/query` - Get analytics
- `POST /api/v1/search` - Semantic search
- `GET /health` - Health check
- `GET /docs` - Swagger documentation

## 🐳 Deployment with Coolify

1. **Add to Coolify:**
   - Repository: `https://github.com/nikeshmalik3/CMS-Spencer-Backend.git`
   - Branch: `main`
   - Build Pack: `Docker Compose`

2. **Set Environment Variables in Coolify:**
```env
# Required API Keys
CMS_API_TOKEN=your_cms_bearer_token
OPENROUTER_API_KEY=your_openrouter_key
GROQ_API_KEY=your_groq_key

# MongoDB (already configured for your VPS)
MONGODB_URI=mongodb://root:password@157.180.62.92:5567/default
MONGODB_DATABASE=spencer_ai

# Redis (already configured for your VPS)
REDIS_URL=redis://:password@157.180.62.92:5599/0
REDIS_PASSWORD=your_redis_password

# Your VPS Services (already running)
VECTOR_SERVICE_URL=http://157.180.62.92:8001
OLLAMA_URL=http://157.180.62.92:11434
SPENCER_FILE_MANAGER_URL=http://157.180.62.92:8002
```

3. **Deploy:**
   - Click Deploy in Coolify
   - Backend will be available at: `https://spencerai-backend.cmsdeskai.com`

## 🚀 Local Development

1. **Clone the repository:**
```bash
git clone https://github.com/nikeshmalik3/CMS-Spencer-Backend.git
cd CMS-Spencer-Backend
```

2. **Create `.env` file:**
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run locally:**
```bash
python main.py
```

Backend will be available at: `http://localhost:8888`

## 📚 API Documentation

Once deployed, access the interactive API docs:
- Swagger UI: https://spencerai-backend.cmsdeskai.com/docs
- ReDoc: https://spencerai-backend.cmsdeskai.com/redoc

## 🧪 Testing the API

```bash
# Health Check
curl https://spencerai-backend.cmsdeskai.com/health

# Chat with AI
curl -X POST https://spencerai-backend.cmsdeskai.com/api/v1/conversation/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Spencer!"}'

# Upload Document
curl -X POST https://spencerai-backend.cmsdeskai.com/api/v1/documents/upload \
  -F "file=@document.pdf"
```

## 🏗️ Architecture

```
Spencer AI Backend
├── 11 AI Agents (Master, API, Analytics, Voice, etc.)
├── FastAPI REST APIs
├── MongoDB (Data Storage)
├── Redis (Cache & Rate Limiting)
├── Vector Service (Semantic Search)
├── Ollama (Embeddings)
├── Groq Whisper (Voice)
├── Gemini 2.5 Flash (OCR)
└── Spencer File Manager (File Storage)
```

## 🔒 Security

- All API keys stored in environment variables
- Rate limiting enabled (100 req/60s)
- CORS configured for production
- `.env` file never committed to Git

## 📦 Docker Support

```bash
# Build and run with Docker
docker-compose up --build

# Or use Docker directly
docker build -t spencer-backend .
docker run -p 8888:8888 --env-file .env spencer-backend
```

## 🛠️ Tech Stack

- **Framework:** FastAPI (Python 3.11)
- **AI Models:** Gemini 2.5 Flash, Groq Whisper, mxbai-embed-large
- **Databases:** MongoDB, Redis
- **File Storage:** Spencer File Manager API
- **Deployment:** Docker, Coolify
- **Domain:** spencerai-backend.cmsdeskai.com

## 📝 License

Proprietary - CMS Desk AI © 2024

## 👨‍💻 Author

**Nikesh Malik**  
CMS Desk AI  
GitHub: [@nikeshmalik3](https://github.com/nikeshmalik3)

---

**Production URL:** https://spencerai-backend.cmsdeskai.com  
**API Docs:** https://spencerai-backend.cmsdeskai.com/docs