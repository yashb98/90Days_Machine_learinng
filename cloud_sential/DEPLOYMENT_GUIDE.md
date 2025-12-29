# Cloud Sentinel - Google Cloud Deployment Guide

## Required Services for Complete Application

### 1. **Backend Service** (Essential)
- **Purpose**: Main API server, RAG processing, policy management
- **Ports**: 8000
- **Dependencies**: MCP Server, Redis
- **Why Required**: Frontend cannot function without API

### 2. **MCP Server** (Essential)  
- **Purpose**: AWS audit tools, security operations
- **Ports**: 8001
- **Dependencies**: AWS credentials
- **Why Required**: Backend chat functionality depends on this

### 3. **Frontend Service** (Essential)
- **Purpose**: User interface, React application
- **Ports**: 80 (via nginx)
- **Dependencies**: Backend API
- **Why Required**: User interface for the application

### 4. **Redis Service** (Important)
- **Purpose**: Caching, session management
- **Ports**: 6379
- **Dependencies**: None
- **Why Important**: Performance and data persistence

## Deployment Order

### Option A: Complete Stack Deployment (Recommended)
Deploy all services in dependency order:

```bash
# 1. Deploy Infrastructure Services First
gcloud run deploy redis-service --image redis:7-alpine

# 2. Deploy Backend Service  
gcloud run deploy cloud-sentinel-backend --source ./backend

# 3. Deploy MCP Server
gcloud run deploy cloud-sentinel-mcp --source ./mcp-server

# 4. Deploy Frontend (build first)
cd frontend && npm run build
gcloud run deploy cloud-sentinel-frontend --source .
```

### Option B: Docker Compose on Cloud Run
```bash
# Build and push all services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build
docker-compose -f docker-compose.yml -f docker-compose.prod.yml push

# Deploy as Cloud Run service with multiple containers
gcloud run deploy cloud-sentinel-app --source .
```

## Environment Variables Required

### Backend Service
- `GOOGLE_API_KEY` - Gemini API key
- `PINECONE_API_KEY` - Vector database
- `PINECONE_INDEX` - Vector index name
- `CORS_ORIGINS` - Frontend URLs
- `MCP_SERVER_URL` - MCP server endpoint

### MCP Server  
- `AWS_ACCESS_KEY_ID` - AWS credentials
- `AWS_SECRET_ACCESS_KEY` - AWS credentials
- `AWS_REGION` - AWS region

### Frontend Service
- `VITE_API_URL` - Backend API endpoint
- `VITE_MCP_URL` - MCP server endpoint

### Redis Service
- `REDIS_PASSWORD` - Redis authentication

## Quick Start Commands

```bash
# Deploy all services at once
./deploy-all.sh

# Or deploy individually in order
gcloud run deploy redis --image redis:7-alpine
gcloud run deploy backend --source ./backend  
gcloud run deploy mcp-server --source ./mcp-server
gcloud run deploy frontend --source ./frontend
```

## Why Frontend-Only Won't Work

| Feature | Backend Required | MCP Required | Result |
|---------|------------------|--------------|---------|
| User Login | ✅ | ❌ | Broken without backend |
| Policy Display | ✅ | ❌ | Shows error messages |
| Chat Interface | ✅ | ✅ | Completely non-functional |
| Document Upload | ✅ | ❌ | Upload fails |
| AI Responses | ✅ | ✅ | No responses |
| Security Audit | ✅ | ✅ | No audit tools |

## Deployment Checklist

- [ ] Set up Google Cloud Project
- [ ] Enable required APIs (Cloud Run, Container Registry)
- [ ] Configure environment variables
- [ ] Deploy Redis service
- [ ] Deploy Backend service  
- [ ] Deploy MCP Server
- [ ] Build and deploy Frontend
- [ ] Configure domain and SSL
- [ ] Test end-to-end functionality

## Recommended Approach

**Deploy the complete stack** to ensure all features work:
1. **Backend** + **MCP Server** + **Redis** + **Frontend**
2. This gives you the full Cloud Sentinel experience
3. All error fixes will be active and working
4. Production-ready with proper error handling
