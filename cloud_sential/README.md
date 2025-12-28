# CloudSentinel - Containerized Application

## 🚀 Quick Start

Get CloudSentinel running in under 5 minutes:

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- Git

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd cloud_sential
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### 3. Start Services
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### 4. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **MCP Server**: http://localhost:8001

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   MCP Server    │
│   (React/Nginx) │◄──►│  (FastAPI)      │◄──►│  (FastMCP)      │
│   Port: 3000    │    │  Port: 8000     │    │  Port: 8001     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │     Redis       │
                       │   (Cache)       │
                       │   Port: 6379    │
                       └─────────────────┘
```

## 📁 Project Structure

```
cloud_sential/
├── backend/                    # Python FastAPI Backend
│   ├── Dockerfile             # Multi-stage Python build
│   ├── requirements.txt       # Python dependencies
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── agent.py          # Security agent with MCP integration
│   │   └── models.py         # Data models
│   └── rag/                  # RAG implementation
│       ├── ingest.py         # Document processing
│       ├── services.py       # Vector database services
│       └── interfaces.py     # Service interfaces
├── mcp-server/                # MCP Server for AWS auditing
│   ├── Dockerfile           # Multi-stage Python build
│   ├── requirements.txt     # Python dependencies
│   ├── server.py            # FastMCP server
│   └── tools/               # AWS audit tools
│       └── aws_audit.py     # S3 compliance checking
├── frontend/                  # React TypeScript Frontend
│   ├── Dockerfile           # Multi-stage Node.js + Nginx build
│   ├── nginx.conf           # Nginx configuration
│   ├── package.json         # Node.js dependencies
│   ├── src/                 # React source code
│   │   ├── api/             # API clients
│   │   ├── components/      # React components
│   │   ├── hooks/           # Custom React hooks
│   │   └── services/        # Business logic
├── docker-compose.yml         # Development orchestration
├── docker-compose.prod.yml    # Production configuration
├── test-containerization.sh   # Automated testing script
└── .env.example              # Environment variables template
```

## 🐳 Docker Configuration

### Development Setup
```bash
# Start all services in development mode
docker-compose up -d

# View logs
docker-compose logs -f [service_name]

# Scale backend for load testing
docker-compose up -d --scale backend=3
```

### Production Setup
```bash
# Use production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Or with environment-specific config
docker-compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.prod.aws.yml up -d
```

## 🔧 Environment Variables

### Required Variables
```env
# Google AI Configuration
GOOGLE_API_KEY=your_google_generative_ai_api_key

# Pinecone Vector Database
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_pinecone_index_name

# AWS Configuration (for MCP Server)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1

# Redis Configuration
REDIS_PASSWORD=secure_redis_password

# CORS Configuration
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Optional Variables
```env
# Application Settings
DEBUG=false
LOG_LEVEL=INFO
MCP_SERVER_URL=http://mcp-server:8001/sse

# Frontend Configuration
VITE_API_URL=http://localhost:8000
VITE_MCP_URL=http://localhost:8001
```

## 🧪 Testing

### Automated Testing
```bash
# Run comprehensive test suite
./test-containerization.sh
```

### Manual Testing
```bash
# Test all services are running
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:80/health

# Test backend API
curl http://localhost:8000/policies

# Test chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "history": []}'
```

### Service Status
```bash
# Check all services
docker-compose ps

# View service logs
docker-compose logs backend
docker-compose logs mcp-server
docker-compose logs frontend
docker-compose logs redis
```

## 🚀 Deployment Options

### Option 1: Local Production
```bash
# Build and start production services
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify deployment
./test-containerization.sh
```

### Option 2: Google Cloud Run (Optional)

**Prerequisites:**
- Google Cloud SDK installed
- Docker Hub account (for image registry)

**Deploy Backend:**
```bash
# Build and push backend image
docker build -t your-registry/cloud-sential-backend:latest ./backend
docker push your-registry/cloud-sential-backend:latest

# Deploy to Cloud Run
gcloud run deploy cloud-sential-backend \
  --image your-registry/cloud-sential-backend:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_API_KEY=your-key,PINECONE_API_KEY=your-key"
```

**Deploy MCP Server:**
```bash
# Build and push MCP server image
docker build -t your-registry/cloud-sential-mcp:latest ./mcp-server
docker push your-registry/cloud-sential-mcp:latest

# Deploy to Cloud Run
gcloud run deploy cloud-sential-mcp \
  --image your-registry/cloud-sential-mcp:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "AWS_ACCESS_KEY_ID=your-key,AWS_SECRET_ACCESS_KEY=your-secret"
```

**Deploy Frontend:**
```bash
# Build and push frontend image
docker build -t your-registry/cloud-sential-frontend:latest ./frontend
docker push your-registry/cloud-sential-frontend:latest

# Deploy to Cloud Run
gcloud run deploy cloud-sential-frontend \
  --image your-registry/cloud-sential-frontend:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "VITE_API_URL=https://your-backend-url"
```

### Option 3: Kubernetes

```yaml
# Example k8s deployment (save as k8s-deployment.yaml)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cloud-sential-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cloud-sential-backend
  template:
    metadata:
      labels:
        app: cloud-sential-backend
    spec:
      containers:
      - name: backend
        image: your-registry/cloud-sential-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: cloud-sential-secrets
              key: google-api-key
---
apiVersion: v1
kind: Service
metadata:
  name: cloud-sential-backend-service
spec:
  selector:
    app: cloud-sential-backend
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

Deploy to Kubernetes:
```bash
kubectl apply -f k8s-deployment.yaml
```

## 🔒 Security Features

- **Non-root containers**: All services run as non-root users
- **Health checks**: Automatic service monitoring
- **Resource limits**: CPU and memory constraints
- **Network isolation**: Custom Docker network
- **Environment variables**: Secure configuration management
- **Image scanning**: Multi-stage builds reduce attack surface

## 📊 Monitoring

### Service Health
```bash
# Monitor all services
watch docker-compose ps

# Real-time logs
docker-compose logs -f --tail=100

# Service-specific monitoring
docker stats cloud_sential_backend cloud_sential_mcp_server cloud_sential_frontend
```

### Log Locations
- Backend: `docker-compose logs backend`
- MCP Server: `docker-compose logs mcp-server`
- Frontend: `docker-compose logs frontend`
- Redis: `docker-compose logs redis`

## 🛠️ Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check Docker daemon
docker info

# Check port conflicts
lsof -i :8000
lsof -i :8001
lsof -i :3000
```

**Backend can't connect to MCP Server:**
```bash
# Check network connectivity
docker-compose exec backend ping mcp-server

# Check MCP server logs
docker-compose logs mcp-server
```

**Frontend can't reach backend:**
```bash
# Check CORS configuration
docker-compose exec frontend curl http://backend:8000/health

# Verify environment variables
docker-compose exec frontend env | grep VITE_API_URL
```

**Redis connection issues:**
```bash
# Test Redis connectivity
docker-compose exec backend redis-cli -h redis -p 6379 ping
```

### Performance Issues
```bash
# Check resource usage
docker stats

# Scale services
docker-compose up -d --scale backend=2 --scale mcp-server=2

# Check disk usage
docker system df
docker system prune
```

## 🔄 Maintenance

### Updates
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Backup
```bash
# Backup Redis data
docker-compose exec redis redis-cli BGSAVE
docker cp cloud_sential_redis:/data/dump.rdb ./backups/redis-backup.rdb

# Backup environment
cp .env ./backups/.env.backup
```

### Cleanup
```bash
# Stop all services
docker-compose down

# Remove unused images and volumes
docker system prune -a --volumes

# Remove named volumes (WARNING: This deletes data)
docker-compose down -v
```

## 📝 API Documentation

### Backend Endpoints
- `GET /health` - Health check
- `GET /policies` - List security policies
- `POST /ingest` - Upload and process documents
- `POST /chat` - AI chat with security analysis

### MCP Server Endpoints
- `GET /health` - Health check
- `POST /sse` - Server-Sent Events for MCP tools
- Tools: `list_s3_buckets`, `audit_bucket_security`, `health_check`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `./test-containerization.sh`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Check the troubleshooting section above
- Review Docker logs: `docker-compose logs [service-name]`
- Open an issue on GitHub
- Contact the development team

---

**🎉 Congratulations!** CloudSentinel is now fully containerized and ready to run anywhere. The "Works on my machine" problem is solved - it now works everywhere!
