# 🛡️ Cloud Sentinel - AI-Powered Cloud Security Platform

[![Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green.svg)](https://github.com/your-repo/cloud-sentinel)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![TypeScript](https://img.shields.io/badge/Frontend-TypeScript-blue.svg)](https://www.typescriptlang.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![PWA](https://img.shields.io/badge/PWA-Enabled-purple.svg)](https://web.dev/progressive-web-apps/)

Cloud Sentinel is a comprehensive, AI-powered cloud security platform that provides real-time security analysis, policy management, and automated compliance checking for cloud infrastructure. Built with modern technologies and containerized for seamless deployment.

## 🚀 Key Features

### 🤖 AI-Powered Security Analysis
- **Intelligent Chat Interface**: Natural language interaction with security AI
- **Real-time Security Analysis**: Instant assessment of cloud security posture
- **Automated Threat Detection**: AI-driven identification of security vulnerabilities
- **Compliance Monitoring**: Continuous compliance checking against security standards
- **Tool Execution Logging**: Real-time display of AI tool executions and results
- **Rate Limiting**: 5 messages per minute to prevent API abuse
- **Conversation History**: Persistent chat history with conversation threading

### 📋 Policy Management System
- **Document Upload & Processing**: Upload security policies via PDF with drag-and-drop
- **Policy Analysis**: AI-powered analysis of security documents
- **Document Processing Pipeline**: 
  - PDF text extraction using LangChain PyPDFLoader
  - Intelligent text chunking (1000 chars with 200 overlap)
  - Batch processing for large documents (50 chunks per batch)
- **Policy Database**: Automatic policy tracking with metadata
- **Compliance Tracking**: Monitor compliance status across multiple frameworks
- **Policy Recommendations**: Automated suggestions for policy improvements
- **File Cleanup**: Temporary file management with automatic cleanup

### 🔍 Advanced Security Tools
- **AWS Security Auditing**: Comprehensive S3 bucket security analysis
- **S3 Bucket Discovery**: List all S3 buckets in AWS account
- **Security Compliance Checks**:
  - Server-Side Encryption (SSE) verification
  - Bucket versioning status validation
  - Public access block configuration verification
- **Infrastructure Scanning**: Automated security assessment of cloud resources
- **Risk Assessment**: AI-driven risk scoring and prioritization
- **Security Reports**: Detailed security reports with actionable insights
- **Violation Reporting**: Automated detection and reporting of security violations
- **Multi-Region Support**: Works across AWS regions

### 🧠 RAG (Retrieval-Augmented Generation) System
- **Vector Embeddings**: Google Gemini 2.0 Flash embeddings (768 dimensions)
- **Semantic Search**: Pinecone vector database for similarity search
- **Policy Knowledge Base**: AI-powered search through uploaded security policies
- **Context-Aware Responses**: AI responses grounded in your specific policies
- **Top-k Retrieval**: Returns top 3 most relevant policy sections
- **Vector Database Integration**: Cloud-hosted Pinecone for semantic search
- **Metadata Storage**: Source, page, and content metadata tracking

### 🔐 Authentication & User Management
- **Clerk Authentication**: Secure user authentication and session management
- **Protected Routes**: Route-based access control
- **User Session Persistence**: Login state persistence across sessions
- **Multi-User Support**: Individual user accounts with separate data
- **Firebase Integration**: Real-time chat message synchronization

### 🎯 User Experience
- **Progressive Web App (PWA)**: Install on any device, works offline
- **Real-time Updates**: Live security status updates and notifications
- **Interactive Dashboard**: Intuitive security metrics and visualizations
- **Mobile-Responsive**: Optimized for desktop, tablet, and mobile devices
- **Terminal-Style UI**: Cyberpunk-inspired dark theme with neon accents
- **Animated Components**: Smooth animations with Framer Motion
- **Custom Scrollbars**: Styled scrollbars matching the terminal theme
- **Chat Interface**: Real-time messaging with message bubbles
- **Tool Log Display**: Expandable logs showing AI tool executions
- **Upload Progress**: Visual feedback for document uploads
- **Status Badges**: Visual indicators for system status

### 🔧 Backend API Features
- **Health Monitoring**: `/health` endpoint for container orchestration
- **Policy Management**: `/policies` endpoint for security policy management
- **Document Ingestion**: `/ingest` endpoint for PDF processing
- **AI Chat**: `/chat` endpoint with rate limiting and tool integration
- **Rate Limiting**: SlowAPI integration for request throttling
- **CORS Protection**: Configurable cross-origin request handling
- **Input Validation**: Pydantic models for request validation
- **Error Handling**: Comprehensive error handling and logging
- **Defensive Programming**: Null checks and data validation

### 🛠️ MCP (Model Context Protocol) Integration
- **AWS S3 Tools**: MCP tools for AWS S3 bucket management
- **Server-Sent Events**: Real-time tool execution via SSE
- **Tool Execution Logging**: Detailed logging of all tool calls
- **Error Recovery**: Graceful handling of tool execution failures
- **List S3 Buckets**: Discover all S3 buckets in the account
- **Audit Bucket Security**: Comprehensive security compliance checking
- **Compliance Reporting**: JSON-formatted security audit reports

### 🗄️ Database & Storage Features
- **Pinecone Vector Database**: Cloud-hosted vector database for semantic search
- **Firestore Integration**: Real-time chat message synchronization
- **In-Memory Policy Database**: Fast access to policy metadata
- **Batch Vector Processing**: Efficient bulk vector insertion
- **Real-time Sync**: Live updates across clients
- **User Isolation**: Separate chat histories per user
- **Timestamp Management**: Automatic timestamp handling

### 🐳 Infrastructure & Deployment
- **Containerization**: Docker multi-stage builds for optimized images
- **Non-root Containers**: Security-first container configuration
- **Resource Limits**: CPU and memory constraints
- **Health Checks**: Container health monitoring
- **Microservices Architecture**: 
  - Backend Service (FastAPI Python application)
  - MCP Server (Tool execution service)
  - Frontend Service (React TypeScript application)
  - Redis Service (Caching and session storage)
- **Deployment Options**:
  - Local development with Docker Compose
  - Production deployment configurations
  - Google Cloud Run deployment scripts
  - Kubernetes deployment ready
  - Horizontal scaling capabilities

### 🔒 Security & Compliance Features
- **Data Protection**: 
  - Encryption in transit (HTTPS for all API communications)
  - Environment variables for secure API key management
  - Rate limiting for API abuse prevention
  - Input sanitization for XSS and injection prevention
- **AWS Security**:
  - S3 compliance auditing with automated security checks
  - Encryption verification for server-side encryption
  - Public access control verification
  - Versioning enforcement compliance
- **Authentication Security**:
  - Enterprise-grade Clerk authentication
  - Secure session handling
  - Route protection for sensitive pages
  - Data separation per user

### 📊 Monitoring & Observability
- **Health Monitoring**:
  - Container health checks with Docker health monitoring
  - Service status with real-time service status monitoring
  - Error logging with comprehensive error tracking
  - Performance metrics with resource usage monitoring
- **Debug Features**:
  - Tool execution logs with detailed tool call logging
  - Error tracing with full error stack traces
  - Debug mode with development debugging capabilities
  - Log aggregation with centralized logging across services

### 🚀 Performance & Optimization
- **Optimization Features**:
  - Batch processing for efficient large document handling
  - Redis caching for session and data caching
  - Lazy loading with optimized frontend loading
  - Code splitting with Vite-based bundle optimization
- **Scalability**:
  - Horizontal scaling with multi-instance deployment support
  - Load balancing with Docker Compose load balancing
  - Database scaling with cloud-hosted database solutions
  - CDN ready with static asset optimization

### 🧪 Error Handling & Quality Assurance
- **Robust Error Handling**:
  - Defensive programming with comprehensive null checks
  - Graceful degradation with fallback for service failures
  - User-friendly errors with clear error messages
  - Recovery mechanisms with automatic retry and recovery
- **Code Quality**:
  - TypeScript type safety throughout frontend
  - Pydantic models for input validation and serialization
  - Docker best practices with secure container configurations
  - Environment management with configuration management

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React + PWA)                 │
│                    ┌─────────────────────────────┐              │
│                    │     User Interface          │              │
│                    │  - Chat Interface          │              │
│                    │  - Policy Dashboard        │              │
│                    │  - Security Analytics      │              │
│                    │  - Upload Interface        │              │
│                    └─────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                                    │ HTTP/WebSocket
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Backend API (FastAPI)                    │
│                    ┌─────────────────────────────┐              │
│                    │    Core Application         │              │
│                    │  - Authentication           │              │
│                    │  - Policy Management        │              │
│                    │  - Chat Processing          │              │
│                    │  - RAG System               │              │
│                    └─────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
┌─────────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│  MCP Server        │  │   Vector DB      │  │     Redis Cache    │
│ (AWS Security)     │  │   (Pinecone)     │  │   (Session Store)   │
│ - S3 Auditing      │  │ - Document Store │  │ - User Sessions    │
│ - Compliance Check │  │ - Embeddings     │  │ - Rate Limiting    │
│ - Risk Assessment  │  │ - Semantic Search│  │ - Performance Cache│
└─────────────────────┘  └──────────────────┘  └─────────────────────┘
```

## 🎯 Core Capabilities

### 1. **Intelligent Security Chat**
- Natural language interaction with AI security expert
- Context-aware responses based on uploaded policies
- Real-time security recommendations and guidance
- Historical conversation tracking and analysis

### 2. **Policy Management & Analysis**
- **Document Upload**: Support for PDF security policies
- **AI Analysis**: Automatic extraction of security requirements
- **Compliance Mapping**: Map policies to security frameworks (SOC2, ISO27001, etc.)
- **Version Control**: Track policy changes and updates

### 3. **AWS Security Auditing**
- **S3 Bucket Analysis**: Comprehensive security assessment
- **IAM Policy Review**: Automated IAM policy analysis
- **Compliance Checking**: Verify against security best practices
- **Risk Scoring**: AI-powered risk assessment and prioritization

### 4. **RAG-Powered Knowledge Base**
- **Document Processing**: Intelligent chunking and embedding
- **Semantic Search**: Find relevant security information instantly
- **Contextual Responses**: AI responses grounded in your security documents
- **Knowledge Expansion**: Continuous learning from new documents

## 🛠️ Technology Stack

### Frontend
- **React 18** with TypeScript for type safety
- **Vite** for fast development and optimized builds
- **Tailwind CSS** for responsive, modern UI
- **Framer Motion** for smooth animations
- **PWA** capabilities with service worker
- **Clerk** for authentication and user management

### Backend
- **FastAPI** for high-performance API development
- **Python 3.11+** for robust backend services
- **Gemini AI** for intelligent security analysis
- **Pinecone** for vector database and semantic search
- **Redis** for caching and session management
- **LangChain** for RAG implementation

### Security & Infrastructure
- **Docker & Docker Compose** for containerization
- **AWS SDK** for cloud security auditing
- **MCP (Model Context Protocol)** for tool integration
- **Rate limiting** and security middleware
- **CORS** and security headers configuration

## 📦 Installation & Setup

### Prerequisites
- Docker 20.10+ and Docker Compose 2.0+
- Node.js 18+ and npm (for development)
- Python 3.11+ (for development)

### Quick Start (Docker)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cloud-sentinel
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   nano .env
   ```

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - **Web Interface**: http://localhost:3000
   - **API Documentation**: http://localhost:8000/docs
   - **MCP Server**: http://localhost:8001

### Development Setup

1. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

2. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **MCP Server Setup**
   ```bash
   cd mcp-server
   pip install -r requirements.txt
   python server.py
   ```

## 🔧 Configuration

### Required Environment Variables

```env
# Google AI Configuration
GOOGLE_API_KEY=your_google_generative_ai_api_key

# Pinecone Vector Database
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=your_pinecone_index_name

# AWS Configuration (for Security Auditing)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1

# Authentication (Clerk)
VITE_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key
CLERK_SECRET_KEY=your_clerk_secret_key

# Redis Configuration
REDIS_PASSWORD=secure_redis_password

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Optional Configuration

```env
# MCP Server Settings
MCP_SERVER_URL=http://mcp-server:8001/sse

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# PWA Settings
VITE_APP_NAME="Cloud Sentinel"
VITE_APP_DESCRIPTION="AI-Powered Cloud Security Platform"
```

## 🚀 Deployment Options

### 1. Local Production Deployment

```bash
# Use production configuration
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify deployment
./test-containerization.sh
```

### 2. Google Cloud Run Deployment

**Complete Stack Deployment:**
```bash
# Make deployment script executable
chmod +x gcloud-deploy.sh

# Deploy all services
./gcloud-deploy.sh
```

**Individual Service Deployment:**
```bash
# Backend
gcloud run deploy cloud-sentinel-backend --source ./backend

# MCP Server
gcloud run deploy cloud-sentinel-mcp --source ./mcp-server

# Frontend
cd frontend && npm run build && gcloud run deploy cloud-sentinel-frontend --source .
```

### 3. Kubernetes Deployment

```bash
# Apply Kubernetes configurations
kubectl apply -f k8s/

# Verify deployment
kubectl get pods
kubectl get services
```

## 📊 API Documentation

### Backend API Endpoints

#### Core Endpoints
- `GET /health` - Health check and system status
- `GET /policies` - List all security policies
- `POST /ingest` - Upload and process security documents
- `POST /chat` - AI-powered security chat interface

#### Policy Management
- `GET /policies/{policy_id}` - Get specific policy details
- `DELETE /policies/{policy_id}` - Remove policy from system
- `GET /policies/{policy_id}/compliance` - Check compliance status

#### Security Analysis
- `POST /analyze/security` - Perform security analysis
- `GET /reports/security` - Generate security reports
- `POST /audit/aws` - Trigger AWS security audit

### MCP Server Endpoints

#### Tool Execution
- `POST /sse` - Server-Sent Events for tool execution
- `GET /tools` - List available security tools
- `POST /tools/aws/s3-audit` - Execute S3 security audit

### Frontend API Integration

```typescript
// Example: Upload policy document
const uploadPolicy = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/api/ingest', {
    method: 'POST',
    body: formData
  });
  
  return response.json();
};

// Example: Security chat
const sendMessage = async (message: string, history: Message[]) => {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history })
  });
  
  return response.json();
};
```

## 🧪 Testing

### Automated Testing
```bash
# Run comprehensive test suite
./test-containerization.sh

# Test individual services
./test-services-individually-fixed.sh

# Performance testing
docker-compose up -d --scale backend=3
```

### Manual Testing
```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:3000/health

# API testing
curl http://localhost:8000/policies
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze S3 bucket security", "history": []}'
```

### Component Testing
```bash
# Frontend component tests
cd frontend && npm test

# Backend API tests
cd backend && python -m pytest tests/

# Integration tests
./test-integration.sh
```

## 🎮 Usage Examples

### 1. **Security Policy Analysis**
```bash
# Upload a security policy document
curl -X POST http://localhost:8000/ingest \
  -F "file=@security-policy.pdf"

# The system will automatically:
# - Extract text from the PDF
# - Chunk the content intelligently
# - Generate embeddings for semantic search
# - Store in vector database for RAG
```

### 2. **AI Security Chat**
```bash
# Ask security questions in natural language
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the key security controls in our current policies?",
    "history": []
  }'
```

### 3. **AWS Security Audit**
```bash
# Trigger S3 bucket security analysis
curl -X POST http://localhost:8001/tools/aws/s3-audit \
  -H "Content-Type: application/json" \
  -d '{"bucket_name": "my-secure-bucket"}'
```

### 4. **Compliance Checking**
```bash
# Check compliance against policies
curl http://localhost:8000/policies/1/compliance
```

## 🔧 Development

### Project Structure
```
cloud-sentinel/
├── backend/                    # FastAPI Backend
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── agent.py          # Security agent with MCP
│   │   └── models.py         # Data models
│   ├── rag/                  # RAG Implementation
│   │   ├── ingest.py         # Document processing
│   │   ├── retrieve.py       # Semantic search
│   │   └── services.py       # Vector DB services
│   └── requirements.txt      # Python dependencies
├── mcp-server/                # MCP Server for AWS tools
│   ├── server.py             # FastMCP server
│   ├── tools/                # Security audit tools
│   └── requirements.txt      # Python dependencies
├── frontend/                  # React TypeScript Frontend
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── hooks/           # Custom hooks
│   │   ├── services/        # API services
│   │   └── types/           # TypeScript types
│   ├── public/              # Static assets
│   └── package.json         # Node.js dependencies
├── docker-compose.yml         # Development orchestration
├── docker-compose.prod.yml    # Production configuration
└── docs/                     # Documentation
```

### Development Workflow

1. **Frontend Development**
   ```bash
   cd frontend
   npm run dev          # Start development server
   npm run build        # Build for production
   npm run test         # Run tests
   npm run lint         # Lint code
   ```

2. **Backend Development**
   ```bash
   cd backend
   uvicorn app.main:app --reload  # Start development server
   python -m pytest tests/        # Run tests
   python -m black .              # Format code
   python -m flake8 .             # Lint code
   ```

3. **MCP Server Development**
   ```bash
   cd mcp-server
   python server.py               # Start development server
   python -m pytest tests/        # Run tests
   ```

### Code Quality
- **TypeScript** for frontend type safety
- **ESLint** and **Prettier** for code formatting
- **Black** and **Flake8** for Python code quality
- **Automated testing** with Jest and pytest
- **Pre-commit hooks** for quality assurance

## 🔒 Security Features

### Data Protection
- **Encryption in Transit**: All API communications use HTTPS
- **Environment Variables**: Sensitive data stored securely
- **Authentication**: Clerk-based user authentication
- **Rate Limiting**: Protection against abuse and DoS
- **CORS Protection**: Controlled cross-origin access

### Container Security
- **Non-root Containers**: Services run as non-root users
- **Minimal Images**: Reduced attack surface
- **Resource Limits**: CPU and memory constraints
- **Network Isolation**: Custom Docker networks
- **Security Scanning**: Automated vulnerability scanning

### API Security
- **Input Validation**: Pydantic models for request validation
- **Error Handling**: Secure error messages without data leakage
- **Logging**: Comprehensive audit logging
- **Authentication**: JWT-based API authentication

## 📊 Monitoring & Observability

### Health Monitoring
```bash
# Monitor service health
watch docker-compose ps

# Check service logs
docker-compose logs -f backend
docker-compose logs -f mcp-server
docker-compose logs -f frontend
```

### Performance Monitoring
```bash
# Resource usage
docker stats

# Application metrics
curl http://localhost:8000/metrics
```

### Log Analysis
```bash
# Search logs
docker-compose logs backend | grep ERROR

# Real-time log monitoring
docker-compose logs -f --tail=100
```

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

# Check resource usage
docker system df
```

**Backend can't connect to MCP Server:**
```bash
# Check network connectivity
docker-compose exec backend ping mcp-server

# Check MCP server status
docker-compose logs mcp-server

# Test MCP endpoint
curl http://mcp-server:8001/health
```

**Frontend can't reach backend:**
```bash
# Check CORS configuration
docker-compose exec frontend curl http://backend:8000/health

# Verify environment variables
docker-compose exec frontend env | grep VITE_API_URL

# Check frontend build
docker-compose logs frontend
```

**Vector database issues:**
```bash
# Check Pinecone connectivity
docker-compose exec backend python -c "import pinecone; pinecone.init()"

# Verify API keys
docker-compose exec backend env | grep PINECONE
```

**Redis connection issues:**
```bash
# Test Redis connectivity
docker-compose exec backend redis-cli -h redis -p 6379 ping

# Check Redis logs
docker-compose logs redis
```

### Performance Optimization

**Memory issues:**
```bash
# Check memory usage
docker stats --no-stream

# Increase memory limits in docker-compose.yml
# Restart with new limits
docker-compose down && docker-compose up -d
```

**Slow API responses:**
```bash
# Check CPU usage
docker stats

# Scale services
docker-compose up -d --scale backend=2

# Check database performance
curl http://localhost:8000/policies
```

## 📈 Performance & Scalability

### Horizontal Scaling
```bash
# Scale backend services
docker-compose up -d --scale backend=3 --scale mcp-server=2

# Load balancer configuration
# Update nginx.conf for load balancing
```

### Vertical Scaling
```yaml
# docker-compose.prod.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

### Caching Strategy
- **Redis** for session storage and rate limiting
- **Pinecone** for vector similarity caching
- **CDN** for static asset delivery
- **Browser caching** for PWA assets

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Set up development environment
4. Make your changes
5. Test thoroughly
6. Submit a pull request

### Code Standards
- Follow existing code style and patterns
- Write comprehensive tests
- Update documentation
- Ensure all tests pass
- Add meaningful commit messages

### Pull Request Process
1. **Tests**: All tests must pass
2. **Code Review**: At least one approval required
3. **Documentation**: Update relevant docs
4. **Security**: No security vulnerabilities
5. **Performance**: No performance regressions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Community

### Getting Help
- **Documentation**: Check this README and inline code comments
- **Issues**: Open an issue on GitHub for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Security**: Report security issues privately via email

### Community
- **GitHub**: https://github.com/your-repo/cloud-sentinel
- **Website**: https://cloud-sentinel.example.com
- **Documentation**: https://docs.cloud-sentinel.example.com

## 🗺️ Roadmap

### Upcoming Features
- [ ] **Multi-Cloud Support**: Azure and GCP security auditing
- [ ] **Advanced Analytics**: Machine learning-powered security insights
- [ ] **Team Collaboration**: Multi-user workspace features
- [ ] **Custom Integrations**: Webhook and API integrations
- [ ] **Mobile App**: Native iOS and Android applications
- [ ] **Enterprise Features**: SSO, audit logs, compliance reporting

### Version History
- **v1.0.0**: Initial release with core features
- **v1.1.0**: PWA support and performance improvements
- **v1.2.0**: Enhanced error handling and security fixes
- **v1.3.0**: AWS auditing and compliance features

---

## 🎉 Quick Start Commands

```bash
# Clone and start
git clone <repo-url> && cd cloud-sentinel
docker-compose up -d

# Access the application
open http://localhost:3000

# Monitor logs
docker-compose logs -f
```

## 📋 Complete Feature Documentation

For a detailed breakdown of all features and capabilities, see the comprehensive [FEATURES.md](./FEATURES.md) document, which includes:

- **60+ Individual Features** across all components
- **Technical Implementation Details** for each feature
- **Use Cases & Applications** for different user types
- **Performance & Security Specifications**

**Built with ❤️ for secure cloud infrastructure**

*Cloud Sentinel - Where AI meets Cloud Security* 🛡️

