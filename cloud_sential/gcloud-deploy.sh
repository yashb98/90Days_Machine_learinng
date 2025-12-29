#!/bin/bash

# Cloud Sentinel - Google Cloud Deployment Script
# Deploy all services in correct dependency order

set -e

echo "🚀 Deploying Cloud Sentinel to Google Cloud..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI is not installed. Please install it first.${NC}"
    exit 1
fi

# Set project
echo -e "${YELLOW}📋 Setting up project...${NC}"
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}🔧 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Create Artifact Registry repository
echo -e "${YELLOW}📦 Creating container registry...${NC}"
gcloud artifacts repositories create cloud-sentinel-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Cloud Sentinel container registry" || true

# Configure Docker authentication
echo -e "${YELLOW}🔐 Configuring Docker authentication...${NC}"
gcloud auth configure-docker us-central1-docker.pkg.dev

# Deploy Redis first (infrastructure service)
echo -e "${GREEN}1️⃣ Deploying Redis...${NC}"
gcloud run deploy redis-service \
    --image redis:7-alpine \
    --platform managed \
    --region us-central1 \
    --memory 512Mi \
    --cpu 1 \
    --set-env-vars REDIS_PASSWORD=cloud_sentinel_redis_$(date +%s) \
    --allow-unauthenticated

# Deploy Backend
echo -e "${GREEN}2️⃣ Deploying Backend...${NC}"
cd backend
gcloud run deploy cloud-sentinel-backend \
    --source . \
    --platform managed \
    --region us-central1 \
    --memory 1Gi \
    --cpu 1 \
    --set-env-vars \
        GOOGLE_API_KEY=$GOOGLE_API_KEY,\
        PINECONE_API_KEY=$PINECONE_API_KEY,\
        PINECONE_INDEX=$PINECONE_INDEX,\
        CORS_ORIGINS=https://YOUR_FRONTEND_URL,\
        MCP_SERVER_URL=https://YOUR_MCP_URL/sse \
    --allow-unauthenticated
cd ..

# Deploy MCP Server
echo -e "${GREEN}3️⃣ Deploying MCP Server...${NC}"
cd mcp-server
gcloud run deploy cloud-sentinel-mcp \
    --source . \
    --platform managed \
    --region us-central1 \
    --memory 512Mi \
    --cpu 0.5 \
    --set-env-vars \
        AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID,\
        AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY,\
        AWS_REGION=us-east-1 \
    --allow-unauthenticated
cd ..

# Build and Deploy Frontend
echo -e "${GREEN}4️⃣ Building Frontend...${NC}"
cd frontend
npm run build

echo -e "${GREEN}5️⃣ Deploying Frontend...${NC}"
gcloud run deploy cloud-sentinel-frontend \
    --source . \
    --platform managed \
    --region us-central1 \
    --memory 256Mi \
    --cpu 0.5 \
    --set-env-vars \
        VITE_API_URL=https://YOUR_BACKEND_URL,\
        VITE_MCP_URL=https://YOUR_MCP_URL \
    --allow-unauthenticated
cd ..

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${YELLOW}📋 Next steps:${NC}"
echo "1. Update environment variables with actual URLs"
echo "2. Configure custom domain for frontend"
echo "3. Set up SSL certificates"
echo "4. Test the application end-to-end"

# Get service URLs
echo -e "${YELLOW}🌐 Service URLs:${NC}"
gcloud run services describe redis-service --platform managed --region us-central1 --format 'value(status.url)'
gcloud run services describe cloud-sentinel-backend --platform managed --region us-central1 --format 'value(status.url)'
gcloud run services describe cloud-sentinel-mcp --platform managed --region us-central1 --format 'value(status.url)'
gcloud run services describe cloud-sentinel-frontend --platform managed --region us-central1 --format 'value(status.url)'
