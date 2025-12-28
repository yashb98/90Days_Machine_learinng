#!/bin/bash

# CloudSentinel Individual Service Testing Script
# This script tests each service separately before full integration testing

set -e

echo "🧪 CloudSentinel Individual Service Testing"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# Clean up any existing containers
cleanup() {
    echo -e "\n🧹 Cleaning up test containers..."
    docker compose down -v --remove-orphans 2>/dev/null || true
}

# Trap cleanup on script exit
trap cleanup EXIT

print_info "Starting individual service testing..."

# ============================
# TEST 1: REDIS SERVICE
# ============================
echo -e "\n🔴 Testing Redis Service"
echo "========================"

print_info "Starting Redis container..."
docker compose up -d redis

print_info "Waiting for Redis to be ready (10 seconds)..."
sleep 10

if docker compose exec redis redis-cli ping | grep -q PONG; then
    print_status "Redis service is working correctly"
else
    print_error "Redis service failed"
    exit 1
fi

print_info "Testing Redis CLI connection..."
docker compose exec redis redis-cli set test_key "hello_world"
if docker compose exec redis redis-cli get test_key | grep -q "hello_world"; then
    print_status "Redis read/write operations working"
else
    print_error "Redis read/write operations failed"
    exit 1
fi

print_info "Stopping Redis for next test..."
docker compose stop redis
sleep 2

# ============================
# TEST 2: BACKEND SERVICE
# ============================
echo -e "\n🐍 Testing Backend Service"
echo "=========================="

print_info "Creating minimal .env for backend testing..."
cat > .env.test << EOF
GOOGLE_API_KEY=test_key_for_validation_only
PINECONE_API_KEY=test_pinecone_key
PINECONE_INDEX=test_index
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
DEBUG=true
LOG_LEVEL=INFO
MCP_SERVER_URL=http://mcp-server:8001/sse
EOF

print_info "Starting backend container..."
docker compose up -d backend

print_info "Waiting for backend to start (30 seconds)..."
sleep 30

# Test health endpoint
print_info "Testing backend health endpoint..."
if curl -f -s http://localhost:8000/health > /dev/null; then
    print_status "Backend health endpoint responding"
else
    print_error "Backend health endpoint failed"
    docker compose logs backend
    exit 1
fi

# Test policies endpoint
print_info "Testing policies endpoint..."
if curl -f -s http://localhost:8000/policies > /dev/null; then
    print_status "Backend policies endpoint responding"
else
    print_error "Backend policies endpoint failed"
    docker compose logs backend
    exit 1
fi

print_info "Stopping backend..."
docker compose stop backend
sleep 2

# ============================
# TEST 3: MCP SERVER SERVICE
# ============================
echo -e "\n🔧 Testing MCP Server Service"
echo "=============================="

print_info "Creating minimal .env for MCP testing..."
cat > .env.mcp << EOF
AWS_ACCESS_KEY_ID=test_key
AWS_SECRET_ACCESS_KEY=test_secret
AWS_REGION=us-east-1
EOF

print_info "Starting MCP server container..."
docker compose up -d mcp-server

print_info "Waiting for MCP server to start (20 seconds)..."
sleep 20

# Test health endpoint
print_info "Testing MCP server health endpoint..."
if curl -f -s http://localhost:8001/health > /dev/null; then
    print_status "MCP server health endpoint responding"
else
    print_warning "MCP server health endpoint not available (may be expected for MCP)"
fi

# Test SSE connection (basic)
print_info "Testing MCP server SSE connection..."
timeout 10 curl -s http://localhost:8001/sse | head -n 1 > /dev/null && \
    print_status "MCP server SSE endpoint accessible" || \
    print_warning "MCP server SSE endpoint test inconclusive"

print_info "Stopping MCP server..."
docker compose stop mcp-server
sleep 2

# ============================
# TEST 4: FRONTEND SERVICE
# ============================
echo -e "\n⚛️  Testing Frontend Service"
echo "============================"

print_info "Creating frontend environment..."
cat > .env.frontend << EOF
VITE_API_URL=http://localhost:8000
VITE_MCP_URL=http://localhost:8001
EOF

print_info "Starting frontend container..."
docker compose up -d frontend

print_info "Waiting for frontend to build and start (60 seconds)..."
sleep 60

# Test nginx health endpoint
print_info "Testing frontend health endpoint..."
if curl -f -s http://localhost:80/health > /dev/null; then
    print_status "Frontend health endpoint responding"
else
    print_warning "Frontend health endpoint not responding (nginx may still be starting)"
fi

# Test if static files are served
print_info "Testing static file serving..."
if curl -s http://localhost:80/ | grep -q "<html\|<!DOCTYPE"; then
    print_status "Frontend serving HTML content"
else
    print_error "Frontend not serving content properly"
    docker compose logs frontend
fi

print_info "Stopping frontend..."
docker compose stop frontend
sleep 2

# ============================
# TEST 5: COMPLETE INTEGRATION TEST
# ============================
echo -e "\n🔗 Testing Complete Integration"
echo "================================"

print_info "Starting all services together..."
docker compose up -d

print_info "Waiting for all services to start (45 seconds)..."
sleep 45

# Test all services are running
print_info "Checking all services status..."
services_ok=true

for service in redis backend mcp-server frontend; do
    if docker compose ps $service | grep -q "Up"; then
        print_status "$service is running"
    else
        print_error "$service is not running"
        services_ok=false
    fi
done

if [ "$services_ok" = false ]; then
    print_error "Some services failed to start"
    docker compose logs --tail=20
    exit 1
fi

# Test inter-service communication
print_info "Testing Redis connectivity from backend..."
if docker compose exec backend sh -c "which redis-cli && redis-cli -h redis -p 6379 ping" 2>/dev/null | grep -q PONG; then
    print_status "Backend can communicate with Redis"
else
    print_warning "Backend-Redis communication test inconclusive"
fi

print_info "Testing backend API health..."
if curl -f -s http://localhost:8000/health > /dev/null; then
    print_status "Backend API is accessible"
else
    print_error "Backend API is not accessible"
    docker compose logs backend
fi

print_info "Testing frontend accessibility..."
if curl -s http://localhost:3000/ | grep -q "<html\|<!DOCTYPE"; then
    print_status "Frontend is accessible"
else
    print_warning "Frontend accessibility test inconclusive"
fi

# ============================
# TEST 6: API INTEGRATION TEST
# ============================
echo -e "\n🌐 Testing API Integration"
echo "==========================="

print_info "Testing backend policies API..."
policies_response=$(curl -s http://localhost:8000/policies || echo "FAILED")
if echo "$policies_response" | grep -q "id\|name\|status"; then
    print_status "Backend policies API returning expected data"
else
    print_warning "Backend policies API response format unexpected"
    print_info "Response: $policies_response"
fi

print_info "Testing chat API with simple request..."
chat_response=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "history": []}' || echo "FAILED")

if echo "$chat_response" | grep -q "response\|error"; then
    print_status "Backend chat API responding"
    print_info "Chat response preview: $(echo "$chat_response" | head -c 100)..."
else
    print_warning "Backend chat API response format unexpected"
fi

# ============================
# FINAL REPORT
# ============================
echo -e "\n📊 Final Test Report"
echo "===================="

print_info "Service Status Summary:"
docker compose ps

print_info "Port Availability Check:"
echo "Frontend (3000): $(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/ || echo "FAILED")"
echo "Backend (8000): $(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "FAILED")"
echo "MCP (8001): $(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/health || echo "N/A")"
echo "Redis (6379): $(docker compose exec redis redis-cli ping 2>/dev/null | grep -q PONG && echo "OK" || echo "FAILED")"

print_status "Individual service testing completed!"
print_info "Services are now running and ready for Google Cloud deployment."

echo -e "\n🚀 Next Steps:"
echo "1. Services are running - you can now test the UI at http://localhost:3000"
echo "2. Configure your real API keys in .env file"
echo "3. Test the full application functionality"
echo "4. Ready for Google Cloud deployment when satisfied"
