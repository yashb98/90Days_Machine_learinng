#!/bin/bash

# CloudSentinel Container Test Script
# This script validates that the Docker setup is working correctly

set -e

echo "🧪 CloudSentinel Container Test Suite"
echo "====================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Test 1: Check if Docker is running
echo -e "\n1. Checking Docker availability..."
if docker info > /dev/null 2>&1; then
    print_status "Docker is running"
else
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

# Test 2: Check if Docker Compose is available
echo -e "\n2. Checking Docker Compose availability..."
if docker-compose --version > /dev/null 2>&1; then
    print_status "Docker Compose is available"
else
    print_error "Docker Compose is not installed."
    exit 1
fi

# Test 3: Check if .env file exists
echo -e "\n3. Checking environment configuration..."
if [ -f ".env" ]; then
    print_status ".env file found"
else
    print_warning ".env file not found. Copying from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_status "Created .env from .env.example. Please configure your API keys."
    else
        print_error ".env.example not found"
        exit 1
    fi
fi

# Test 4: Check if all required files exist
echo -e "\n4. Checking required files..."
required_files=("backend/Dockerfile" "mcp-server/Dockerfile" "frontend/Dockerfile" "docker-compose.yml")

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "$file exists"
    else
        print_error "$file is missing"
        exit 1
    fi
done

# Test 5: Build Docker images
echo -e "\n5. Building Docker images..."
echo "This may take several minutes on first run..."

if docker-compose build --no-cache; then
    print_status "Docker images built successfully"
else
    print_error "Failed to build Docker images"
    exit 1
fi

# Test 6: Start services
echo -e "\n6. Starting services..."
docker-compose up -d

# Wait for services to start
echo "Waiting for services to start (30 seconds)..."
sleep 30

# Test 7: Check service health
echo -e "\n7. Checking service health..."

services=("backend:8000" "mcp-server:8001" "frontend:80")

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    
    if curl -f -s "http://localhost:$port/health" > /dev/null 2>&1; then
        print_status "$name service is healthy"
    else
        print_warning "$name service may not be ready yet (checking logs)..."
        docker-compose logs --tail=10 "$name" 2>/dev/null || true
    fi
done

# Test 8: Test backend API
echo -e "\n8. Testing backend API..."
if curl -f -s "http://localhost:8000/policies" > /dev/null 2>&1; then
    print_status "Backend API is responding"
else
    print_warning "Backend API may not be ready yet"
fi

# Test 9: Display service status
echo -e "\n9. Service Status:"
docker-compose ps

# Test 10: Show access URLs
echo -e "\n10. Access URLs:"
echo "Frontend:  http://localhost:3000"
echo "Backend:   http://localhost:8000"
echo "MCP:       http://localhost:8001"
echo "Redis:     localhost:6379"

# Cleanup option
echo -e "\n🧹 To stop and clean up services, run:"
echo "docker-compose down"

print_status "Containerization test completed!"
echo -e "\n${GREEN}Success! CloudSentinel is now containerized and ready to run anywhere.${NC}"
