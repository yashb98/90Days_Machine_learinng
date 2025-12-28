# Individual Service Testing Guide

## 🧪 Testing Strategy

Before deploying to Google Cloud, we'll test each service individually to ensure everything works perfectly.

## 📋 Testing Script Overview

The `test-services-individually.sh` script performs the following tests:

### 1. **Redis Service Test** 🔴
- Starts Redis container
- Tests basic `PING/PONG` functionality
- Tests read/write operations
- Validates CLI connectivity

### 2. **Backend Service Test** 🐍
- Creates minimal test environment
- Starts Backend container
- Tests `/health` endpoint
- Tests `/policies` endpoint
- Validates API responses

### 3. **MCP Server Service Test** 🔧
- Creates AWS test environment
- Starts MCP Server container
- Tests health endpoint
- Tests SSE connection
- Validates tool availability

### 4. **Frontend Service Test** ⚛️
- Creates frontend environment
- Builds React application
- Tests Nginx serving
- Validates static file delivery
- Tests health endpoint

### 5. **Integration Test** 🔗
- Starts all services together
- Tests inter-service communication
- Validates service dependencies
- Checks port accessibility

### 6. **API Integration Test** 🌐
- Tests backend policies API
- Tests chat API functionality
- Validates response formats
- Confirms data flow

## 🚀 How to Run Tests

### Step 1: Navigate to Project Directory
```bash
cd cloud_sential
```

### Step 2: Run Individual Service Tests
```bash
./test-services-individually.sh
```

### Step 3: Review Results
The script will show:
- ✅ Green checkmarks for successful tests
- ⚠️ Yellow warnings for non-critical issues
- ❌ Red errors for failed tests

## 📊 Expected Results

### If All Tests Pass ✅
- All services start successfully
- APIs respond correctly
- Inter-service communication works
- Ready for Google Cloud deployment

### If Some Tests Fail ❌
- Check Docker logs: `docker-compose logs [service-name]`
- Verify ports are available
- Ensure Docker daemon is running
- Check environment variables

## 🔧 Troubleshooting Common Issues

### Backend Won't Start
```bash
# Check logs
docker-compose logs backend

# Common issues:
# - Missing environment variables
# - Port 8000 already in use
# - Dependencies not installed
```

### Frontend Build Fails
```bash
# Check logs
docker-compose logs frontend

# Common issues:
# - Node.js dependencies missing
# - Build memory issues
# - Port 3000 already in use
```

### Services Can't Communicate
```bash
# Check network
docker network ls
docker-compose exec backend ping mcp-server

# Common issues:
# - Network not created
# - Service names incorrect
# - Firewall/port blocking
```

## ✅ Success Criteria

After running the tests, you should see:
- All 4 services (Redis, Backend, MCP, Frontend) running
- Health endpoints responding (except MCP which may vary)
- API endpoints returning expected data
- No critical errors in the output

## 🎯 Next Steps After Testing

1. **If All Tests Pass**: ✅ Ready for Google Cloud deployment
2. **If Issues Found**: 🔧 Fix problems and re-run tests
3. **Manual Testing**: 🧪 Test the web interface at http://localhost:3000
4. **Performance Check**: 📊 Verify response times and resource usage

## 🏃‍♂️ Quick Test Commands

If you want to test services individually without the full script:

```bash
# Test only Redis
docker-compose up -d redis
curl http://localhost:6379  # or use redis-cli ping

# Test only Backend
docker-compose up -d backend
curl http://localhost:8000/health

# Test only Frontend
docker-compose up -d frontend
curl http://localhost:3000/

# Test all together
docker-compose up -d
./test-containerization.sh
```

This systematic approach ensures we catch any issues before cloud deployment!
