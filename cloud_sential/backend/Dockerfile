# # Multi-stage build for Backend
# FROM python:3.11-slim as builder

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PIP_NO_CACHE_DIR=1 \
#     PIP_DISABLE_PIP_VERSION_CHECK=1

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# # Create and activate virtual environment
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"

# # Copy requirements and install dependencies
# COPY requirements.txt .
# RUN pip install --upgrade pip && \
#     pip install -r requirements.txt

# # Production stage
# FROM python:3.11-slim

# # Set environment variables
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     PATH="/opt/venv/bin:$PATH"

# # Install runtime dependencies
# RUN apt-get update && apt-get install -y \
#     curl \
#     && rm -rf /var/lib/apt/lists/* \
#     && apt-get clean

# # Copy virtual environment from builder stage
# COPY --from=builder /opt/venv /opt/venv

# # Create app user for security
# RUN groupadd -r appuser && useradd -r -g appuser appuser

# # Create app directory
# WORKDIR /app

# # Copy application code
# COPY . .

# # Create necessary directories
# RUN mkdir -p temp_uploads data logs && \
#     chown -R appuser:appuser /app

# # Switch to non-root user
# USER appuser

# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# # Expose port
# EXPOSE 8000

# # Start the application
# CMD ["python", "-m", "backend.app.main"]

# Multi-stage build for Backend
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libyaml-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Add /app to PYTHONPATH so python can find the 'backend' module
    PYTHONPATH="/app:$PYTHONPATH" \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create the root app directory
WORKDIR /app

# -----------------------------------------------------------
# CRITICAL FIX: Copy code into a 'backend' subdirectory
# This ensures imports like 'from backend.rag...' still work
# -----------------------------------------------------------
COPY . ./backend

# Create necessary directories (inside the backend folder or root)
RUN mkdir -p backend/temp_uploads backend/data backend/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start the application using the full module path
CMD ["python", "-m", "backend.app.main"]