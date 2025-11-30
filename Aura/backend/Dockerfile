FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# **CRITICAL FIX** - Install complete OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"



## FOR DOCKER LOCALLY 

# FROM python:3.11-slim

# # Create non-root user
# RUN groupadd -r appuser && useradd -r -g appuser appuser

# WORKDIR /app

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY . .

# # Fix permissions and switch to non-root
# RUN chown -R appuser:appuser /app
# USER appuser

# EXPOSE 8080
# CMD ["python", "main.py"]