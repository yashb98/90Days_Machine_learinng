# 1. Use an official lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements first (this helps Docker cache dependencies)
COPY requirements.txt .

# 4. Install dependencies
# --no-cache-dir keeps the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
COPY . .

# 6. Expose the port Cloud Run expects
EXPOSE 8080

# 7. Run the application
# Ensure 'main:app' matches your file name (main.py) and app instance (app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
