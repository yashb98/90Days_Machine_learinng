# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
# We install system dependencies for some heavy python packages if needed
RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["python3", "app.py"]