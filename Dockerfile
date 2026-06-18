# Use a slim, stable python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your scripts into the image
COPY . .

# Keep container open for interactive script running
CMD ["tail", "-f", "/dev/null"]