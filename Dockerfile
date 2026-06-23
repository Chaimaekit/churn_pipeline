# Use an optimized official Python runtime matrix
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for specific heavy ML compilation (like XGBoost/CatBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements configurations
COPY requirements.txt .

# Install dependencies cleanly
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source directories
COPY . .

# Expose FastAPI default container routing target
EXPOSE 8000

# Run using Uvicorn optimized for container process tracking
CMD ["uvicorn", "main.py:app", "--host", "0.0.0.0", "--port", "8000"]