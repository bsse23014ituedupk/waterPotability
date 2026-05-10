FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary runtime directories
RUN mkdir -p logs artifacts/models artifacts/plots artifacts/reports mlruns

# Set PYTHONPATH so src package is importable
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Health check using curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI via uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
