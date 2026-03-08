FROM python:3.12-slim

LABEL maintainer="FinEdge AI" \
      version="1.0.0" \
      description="Production ONNX Financial LLM Engine"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (models mounted at runtime via volume)
COPY amifi_ai/ ./amifi_ai/
COPY config.py .
COPY .env .

# Create dirs
RUN mkdir -p models logs artifacts

EXPOSE 8000

ENV MODEL_DIR=models/tinyllama_onnx \
    LOG_LEVEL=INFO \
    HOST=0.0.0.0 \
    PORT=8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "amifi_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
