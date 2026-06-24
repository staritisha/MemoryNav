# backend/Dockerfile
FROM python:3.11-slim

# System deps: libgl1 + libglib2.0-0 for OpenCV, build-essential for native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer — only re-runs when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create data directories that will be bind-mounted at runtime.
# Creating them at build time means the app starts correctly even if
# the host hasn't mounted volumes yet (e.g. first run).
RUN mkdir -p /app/data/chroma_store /app/data/models

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    # Tell config.py where to find data when running in Docker
    CHROMA_PERSIST_DIR=/app/data/chroma_store \
    SQLITE_DB_PATH=/app/data/memorynav.db \
    YOLO_MODEL_PATH=/app/data/models/yolov8n.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
