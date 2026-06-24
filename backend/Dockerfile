# backend/Dockerfile
# Python 3.11 base, copy app, pip install, uvicorn entrypoint

FROM python:3.11-slim

# --- System dependencies ---
# libgl1 / libglib2.0-0: required by opencv-python (cv2) for image decoding
# build-essential: needed if any pip package compiles native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Install Python dependencies first (better layer caching) ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Copy application code ---
COPY app/ ./app/
COPY scripts/ ./scripts/

# --- Model + data directories (mount volumes over these in production) ---
RUN mkdir -p /app/models /app/chroma_db

# Models can be downloaded at build time or mounted as a volume at runtime.
# Uncomment if you want the YOLO weights baked into the image:
# COPY models/ ./models/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# Basic healthcheck against FastAPI (adjust path if you don't have /health)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# --- Entrypoint: uvicorn serving the FastAPI app ---
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]