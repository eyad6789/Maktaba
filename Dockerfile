# Image for the API server and the RQ ingestion worker (CPU base).
# For GPU embedding/OCR, build on an NVIDIA CUDA base instead and also install
# requirements-gpu.txt (and a CUDA torch build).

# Stage 1: build the React SPA. Node exists only here — the runtime image
# receives nothing but the static build output.
FROM node:20-slim AS webbuild
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
# vite.config.ts outputs to ../api/static/dist relative to /web -> /api/static/dist
RUN npm run build

# Stage 2: the Python runtime.
FROM python:3.11-slim

# Tesseract (fallback OCR) with Arabic + English language data.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=webbuild /api/static/dist /app/api/static/dist

EXPOSE 8000

# API by default; override the command to run the worker:
#   docker run ... python -m ingest.worker
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
