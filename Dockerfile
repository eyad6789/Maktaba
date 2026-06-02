# Image for the API server and the RQ ingestion worker (CPU base).
# For GPU embedding/OCR, build on an NVIDIA CUDA base instead and also install
# requirements-gpu.txt (and a CUDA torch build).
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

EXPOSE 8000

# API by default; override the command to run the worker:
#   docker run ... python -m ingest.worker
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
