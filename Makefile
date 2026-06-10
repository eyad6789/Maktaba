.PHONY: help install install-gpu infra-up infra-down api worker ingest eval bench e2e test compile fmt web-dev web-build

help:
	@echo "Targets:"
	@echo "  install      Install core Python deps"
	@echo "  install-gpu  Install GPU/OCR deps (run on GPU server)"
	@echo "  infra-up     Start Qdrant + Redis via docker compose"
	@echo "  infra-down   Stop infra"
	@echo "  api          Run the FastAPI server (port 8000)"
	@echo "  worker       Run an RQ ingestion worker"
	@echo "  ingest DIR=path/to/pdfs   Bulk-ingest a directory"
	@echo "  eval Q=questions.jsonl     Run the retrieval eval harness"
	@echo "  bench DIR=path/to/pdfs     Benchmark ingestion + project to full corpus"
	@echo "  e2e          Run live-service integration tests (needs Qdrant+Redis)"
	@echo "  test         Run unit tests"
	@echo "  compile      Byte-compile all sources (syntax check)"
	@echo "  web-dev      Run the React SPA dev server (proxies to :8000)"
	@echo "  web-build    Build the SPA into api/static/dist"

install:
	pip install -r requirements.txt

install-gpu:
	pip install -r requirements.txt -r requirements-gpu.txt

infra-up:
	docker compose up -d

infra-down:
	docker compose down

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

worker:
	python -m ingest.worker

ingest:
	python -m scripts.ingest_dir $(DIR)

eval:
	python -m scripts.eval $(Q)

bench:
	python -m scripts.benchmark --books $(DIR) --total-books 1000 --pages-per-book 400 --scanned-frac 0.30 --workers 4

e2e:
	python -m tests.integration.run_e2e

test:
	python -m pytest -q

compile:
	python -m compileall -q config.py core ingest retrieval llm api scripts

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm ci && npm run build
