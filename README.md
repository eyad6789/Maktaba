# 📚 Book RAG — Arabic / English

A Retrieval-Augmented Generation system that ingests **PDF books** (native text
**or** scanned images), stores them in a vector database, and answers questions
**grounded in the books** with citations back to *book + page*.

Built for scale: **1000+ books × ~400 pages ≈ 400k pages → ~500k–1M chunks.**

## Architecture

```
INGESTION (batch, local GPU)
  PDF ──▶ per-page classify (text vs scan)
        ├─ native text ─▶ PyMuPDF extract
        └─ scanned     ─▶ VLM OCR (Qwen2.5-VL) / Surya / Tesseract
        ──▶ Arabic+EN normalize ──▶ chunk (page-aware, +metadata)
        ──▶ BGE-M3 embed (dense + sparse) ──▶ upsert ▶ Qdrant

QUERY (online)
  question ─▶ BGE-M3 embed (dense+sparse)
           ─▶ Qdrant hybrid search (RRF fusion, top-50)
           ─▶ bge-reranker-v2-m3 rerank ─▶ top-8
           ─▶ grounded prompt
           ─▶ local LLM (Qwen2.5-Instruct) ─▶ answer + citations [book, page]
```

**Fully offline:** OCR, embeddings, reranking, the vector DB, **and the answer
LLM** all run locally on your GPU server — no data leaves the machine and no API
key is required. (A cloud Claude backend is available as an opt-in, off by default.)

### Component choices

| Concern | Tool | Why |
|---|---|---|
| Native text | **PyMuPDF** | fast, accurate, detects scans |
| OCR (Arabic) | **Qwen2.5-VL** (Surya / Tesseract fallback) | best Arabic OCR; local on GPU |
| Normalization | custom | alef/ya/tatweel/diacritics — critical for Arabic recall |
| Embeddings | **BGE-M3** | multilingual incl. Arabic; dense + sparse in one model |
| Vector DB | **Qdrant** | scales to 1M+; native dense+sparse + RRF; quantization |
| Reranker | **bge-reranker-v2-m3** | multilingual cross-encoder; big accuracy lift |
| Answer LLM | **local Qwen2.5-Instruct** (transformers / vLLM / Ollama) | offline, strong AR/EN, grounded + citations |
| Serving | **FastAPI** | web chat UI at `/` + `/chat`, `/query`, `/ingest`, `/status` |
| Batch queue | **RQ + Redis** | parallel, resumable ingestion |

## Layout

```
config.py            # all settings (env-overridable)
core/                # models.py (shared types), schema.py (Qdrant), logging.py
ingest/              # classify, extract, ocr, normalize, chunk, embed, registry, pipeline, worker
retrieval/           # search (Qdrant hybrid), rerank
llm/                 # prompts, engine (offline backends), answer, chat
api/main.py          # FastAPI app
scripts/             # ingest_dir.py, eval.py
tests/
CONTRACT.md          # module-by-module implementation spec
```

## Setup

### 1. Infra
```bash
docker compose up -d        # Qdrant (:6333) + Redis (:6379)
```

### 2. Python deps
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # core (CPU-OK)
pip install -r requirements-gpu.txt        # GPU server only (torch + Qwen2.5-VL + surya)
```
For the Tesseract fallback: `apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng`.

### 3. Config
```bash
cp .env.example .env        # offline defaults — no API key needed; *_DEVICE=cpu for dev
```

> **Local dev without a GPU:** set `EMBEDDING_DEVICE=cpu`, `RERANKER_DEVICE=cpu`,
> and `OCR_BACKEND=tesseract`. BGE-M3 on CPU is slow but functional for a few books.

## Usage

**Ingest** a folder of PDFs (enqueues background jobs; run a worker to process):
```bash
python -m ingest.worker &                  # start a worker (repeat for parallelism)
python -m scripts.ingest_dir data/books    # enqueue all PDFs
# or ingest inline without the queue:
python -m scripts.ingest_dir data/books --sync
```

**Serve** the API:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Ask** a question (single-shot):
```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "ما هي الفكرة الرئيسية في الفصل الأول؟"}' | jq
```

### 💬 Chat with your library

Once the API is running, open **http://localhost:8000** in a browser — a
bilingual (Arabic RTL / English LTR) chat UI is served there. It's multi-turn,
shows the page-level sources behind each answer, and reports when something
isn't in the books.

Prefer the terminal?
```bash
python -m scripts.chat                 # talks to the running API
```

Under the hood the UI/CLI call `POST /chat`, which keeps conversation history,
**condenses follow-ups into a standalone search query** (so "and the second
one?" still retrieves correctly), then runs the same hybrid-search → rerank →
grounded-answer path as `/query`:
```bash
curl -s localhost:8000/chat -H 'content-type: application/json' -d '{
  "messages": [
    {"role": "user", "content": "What does the book say about justice?"},
    {"role": "assistant", "content": "It frames justice as fairness [1]."},
    {"role": "user", "content": "and how does that relate to law?"}
  ]
}' | jq

**Evaluate** retrieval quality:
```bash
python -m scripts.eval data/questions.jsonl --k 8
```

**Benchmark** ingestion throughput on a sample and project to the full corpus:
```bash
python -m scripts.benchmark --books data/sample \
    --total-books 1000 --pages-per-book 400 --scanned-frac 0.30 --workers 4
```
Run this on ~10 representative books (mixing native + scanned) **before**
committing the whole corpus — OCR dominates runtime, so the projection tells you
how long the full ingest will take and how big the vector DB will get.

## Testing

Unit tests (pure logic — Arabic normalization, chunking, classification,
registry, benchmark math) run anywhere:
```bash
pip install -r requirements-dev.txt
make test                      # pytest
```

A full **live-service** integration run exercises real PyMuPDF parsing, the real
Qdrant hybrid-search + RRF code path, the FastAPI routes, and the RQ enqueue —
using a deterministic stand-in embedder/reranker and a fake LLM engine, so it
needs **no GPU and no API key**, only Qdrant + Redis:
```bash
docker compose up -d           # Qdrant (:6333) + Redis (:6379)
make e2e                       # python -m tests.integration.run_e2e
```

## Scaling to the full corpus (400k pages)

- **Throughput:** OCR dominates. Run multiple workers; one GPU process per worker.
  Benchmark ~10 books to extrapolate total runtime before committing the corpus.
- **Idempotency / resume:** every book is hashed; re-running skips completed books
  (`ingest/registry.py`), so ingestion is interruptible.
- **Memory:** ~500k–1M vectors. Scalar quantization (on by default) keeps Qdrant
  RAM in check; payload/text stored on disk.
- **Accuracy tuning:** chunk size, `search_top_k`, `rerank_top_k`, RRF weights, and
  Sonnet-vs-Opus are all in `config.py`. Use `scripts/eval.py` to measure changes.

## Notes

- Citations: answers cite sources as `[n]`, mapped back to `{title, author, page}`.
- Out-of-corpus questions: the model is instructed to say the answer is **not in
  the books** rather than fabricate.
- Diacritics: kept by default (`STRIP_DIACRITICS=false`) — important for classical
  and religious Arabic texts; flip to `true` if it helps retrieval on your corpus.

See `CONTRACT.md` for the precise module API.
