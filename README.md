# 📚 Book RAG — Arabic / English

A Retrieval-Augmented Generation system that ingests **PDF books** (native text
**or** scanned images), stores them in a vector database, and answers questions
**grounded in the books** with citations back to *book + page*.

Built for scale: **1000+ books × ~400 pages ≈ 400k pages → ~500k–1M chunks.**

## Architecture

```
INGESTION (batch)
  PDF ──▶ per-page classify (text vs scan)
        ├─ native text ─▶ PyMuPDF extract
        └─ scanned     ─▶ VLM OCR (Qwen2.5-VL) / Surya / Tesseract
        ──▶ Arabic+EN normalize ──▶ chunk (page-aware, +metadata)
        ──▶ COMPREHENSION LAYER (optional): detect structure (TOC) ──▶
            summarize each chapter + the whole book (small local LLM)
        ──▶ BGE-M3 embed (dense + sparse) ──▶ upsert ▶ Qdrant
            (raw passages + chapter/book summary nodes, tagged by `level`)

QUERY (online)
  question ─▶ route: LOCAL (factual) vs GLOBAL (whole-book / thematic)
           ─▶ BGE-M3 embed (dense+sparse)
           ─▶ LOCAL : hybrid search passages ─▶ rerank ─▶ top-k
              GLOBAL: hybrid search chapter+book summaries ─▶ rerank ─▶
                      drill into child passages ─▶ rerank ─▶ merge
           ─▶ grounded, route-aware prompt
           ─▶ local LLM (Qwen2.5 via Ollama) ─▶ answer + citations [book, page]
```

**Fully offline:** OCR, embeddings, reranking, the vector DB, **and the answer
LLM** all run locally — no data leaves the machine and no API key is required.
The default answer backend is a **quantized model served by Ollama**, so a 7B
model is practical even on a **CPU-only** box.

**Optional cloud quality with an offline guarantee:** set `LLM_BACKEND=fallback`
to answer with a chain of providers — **MiniMax → Gemini → local Qwen**. Each
cloud provider is used only if its API key is set, and any failure (bad key, no
network, error) automatically falls through to the next, ending at the local
model. So you get cloud-grade answers when online and never break when offline.
Cheap utility calls (summarization, routing, query condensation) always use the
local model to save cloud quota. (A cloud Claude backend is also available.)

**Reads like it understands the book, not just searches it.** The optional
comprehension layer builds chapter- and book-level summaries at ingest time, and
the query router sends whole-book / thematic questions to reason over those
summaries (then cites the underlying passages) — so "what is the author's main
argument?" gets a synthesized answer, while "how many…?" still gets a precise
passage lookup.

### Component choices

| Concern | Tool | Why |
|---|---|---|
| Native text | **PyMuPDF** | fast, accurate, detects scans |
| OCR (Arabic) | **Qwen2.5-VL** (Surya / Tesseract fallback) | best Arabic OCR; local on GPU |
| Normalization | custom | alef/ya/tatweel/diacritics — critical for Arabic recall |
| Embeddings | **BGE-M3** | multilingual incl. Arabic; dense + sparse in one model |
| Vector DB | **Qdrant** | scales to 1M+; native dense+sparse + RRF; quantization |
| Reranker | **bge-reranker-v2-m3** | multilingual cross-encoder; big accuracy lift |
| Answer LLM | **Qwen2.5-7B-Instruct via Ollama** (Q4, CPU-friendly; transformers/vLLM also supported) | offline, strong AR/EN, grounded + citations |
| Utility LLM | **Qwen2.5-3B-Instruct** | cheap high-volume calls: summarization, query condensation, routing |
| Comprehension | **chapter + book summaries** (RAPTOR-style) | answers whole-book / thematic questions, not just snippet lookups |
| Routing | **bilingual heuristic** (LOCAL vs GLOBAL) | factual lookups stay precise; thematic questions reason over summaries |
| Serving | **FastAPI** | web chat UI at `/` + `/chat`, `/query`, `/ingest`, `/status` |
| Batch queue | **RQ + Redis** | parallel, resumable ingestion |

## Layout

```
config.py            # all settings (env-overridable)
core/                # models.py (shared types), schema.py (Qdrant), logging.py
ingest/              # classify, extract, ocr, normalize, chunk, embed, registry, pipeline, worker
                     #   structure (TOC -> chapters), summarize (chapter/book summary nodes)
retrieval/           # search (Qdrant hybrid + level filter), rerank,
                     #   route (LOCAL/GLOBAL), pipeline (route-aware retrieval)
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

### 3. Answer model (Ollama — CPU-friendly default)
The default `LLM_BACKEND=openai` talks to a local **Ollama** server. Install
Ollama, then pull the answer + utility models **once** (then it runs fully
offline):
```bash
ollama pull qwen2.5:7b-instruct-q4_K_M     # answers   (~4.7 GB)
ollama pull qwen2.5:3b-instruct-q4_K_M     # utility   (~2 GB: summaries, routing, condensation)
# keep both warm so each request doesn't reload the model:
export OLLAMA_KEEP_ALIVE=30m OLLAMA_MAX_LOADED_MODELS=2
```
On a box with <12 GB RAM, run a single model: pull only the 3B and set both
`OPENAI_MODEL` and `UTILITY_MODEL` to `qwen2.5:3b-instruct-q4_K_M`.
Prefer a GPU and in-process weights instead? set `LLM_BACKEND=transformers`.

### 4. Config
```bash
cp .env.example .env        # CPU + Ollama offline defaults — no API key needed
```

> **Defaults are CPU-first**: `EMBEDDING_DEVICE`/`RERANKER_DEVICE=cpu`,
> `OCR_BACKEND=tesseract`, Ollama answer backend. On a GPU box set `*_DEVICE=cuda`,
> `*_USE_FP16=true`, and `OCR_BACKEND=qwen` for best OCR quality.

### 5. Comprehension layer (answer like it understood the book)
Off by default (it costs many summarization calls, one-time per book). To turn it
on, set `ENABLE_COMPREHENSION=true` in `.env`, then ingest — or **rebuild** it
over books you already ingested with `--force`:
```bash
ENABLE_COMPREHENSION=true python -m scripts.ingest_dir data/books --sync --force
```
This detects each book's chapters (from its PDF table of contents; enable
`ENABLE_HEADING_DETECTION=true` to also try heading heuristics when there's no
TOC) and stores chapter- and book-level summaries alongside the raw passages.
Whole-book / thematic questions then reason over those summaries. The summary
model is the small `SUMMARY_MODEL` (defaults to the 3B utility model); on CPU,
expect this to take a while for a large corpus, so run it via the worker.

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
ChatGPT-style bilingual (Arabic RTL / English LTR) web app is served there:

- **Streaming answers** (SSE) with inline `[n]` citations and the page-level
  sources shown *before* generation starts.
- **Conversations sidebar** — chats are persisted server-side (SQLite) and
  survive reloads; every conversation has a shareable URL (`/c/<id>`).
- **Model picker** — *Auto* (Gemini → Claude → Local fallback) or pin one
  model. A pinned model that hits its rate limit does **not** silently fall
  back; the UI says so and offers to switch to Auto.
- **Book scope picker** — ask across all books, one book, or any subset.
- **Library dashboard** at **/dashboard** — upload PDFs from the browser,
  watch vectorization progress live (queued → reading pages → embedding →
  indexing), see book/passage totals, and delete books.

The SPA lives in `web/` (React + Vite) and builds into `api/static/dist`
(`make web-build`); without a build the legacy single-file UI is served, so
the API works with no Node installed. For SPA development run `make api` and
`make web-dev` (Vite proxies API calls to :8000).

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
```

The full HTTP surface on top of `/query` and `/chat`:

| Endpoint | Purpose |
|---|---|
| `GET /models` | Providers for the model picker (Auto + chain, availability) |
| `POST /chat/stream` | One chat turn as SSE (`meta` → `provider` → `delta`* → `done`/`error`); history kept server-side |
| `GET/POST /conversations`, `GET/PATCH/DELETE /conversations/{id}` | Persisted conversation CRUD |
| `POST /upload` | Browser PDF upload (multipart, 200 MB cap) → enqueue ingestion |
| `GET /jobs`, `GET /jobs/{id}` | Live ingestion progress (stage + page counts from RQ job meta) |
| `GET /books`, `DELETE /books/{id}` | Library listing/totals; remove a book's vectors + registry row |

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
- **Accuracy tuning:** chunk size, `search_top_k`, `rerank_top_k`, RRF weights,
  the GLOBAL-route knobs (`global_summary_keep`, `global_child_keep`), and the
  answer/summary models are all in `config.py`. Use `scripts/eval.py` to measure changes.

## Notes

- Citations: answers cite sources as `[n]`, mapped back to `{title, author, page}`.
- Out-of-corpus questions: the model is instructed to say the answer is **not in
  the books** rather than fabricate.
- Diacritics: kept by default (`STRIP_DIACRITICS=false`) — important for classical
  and religious Arabic texts; flip to `true` if it helps retrieval on your corpus.

See `CONTRACT.md` for the precise module API.
