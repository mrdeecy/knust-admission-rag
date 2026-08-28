# KNUST Admission GPT

A production-ready Retrieval-Augmented Generation (RAG) system that answers questions about **Kwame Nkrumah University of Science and Technology (KNUST)** undergraduate admissions — entry requirements, programmes, cut-off aggregates, application steps, and fees.

The assistant is grounded in KNUST's official admissions document and cites its sources. If an answer isn't in the source material, it says so rather than guessing.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit  │────▶│    FastAPI       │────▶│  Pinecone       │
│   (UI)      │     │   (Backend)      │     │  (Vector DB)    │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
                    ┌──────────────────┐              │
                    │   OpenAI         │◀─────────────┘
                    │   (Embeddings +  │
                    │    Generation)   │
                    └──────────────────┘
```

### Retrieval Pipeline (AdvancedRetriever)

The retriever combines multiple techniques for high precision:

| Stage | Technique | Purpose |
|-------|-----------|---------|
| 1 | **Query Rewrite** | Normalize messy/conversational questions into clean search queries |
| 2 | **Multi-Query Expansion** | Generate alternative phrasings to improve recall |
| 3 | **Hybrid Search** | Dense (Pinecone) + Sparse (BM25) retrieval per phrasing |
| 4 | **Reciprocal Rank Fusion (RRF)** | Fuse all result lists into one ranked pool |
| 5 | **Cross-Encoder Rerank** | Pinecone-hosted `bge-reranker-v2-m3` for final precision |

### Key Features

- **Conversational context** — Follow-up questions ("what about the Obuasi campus version?") are resolved into standalone queries before retrieval
- **Source citations** — Every answer includes the retrieved chunks with relevance scores
- **Quality testing** — RAGAS-based regression suite (`context_recall`, `faithfulness`, `answer_relevancy`)
- **Cost transparency** — Token counting and estimated embedding costs logged during indexing

---

## Project Structure

```
rag_project/
├── rag_deploy/                 # Production deployment code
│   ├── main.py                 # FastAPI backend (POST /query, GET /health)
│   ├── retrieval_pipeline.py   # AdvancedRetriever + generation logic
│   ├── indexing_pipeline.py    # PDF → chunks → embeddings → Pinecone
│   └── streamlit_app.py        # Chat UI (talks to FastAPI via HTTP)
├── rag_test/                   # Quality regression suite
│   ├── test_rag_quality.py     # RAGAS metrics on golden set
│   ├── retrieval_pipeline.py   # Test-specific retriever config
│   ├── golden_set.json         # Q/A pairs with ground truth
│   └── conftest.py             # Pytest fixtures
├── rag_app/                    # Notebook exploration
│   ├── rag_indexing_pipeline.ipynb
│   └── rag_retrieval_pipeline.ipynb
├── simple_rag_app/             # Simpler prototypes (OpenAI / Ollama)
├── data/
│   └── admission_requirement.pdf  # Source document
├── overview/                   # Project documentation / course materials
├── .env                        # Environment variables (not committed)
├── pyproject.toml              # Dependencies (uv)
├── requirements.txt            # Minimal pip dependencies
└── uv.lock                     # Locked dependency versions
```

---

## Prerequisites

- **Python 3.13+**
- **Pinecone account** (serverless indexes)
- **OpenAI API key** (embeddings + chat)
- **uv** (recommended) or pip

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd rag_project

# Using uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_key
OPENAI_API_KEY=your_openai_key
INDEX_NAME=knust-admission-rag
SPARSE_INDEX_NAME=knust-rag-sparse
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
CHAT_MODEL=gpt-4o-mini
```

### 3. Build the index (run once)

```bash
cd rag_deploy
python indexing_pipeline.py
```

This will:
- Load and clean the KNUST admissions PDF
- Chunk with `RecursiveCharacterTextSplitter` (500 chars, 50 overlap)
- Embed with `text-embedding-3-small`
- Create two Pinecone indexes: dense (cosine) + sparse (BM25)
- Upsert all chunks to both indexes

### 4. Start the backend

```bash
# From rag_deploy/
uvicorn main:app --reload --port 8000
```

API docs available at `http://127.0.0.1:8000/docs`

### 5. Launch the UI

```bash
# From rag_deploy/ (in a new terminal)
streamlit run streamlit_app.py
```

Open `http://localhost:8501` and start asking questions.

---

## API Reference

### `GET /health`

Health check endpoint.

**Response:**
```json
{ "status": "ok" }
```

### `POST /query`

Ask a question.

**Request:**
```json
{
  "question": "What are the WASSCE requirements for BSc Computer Science?",
  "top_k": 3,
  "history": [
    { "role": "user", "content": "Tell me about engineering programmes" },
    { "role": "assistant", "content": "KNUST offers several engineering programmes..." }
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | Yes | The applicant's question |
| `top_k` | integer | No | Number of cited chunks (1-10, default: retriever's configured value) |
| `history` | array | No | Prior conversation turns for follow-up resolution |

**Response:**
```json
{
  "answer": "For BSc Computer Science, WASSCE applicants need... [1][2]",
  "sources": [
    {
      "chunk_id": "KNUST_admission_requirements_pdf_005",
      "title": "Faculty of Computing and Mathematical Sciences",
      "text": "BSc Computer Science: WASSCE passes in English, Core Mathematics...",
      "score": 0.847
    }
  ]
}
```

---

## Running Tests

### Unit / Integration Tests (fast, no LLM)

```bash
cd rag_test
pytest -m "not llm" -v
```

### RAG Quality Regression (requires LLM + Pinecone)

```bash
cd rag_test
pytest -m llm -v
```

This runs the golden set through the full pipeline and asserts RAGAS metrics:
- `context_recall >= 0.5`
- `faithfulness >= 0.5`
- `answer_relevancy >= 0.5`

---

## Configuration

Key environment variables (in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `INDEX_NAME` | `knust-admission-rag` | Pinecone dense index name |
| `SPARSE_INDEX_NAME` | `knust-rag-sparse` | Pinecone sparse (BM25) index name |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING_DIM` | `1536` | Embedding dimensions |
| `CHAT_MODEL` | `gpt-4o-mini` | Generation model |
| `RERANK_MODEL` | `bge-reranker-v2-m3` | Pinecone hosted reranker |

Retriever knobs (in `AdvancedRetriever.__init__`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | `3` | Final chunks returned after rerank |
| `candidates` | `10` | Candidates per search before RRF |

---

## Data Source

The knowledge base is built from **KNUST's official undergraduate admissions requirements document** (`data/admission_requirement.pdf`), covering:

- Entry requirements by qualification (WASSCE, A-Level, IB, HND, Mature)
- Programme-specific subject combinations and cut-off aggregates
- Application procedures, deadlines, and fees
- Campus-specific requirements (Main campus, Obuasi, etc.)

---

## Deployment Notes

### Backend (FastAPI)
- Stateless — scale horizontally behind a load balancer
- Set `KNUST_API_URL` in the Streamlit config to point to the deployed backend
- CORS enabled for all origins (`allow_origins=["*"]`)

### Frontend (Streamlit)
- Pure client-side — talks to backend via HTTP only
- Deploy to Streamlit Community Cloud, Render, Railway, etc.

### Vector Database (Pinecone)
- Serverless indexes (AWS us-east-1)
- Dense index: cosine similarity, 1536 dimensions
- Sparse index: Pinecone integrated BM25 (`pinecone-sparse-english-v0`)

---

## Cost Estimation

| Operation | Model | Cost (approx) |
|-----------|-------|---------------|
| Embedding (indexing) | text-embedding-3-small | $0.02 / 1M tokens |
| Embedding (query) | text-embedding-3-small | ~500 tokens/query |
| Generation | gpt-4o-mini | $0.15 / 1M input, $0.60 / 1M output |
| Rerank | bge-reranker-v2-m3 | Included in Pinecone serverless |

Typical query: ~2K input tokens (context + history) + ~300 output tokens ≈ **$0.001/query**

---

## License

MIT License — see `LICENSE` file for details.

---

## Acknowledgments

- **Pinecone** for vector database and hosted reranking
- **OpenAI** for embeddings and generation models
- **LangChain** for document loading and text splitting
- **RAGAS** for quality evaluation framework
- **KNUST** for the admissions reference document
