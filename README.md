# Agentic RAG System with Visual Grounding

> A production-ready **Agentic RAG** (Retrieval-Augmented Generation) pipeline. This system processes complex PDFs via LandingAI ADE, indexes semantic and keyword chunks locally, and orchestrates a resilient Gemini-powered ReAct agent. It features **Hybrid Search (Vector + BM25)**, an automated evaluation suite (Ragas), a robust LLM API router, and **visual grounding** to highlight source evidence in the original PDFs.

---

## Architecture Overview

This project implements a complete RAG lifecycle, from asynchronous document ingestion to resilient LLM inference.

```text
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT INGESTION PIPELINE                  │
│  documents/*.pdf  →  S3 (input/)  →  AWS Lambda                 │
│                                       ↓                          │
│                              LandingAI ADE (parsing)            │
│                                       ↓                          │
│                         S3 (output/chunks/*.json)               │
│                                       ↓                          │
│    ChromaDB (Vector) ← SentenceTransformers (all-MiniLM-L6-v2)  │
│    BM25Okapi (Lexical) ← NLTK Tokenization                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      INFERENCE & AGENT PIPELINE                 │
│  User Query → LLM Router (Exponential Backoff, Key Rotation)    │
│                       ↓                                         │
│              ReAct Agent Loop (Tool Use)                        │
│                       ↓                                         │
│  Hybrid Search (Reciprocal Rank Fusion: BM25 + Cosine Sim)     │
│                       ↓                                         │
│  Agent synthesizes → Cited Answer + S3 Pre-signed Visual Crop   │
└─────────────────────────────────────────────────────────────────┘
```

### Core Technologies
- **Compute / Orchestration**: AWS Lambda (Python 3.12, containerized/Manylinux), GitHub Actions (CI)
- **Document Parsing**: LandingAI ADE (Bounding boxes + Layout extraction)
- **Retrieval Engine**: ChromaDB (Vector) + Rank_BM25 (Lexical) + RRF (Fusion)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local execution)
- **LLM / Agent**: Google Gemini API via custom resilient `GeminiRouter`
- **Evaluation**: Ragas, PyTest

---

## Technical Implementations & Engineering Results

### 1. Hybrid Search (Vector + Lexical)
Standard cosine similarity struggles with exact keyword matching (e.g., acronyms, IDs). We implemented a Hybrid Search module (`hybrid_search.py`) that executes parallel queries against ChromaDB (Dense) and BM25Okapi (Sparse), merging results via **Reciprocal Rank Fusion (RRF)**.

**Evaluation Results (Measured via automated testset generation):**
- **MRR@5**: Improved from 0.7949 (Vector) to **0.8846** (Hybrid) — a **+11.2%** increase.
- **Recall@1**: Improved from 0.7692 to **0.8462** (The exact context is retrieved on the first try ~85% of the time).

### 2. Resilient LLM Routing (`llm_router.py`)
To achieve production reliability on rate-limited LLM tiers (e.g., Gemini Free Tier), we implemented a custom client router utilizing `tenacity`.
- **Key Rotation**: Automatically hot-swaps between API keys (e.g., `GEMINI_API_KEY`, `GEMINI_API_KEY_2`) upon encountering `429 Quota Exceeded` errors.
- **Model Fallback**: Gracefully degrades down a predefined chain (`gemini-3.6-flash` → `3.5-flash` → `3.5-flash-lite`) if all keys are exhausted.
- **Exponential Backoff**: Jittered retry loops to survive transient `503` service unavailability.

### 3. Automated Evaluation Pipeline (`eval/`)
Rather than relying on qualitative spot-checks, the system includes a quantitative evaluation pipeline:
- `generate_testset.py`: Uses the LLM to autonomously generate a "Golden Dataset" of challenging queries based directly on the ingested ChromaDB chunks.
- `calculate_retrieval_metrics.py`: Computes Precision@K, Recall@K, and MRR.
- `evaluate_ragas.py`: Implements the `ragas` framework to mathematically score generation **Faithfulness** (hallucination checks) and **Answer Relevancy**.

### 4. CI/CD Pipeline
The repository enforces code quality and functionality via GitHub Actions (`.github/workflows/ci.yml`):
- Executes `flake8` for strict PEP8 compliance and syntax validation.
- Runs the complete `pytest` suite covering unit, integration, and end-to-end agent workflows on every push to `main`.

---

## Project Structure

```text
Agentic-RAG/
├── agent.py                      # Standalone CLI agent entry point
├── app.py                        # Gradio Web UI (Chat + Visual Grounding)
├── llm_router.py                 # Resilient GenAI Client (Rotation, Fallback)
├── hybrid_search.py              # BM25 + Vector Search + Reciprocal Rank Fusion
├── gemini_helpers.py             # ChromaDB abstractions, tool definitions
├── lambda_helpers.py             # AWS Infrastructure automation (S3, IAM, Lambda)
├── visual_grounding_helper.py    # PyMuPDF rendering and S3 upload logic
├── ade_s3_handler.py             # Lambda handler for LandingAI ADE parsing
├── eval/                         # Evaluation Pipeline
│   ├── generate_testset.py       # Golden Dataset generator
│   ├── calculate_retrieval_metrics.py
│   ├── evaluate_baseline.py      # Runs testset using Vector only
│   ├── evaluate_hybrid.py        # Runs testset using Hybrid Search
│   └── evaluate_ragas.py         # Ragas generation metric scoring
├── tests/                        # Pytest Suite
│   ├── test_unit.py
│   ├── test_integration.py
│   └── test_e2e.py
├── .github/workflows/ci.yml      # GitHub Actions CI configuration
├── Makefile                      # Make targets (lint, test)
├── EVALUATION_REPORT.md          # Detailed benchmarking results
└── project_challenges.md         # Historical debugging and architecture logs
```

---

## Local Deployment & Usage

### Prerequisites
- Python 3.12
- AWS Account (S3, Lambda, IAM)
- Google GenAI API Key (Supports multiple via `GEMINI_API_KEY_2`)
- LandingAI Vision Agent API Key

### Setup
```bash
git clone https://github.com/ArvindPadala/Agentic-RAG.git
cd Agentic-RAG
python -m pip install -r requirements.txt
cp .env.example .env
```

### Running the System
```bash
# Run tests and linter
make lint
make test

# Launch the Gradio Web Application
python app.py

# Launch the CLI Agent (Headless)
python agent.py -q "Explain the revenue growth mentioned in the Q3 report."
```

## Challenges & Architecture Decisions
For a detailed post-mortem on early design challenges—including cross-platform packaging for AWS Lambda C-extensions, API rate limit exhaustion, and Pydantic validation errors—refer to [`project_challenges.md`](./project_challenges.md).
