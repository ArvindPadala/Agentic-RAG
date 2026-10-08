---
title: Agentic Document RAG
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Agentic-RAG: Production-Ready Retrieval with Visual Grounding

**Live Deployment:** [Try it on Hugging Face Spaces!](https://huggingface.co/spaces/ArvindPadala/Agentic-Document-RAG)

A comprehensive, production-grade Agentic RAG (Retrieval-Augmented Generation) pipeline. This system moves beyond basic vector search by implementing **Hybrid Search (Vector + BM25 + Cross-Encoder Reranking)**, query decomposition, self-corrective retrieval, a live faithfulness guardrail, layout-aware document parsing, a resilient LLM routing system, and automated evaluation pipelines.

---

## 1. System Architecture

The architecture addresses the common failure modes of standard RAG implementations (destructive chunking, poor lexical matching, passive generation, and ungrounded hallucinations).

### 1.1 Layout-Aware Ingestion
Documents are ingested asynchronously through AWS Lambda into the **LandingAI ADE (Document Parsing)** model. This extracts text alongside precise spatial bounding boxes, preserving the visual layout and structural hierarchy of the document prior to embedding.

### 1.2 Multi-Stage Hybrid Retrieval
The retrieval engine (`hybrid_search.py`) implements a 3-stage pipeline:
1. **Dense Retrieval:** Queries ChromaDB using `sentence-transformers/all-MiniLM-L6-v2` for semantic meaning.
2. **Sparse Retrieval:** Queries a local `BM25Okapi` index for exact keyword and lexicon matching.
3. **Reciprocal Rank Fusion (RRF) & Reranking:** Mathematically merges the Dense and Sparse ranks, then passes the top fused results through a Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) to accurately score the contextual relationship between query and chunk.

### 1.3 Agentic Orchestration
The core ReAct Agent is orchestrated using a **LangGraph StateGraph** (`agent.py`).
- **Query Decomposition:** Complex user questions are mathematically decomposed into an array of orthogonal sub-queries via a pre-processing LLM layer (`query_optimizer.py`).
- **Parallel Execution:** When the agent requests multiple searches, the `execute_tools` node uses a `ThreadPoolExecutor` to run I/O-heavy ChromaDB/S3 searches concurrently.
- **Self-Correction:** The agent loop runs up to 7 iterations. After each retrieval, the agent evaluates its own context against a 6-point Self-Correction Protocol, formulating new queries if context is insufficient.

### 1.4 Visual Grounding
To prevent hallucinations, the system provides visual proof. Using the bounding boxes from the ingestion phase, PyMuPDF crops the exact region of the source PDF, uploads it to AWS S3, and returns a time-limited presigned URL directly in the UI.

### 1.5 Resilient LLM Routing & Guardrails
- **Router (`llm_router.py`):** Handles free-tier API rate limits via key rotation, graceful model degradation (e.g., `gemini-3.5-flash` fallback), and jittered exponential backoff.
- **Live Guardrail (`live_guardrail.py`):** An LLM-as-a-judge interceptor that evaluates generated responses against retrieved context to detect hallucinations before streaming to the user.

---

## 2. Evaluation & Metrics

The system relies on a Two-Tier Evaluation Strategy located in the `eval/` directory.

### Tier 1: Hardware-Bound Retrieval Metrics
Measures pure retrieval accuracy on a ~100-question Golden Dataset using non-generative metrics.
* **MRR@5:** 0.8818 (+13% improvement over standard Vector Search).
* **Recall@1:** 0.8462 (The exact document chunk is retrieved first ~85% of the time).

### Tier 2: LLM-Bound Generation Metrics
Evaluates a smoke-test subset via the **Ragas** framework using a local `llama3.2:3b` Ollama judge to circumvent strict API limits.
* **Faithfulness:** The Reranker provided highly accurate context, boosting the LLM's Faithfulness score from 0.66 to **0.77**, heavily reducing hallucinations.

---

## 3. Project Structure

```text
Agentic-RAG/
├── agent.py                      # LangGraph StateGraph, Nodes, and Tool definitions
├── app.py                        # Gradio Web UI with streaming/visual grounding
├── llm_router.py                 # Resilient GenAI Client (Rotation, Fallback)
├── live_guardrail.py             # Runtime LLM-as-a-judge Faithfulness Interceptor
├── query_optimizer.py            # Pre-processing LLM layer for Query Decomposition
├── hybrid_search.py              # BM25 + Vector Search + RRF + Reranker
├── gemini_helpers.py             # ChromaDB abstractions
├── lambda_helpers.py             # AWS Infrastructure automation (S3, IAM, Lambda)
├── visual_grounding_helper.py    # PyMuPDF rendering and S3 upload logic
├── ade_s3_handler.py             # Lambda handler for LandingAI ADE parsing
├── eval/                         # Two-Tier Evaluation Pipeline (MRR, Ragas, Golden Datasets)
├── tests/                        # Pytest Suite (Unit, Integration, E2E)
└── .github/workflows/ci.yml      # GitHub Actions CI configuration
```

---

## 4. Local Deployment & Usage

### Runtime privacy boundary

The web application now defaults to `APP_MODE=public_demo`. This mode:
- uses only browser-session chat context and never loads or saves `memory.json`;
- searches chunks explicitly marked `visibility=public` in both vector and
  hybrid retrieval; private and unclassified legacy records are excluded;
- rejects uploads and memory saves server-side, including direct handler calls;
- rejects non-empty file bodies at Gradio's framework upload route as well;
- requires an exact approved `source_pdf_key` for visual citations;
- disables content-bearing LangSmith tracing and hides internal error details.

Existing documents are **not automatically published**. Until a reviewed corpus
is explicitly approved, the public app returns no documents. Keep database and
private data artifacts out of deployment packages; metadata filters are not a
replacement for safe artifact packaging.

Use the offline administrative tool on a reviewed local index, replacing these
example IDs and source key with your explicitly approved selections:

```bash
python manage_public_corpus.py --publish --ids chunk-id-1 chunk-id-2 \
  --source-pdf-key input/documents/approved-public-document.pdf
python manage_public_corpus.py --revoke --ids chunk-id-1 chunk-id-2
```

The tool updates metadata only, requires existing IDs, and never publishes a
whole collection implicitly. Review both text and its corresponding PDF before
publication. New ingestion is private by default. Public BM25 views rebuild
from filtered records, so an unrestricted index or stale visibility cache cannot
be reused. Already-issued S3 URLs remain valid until their expiry.

For the existing single-user upload and personal-memory workflow, set
`APP_MODE=local_private` and run `python app.py`. This mode binds to `127.0.0.1`,
rejects `--share`, and refuses to start in a Hugging Face Space. It is not an
authenticated multi-user deployment. Compose publishes to loopback as well;
its default public mode can still listen within the container. A loopback-only
process running in local/private mode is best started directly on the host.

Authenticated uploads and recoverable permanent accounts remain follow-on work.
No external account or existing corpus has been migrated by these changes.

**Optional private demo memory is now implemented**, behind
`SUPABASE_ENABLED=true` and the setup described in [SUPABASE_SETUP.md](SUPABASE_SETUP.md).
Ordinary visitors can still chat without signing up. Opting into a private demo
session uses Supabase anonymous Auth and a CAPTCHA challenge, then stores memory
under that verified identity. Sessions expire within one hour; recoverable
permanent accounts and hosted uploads remain follow-on work. The running Space
has not been deployed or configured by these local changes.

Deployment packaging is source-only: `deploy.py` uses an exact allowlist, and
the Dockerfile copies explicit runtime files. Neither copies your local Chroma
database, documents, memory, or notebooks. A fresh deployment therefore needs
an explicitly provisioned approved corpus or the upcoming durable datastore;
the source upload does not migrate data. It also does not remove data already
present in an existing remote Space. Audit and explicitly clean those artifacts
before a public release. Importing `deploy.py` no longer triggers a deployment.

### Prerequisites
- Python 3.10+
- AWS Account (S3, Lambda, IAM permissions)
- Google GenAI API Key (Supports multiple for rotation)
- LandingAI Vision Agent API Key

### Installation
```bash
git clone https://github.com/ArvindPadala/Agentic-RAG.git
cd Agentic-RAG
python -m pip install -r requirements.txt
cp .env.example .env
# Configure environment variables in .env
```

### Usage
```bash
# Run tests and linter
make lint
make test

# Launch the interactive Gradio Web Application
python app.py

# Launch the CLI Agent (Headless)
python agent.py -q "Explain the self-attention mechanism in Transformers."

# Launch with Query Decomposition + Self-Correction
python agent.py -q "What is RRF and how does cross-encoder reranking improve it?" --use-decomposition

# Test the Live Faithfulness Guardrail with a hallucination-inducing prompt
python agent.py -q "What is the capital of France?" --use-guardrail
```

---

## 5. Future Scalability

To transition from this prototype to a distributed enterprise environment:
1. **Frontend / API:** Replace the single-threaded Gradio interface with a dedicated frontend (Next.js) and a highly concurrent backend (FastAPI).
2. **Lexical Index:** Replace the in-memory `rank_bm25` index with a distributed search cluster (Elasticsearch or OpenSearch).
3. **Vector Database:** Migrate from local ChromaDB to a managed, distributed vector store (Pinecone, Milvus, Qdrant).
4. **GPU Inference:** Move the Cross-Encoder off the local CPU to a dedicated GPU instance (NVIDIA Triton or vLLM).
