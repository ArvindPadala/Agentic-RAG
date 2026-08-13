# Agentic-RAG: Production-Ready Retrieval with Visual Grounding

> A comprehensive, production-grade Agentic RAG (Retrieval-Augmented Generation) pipeline. This system moves beyond basic vector search by implementing **Elite Hybrid Search (Vector + BM25 + Cross-Encoder Reranking)**, layout-aware document parsing, a resilient LLM routing system, and automated evaluation pipelines.

---

## 🛑 The Problem with "Standard" RAG
The industry-standard RAG tutorial (PDF → LangChain TextSplitter → Vector DB → LLM) fails in production. 
1. **Dumb Chunking:** Splitting by character count destroys tables, diagrams, and layout context.
2. **Dense-Only Retrieval:** Cosine similarity struggles with exact keyword matching (e.g., SKUs, IDs, acronyms).
3. **Passive Generation:** The LLM is forced to answer based *only* on a single retrieved context, even if the user's prompt requires multi-hop reasoning or multiple separate searches.
4. **Hallucination:** Without strict attribution, the LLM hallucinates facts.

## 🏗️ Our Architecture & Solutions

We engineered this system to address those exact failure points.

### 1. Layout-Aware Ingestion (LandingAI)
Instead of blindly splitting text, we ingest PDFs asynchronously through AWS Lambda into **LandingAI's ADE (Document Parsing)** model. This extracts text alongside spatial bounding boxes, preserving the visual layout and hierarchy of the document before storing the chunks locally.

### 2. "Elite" Hybrid Retrieval (Vector + Sparse + Reranker)
We abandoned single-path vector retrieval. Our `hybrid_search.py` implements a 3-stage pipeline:
1. **Dense Retrieval:** Queries ChromaDB using `sentence-transformers/all-MiniLM-L6-v2` for semantic meaning.
2. **Sparse Retrieval:** Queries a local `BM25Okapi` index for exact keyword and lexicon matching.
3. **Reciprocal Rank Fusion (RRF):** Mathematically merges the Dense and Sparse ranks.
4. **Cross-Encoder Reranking:** Passes the top fused results through a heavy Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) to accurately score the contextual relationship between the query and the chunk. 

### 3. Agentic LLM Orchestration
Instead of a passive prompt chain, we implemented a **ReAct Agent**. The Gemini model is provided a `search_knowledge_base` tool. It autonomously decides *whether* to search, *what* queries to formulate, and *when* it has enough information to stop searching and generate a final answer.

### 4. Visual Grounding & S3 Presigned URLs
To prevent hallucinations and build trust, the Agent doesn't just cite its sources—it provides visual proof. Using the bounding boxes from the ingestion phase, the system uses PyMuPDF to crop the exact region of the source PDF, uploads it to AWS S3, and returns a time-limited presigned URL directly in the UI.

### 5. Resilient LLM Routing (`llm_router.py`)
To survive rate-limited free-tier APIs in production, we built a custom client router:
- **Key Rotation**: Hot-swaps between API keys (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`) on `429 Quota Exceeded` errors.
- **Model Fallback**: Gracefully degrades (e.g., `3.6-flash` → `3.5-flash`) if primary endpoints fail.
- **Exponential Backoff**: Jittered retry loops to survive `503` service drops.

---

## 📊 Evaluation & Metrics (The Proof)

We don't rely on vibes. The `eval/` directory contains an automated suite powered by **Ragas** and an LLM-as-a-judge (`llama3.1:8b` via Ollama) to continuously benchmark the architecture.

In our latest smoke-test evaluation against the `transformer` and `rag` academic papers:
* **Answer Relevancy:** The Elite Hybrid Reranker boosted Answer Relevancy to **0.88** (up from the Baseline Vector's 0.69). The Cross-Encoder successfully surfaced the most semantically relevant documents to the top.
* **Faithfulness Tuning:** We noted a drop in faithfulness when the reranker was configured too strictly (filtering out background context). The architecture exposes `n_results` tuning to dynamically adjust this tradeoff.
* **Retrieval Accuracy:** Achieved **1.000 MRR@5** and **Recall@5** on ground-truth document chunks.

---

## 🛠️ Project Structure

```text
Agentic-RAG/
├── agent.py                      # ReAct Agent loop and Tool definitions
├── app.py                        # Gradio Web UI with streaming/visual grounding
├── llm_router.py                 # Resilient GenAI Client (Rotation, Fallback)
├── hybrid_search.py              # BM25 + Vector Search + RRF + Reranker
├── gemini_helpers.py             # ChromaDB abstractions
├── lambda_helpers.py             # AWS Infrastructure automation (S3, IAM, Lambda)
├── visual_grounding_helper.py    # PyMuPDF rendering and S3 upload logic
├── ade_s3_handler.py             # Lambda handler for LandingAI ADE parsing
├── eval/                         # Evaluation Pipeline (Ragas, MRR, Recall)
├── tests/                        # Pytest Suite (Unit, Integration, E2E)
└── .github/workflows/ci.yml      # GitHub Actions CI configuration
```

---

## 🚀 Local Deployment & Usage

### Prerequisites
- Python 3.12
- AWS Account (S3, Lambda, IAM)
- Google GenAI API Key (Supports multiple)
- LandingAI Vision Agent API Key

### Setup
```bash
git clone https://github.com/ArvindPadala/Agentic-RAG.git
cd Agentic-RAG
python -m pip install -r requirements.txt
cp .env.example .env # Configure your keys here
```

### Running the System
```bash
# Run tests and linter
make lint
make test

# Launch the interactive Gradio Web Application
python app.py

# Launch the CLI Agent (Headless)
python agent.py -q "Explain the self-attention mechanism in Transformers."
```
