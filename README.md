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

I engineered this system to address those exact failure points.

### 1. Layout-Aware Ingestion (LandingAI)
Instead of blindly splitting text, I ingest PDFs asynchronously through AWS Lambda into **LandingAI's ADE (Document Parsing)** model. This extracts text alongside spatial bounding boxes, preserving the visual layout and hierarchy of the document before storing the chunks locally.

### 2. "Elite" Hybrid Retrieval (Vector + Sparse + Reranker)
I abandoned single-path vector retrieval. My `hybrid_search.py` implements a 3-stage pipeline:
1. **Dense Retrieval:** Queries ChromaDB using `sentence-transformers/all-MiniLM-L6-v2` for semantic meaning.
2. **Sparse Retrieval:** Queries a local `BM25Okapi` index for exact keyword and lexicon matching.
3. **Reciprocal Rank Fusion (RRF):** Mathematically merges the Dense and Sparse ranks.
4. **Cross-Encoder Reranking:** Passes the top fused results through a heavy Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) to accurately score the contextual relationship between the query and the chunk. 

### 3. Query Decomposition (Pre-processing)
Raw user questions are often vague or multi-part. Before hitting the search index, I implemented a fast LLM layer (`query_optimizer.py` using `gemini-3.5-flash-lite`) to mathematically decompose complex questions into an array of orthogonal sub-queries. 

### 4. Agentic LLM Orchestration
Instead of a passive prompt chain, I implemented a **ReAct Agent**. The Gemini model is provided a `search_knowledge_base` tool and the decomposed sub-queries. It autonomously decides *whether* to search, *what* queries to formulate, and *when* it has enough information to stop searching and generate a final answer.

### 5. Visual Grounding & S3 Presigned URLs
To prevent hallucinations and build trust, the Agent doesn't just cite its sources—it provides visual proof. Using the bounding boxes from the ingestion phase, the system uses PyMuPDF to crop the exact region of the source PDF, uploads it to AWS S3, and returns a time-limited presigned URL directly in the UI.

### 6. Resilient LLM Routing (`llm_router.py`)
To survive rate-limited free-tier APIs in production, I built a custom client router:
- **Key Rotation**: Hot-swaps between API keys (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`) on `429 Quota Exceeded` errors.
- **Model Fallback**: Gracefully degrades (e.g., `3.6-flash` → `3.5-flash`) if primary endpoints fail.
- **Exponential Backoff**: Jittered retry loops to survive `503` service drops.

---

## 📊 Evaluation & Metrics (The Proof)

I don't rely on vibes. Evaluating complex RAG pipelines in production is notoriously difficult due to LLM-as-a-judge rate limits and latency. To solve this, I engineered a robust **Two-Tier Evaluation Strategy** located in the `eval/` directory.

### Tier 1: Hardware-Bound Retrieval Metrics (Fast & Free)
I measure the system's pure retrieval accuracy on a ~100-question Golden Dataset using non-generative metrics (BM25 & local embeddings). 
* **MRR@5:** The Elite Hybrid Reranker achieved **0.8818** (a +13% improvement over standard Vector Search).
* **Recall@1:** Reached **0.8462**, meaning the absolute perfect document chunk is retrieved first ~85% of the time.

### Tier 2: LLM-Bound Generation Metrics (Ragas Smoke Test)
For generation metrics, I evaluate a sliced smoke-test subset via **Ragas** using a local `llama3.2:3b` Ollama judge to circumvent strict API limits and avoid costly LLM calls.
* **Semantic Similarity:** Cosine similarity against ground truth improved across the board.
* **Faithfulness:** The Reranker provided highly accurate context, boosting the LLM's Faithfulness score to **0.77** (up from the Baseline's 0.66) by heavily reducing hallucinations.

---

## 🛠️ Project Structure

```text
Agentic-RAG/
├── agent.py                      # ReAct Agent loop and Tool definitions
├── app.py                        # Gradio Web UI with streaming/visual grounding
├── llm_router.py                 # Resilient GenAI Client (Rotation, Fallback)
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
