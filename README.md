# Agentic RAG System with Visual Grounding

> An end-to-end **agentic RAG** (Retrieval-Augmented Generation) pipeline that processes complex PDFs, indexes semantic chunks into a local vector database, and runs a Gemini-powered chatbot with **visual grounding** — highlighting the exact PDF regions that back every answer.

---

## What It Does

You ask a question about your documents in plain English. The agent:

1. **Calls a search tool** → queries indexed chunks from your documents
2. **Retrieves top-5 matches** → using cosine similarity over `sentence-transformers` embeddings
3. **Generates an answer** → Gemini 2.5 Flash reasons over the retrieved content
4. **Provides visual grounding** → returns a presigned S3 URL pointing to a cropped, highlighted image of the exact PDF region that contains the evidence

```
You: "What was the Q3 revenue for the enterprise segment?"
       ↓
Agent calls: search_knowledge_base(query="Q3 revenue enterprise segment")
       ↓
ChromaDB returns: 5 relevant chunks from financial reports
       ↓
Gemini synthesizes: Evidence-based answer with citations + page numbers
       ↓
Output: Answer + 🔍 Visual Reference: https://s3.aws.../chunk_image.png
```

---

## Architecture

![Architecture](images/architecture_1.png)

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT PIPELINE (one-time)                 │
│                                                                 │
│  documents/*.pdf  →  S3 (input/)  →  AWS Lambda                 │
│                                       ↓                          │
│                              LandingAI ADE (parsing)            │
│                                       ↓                          │
│                         S3 (output/chunks/*.json)               │
│                                       ↓                          │
│                    ChromaDB (local) ← sentence-transformers      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      AGENT (every query)                        │
│                                                                 │
│  User question → Gemini 2.5 Flash (function calling)           │
│                       ↓                                         │
│              search_knowledge_base()                            │
│                       ↓                                         │
│  ChromaDB query → top-5 chunks → visual grounding (S3 crop)    │
│                       ↓                                         │
│  Gemini synthesizes → cited answer + visual reference URLs      │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Serverless compute** | AWS Lambda (Python 3.12) | Triggers on S3 upload, runs ADE |
| **Document parsing** | LandingAI ADE | Extracts chunks with bounding boxes from PDFs |
| **Storage** | AWS S3 | PDFs, chunk JSONs, cropped chunk images |
| **Vector DB** | ChromaDB (local) | Stores and retrieves semantic embeddings |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Local, free, no rate limits |
| **LLM + Agent** | Google Gemini 2.5 Flash | Reasoning, function calling, response generation |
| **Visual grounding** | PyMuPDF + Pillow | Crops and highlights PDF regions |
| **Memory** | JSON on disk | Cross-session context persistence |

---

## Key Technical Highlights

- **Agentic function calling** — Gemini autonomously decides when and how many times to call the search tool before answering; implemented as a `while True` agent loop without any framework
- **Visual grounding** — every answer includes presigned S3 URLs to cropped, highlighted images of the exact PDF regions the evidence came from
- **Cross-platform Lambda packaging** — uses `--platform manylinux2014_x86_64 --python-version 312` pip flags to correctly build Linux-compatible C-extension wheels (e.g., `pydantic_core`) on macOS
- **Local embeddings** — switched from Gemini Embedding API (rate-limited) to ChromaDB's built-in `sentence-transformers` for unlimited local inference
- **Portable CLI** — `agent.py` runs the full agent from the terminal with `--question`, `--collection`, and `--model` flags; no Jupyter required

---

## Getting Started

### Prerequisites

- Python 3.12
- AWS account (free tier sufficient) with S3, Lambda, IAM, CloudWatch access
- [Google AI Studio](https://aistudio.google.com/) API key (free)
- [LandingAI](https://va.landing.ai/) Vision Agent API key (free tier)

### 1. Clone and install

```bash
git clone https://github.com/ArvindPadala/Agentic-RAG.git
cd Agentic-RAG
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your keys:
```

```env
GEMINI_API_KEY=your_gemini_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-2
S3_BUCKET=your-s3-bucket-name
VISION_AGENT_API_KEY=your_landingai_api_key
```

### 3. Run the document pipeline (one-time setup)

Open `pipeline_setup.ipynb` and run Steps 1–9. This:
- Deploys the Lambda function (Steps 3–5)
- Uploads PDFs to S3 and triggers document parsing (Steps 7)
- Indexes 759 chunks into ChromaDB with local embeddings (Steps 8–9)

> ⏩ **Skip Steps 3–9** if someone already ran them — `chroma_db/` is ready on disk.

### 4. Run the agent

```bash
# Interactive chat
python agent.py

# Single question
python agent.py -q "What are the symptoms of the common cold?"

# Use a different document collection
python agent.py --collection flu_chunks

# All options
python agent.py --help
```

---

## Project Structure

```
Agentic-RAG/
├── agent.py                    # Standalone CLI agent (no Jupyter required)
├── app.py                      # Gradio web UI with visual grounding panel
├── pipeline_setup.ipynb        # Step-by-step notebook (setup + exploration)
├── gemini_helpers.py           # ChromaDB init, search, memory, embedding utils
├── lambda_helpers.py           # Lambda deployment, S3 triggers, IAM role setup
├── visual_grounding_helper.py  # PDF cropping and S3 image upload for grounding
├── ade_s3_handler.py           # Lambda function: S3 trigger → LandingAI ADE → chunks
├── documents/                  # Example PDFs for the knowledge base
├── project_challenges.md       # Debugging log: issues faced and how they were resolved
└── .env.example                # Environment variable template
```

---

## Demo

![Gradio Web UI Demo](images/gradio_ui.png)

---

## Sample Output

```
You: "What was the year-over-year revenue growth for the enterprise segment?"

   🔧 search_knowledge_base(query='enterprise segment year over year revenue growth')

Agent: Based on the Q3 earnings report, the enterprise segment saw significant growth:

- **Revenue growth**: The enterprise segment grew by 24% year-over-year, driven by strong cloud adoption.
  (Q3_Financial_Results, Page 4)
  🔍 Visual Reference: https://...s3.amazonaws.com/...chunk_image.png

- **Operating margin**: The operating margin for this segment also improved by 300 basis points.
  (Q3_Financial_Results, Page 7)
  🔍 Visual Reference: https://...s3.amazonaws.com/...chunk_image.png
```

---

## Challenges & Debugging

See [`project_challenges.md`](./project_challenges.md) for a detailed log of every major issue encountered, how it was diagnosed, and how it was resolved — including:

- Lambda `pydantic_core` platform mismatch (macOS wheels on Linux Lambda)
- Gemini Embedding API rate limits → switched to local `sentence-transformers`
- Pydantic v2 union validation errors in the Gemini SDK's `generate_content`
- VS Code notebook caching overwriting programmatic file edits

---

## What I Learned

- **RAG architecture** from scratch — chunking, embedding, indexing, retrieval, synthesis
- **Agentic function calling** — implementing the agent loop manually without a framework (LangChain, LlamaIndex, etc.)
- **Cross-platform AWS Lambda packaging** — pip platform flags for binary wheels
- **Visual grounding** — connecting bounding box metadata from document parsing to rendered PDF crops
- **Debugging production AI systems** — isolating failures across cloud, SDK, and local layers

---

## License

Apache 2.0 — based on the [DeepLearning.AI Short Courses](https://www.deeplearning.ai/short-courses/) framework, modified and extended independently.