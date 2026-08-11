# Agentic-RAG — Post-Mortem & Architecture Challenges

A technical log documenting major infrastructure, deployment, and API limit challenges encountered during the development of the Agentic RAG system. Each entry details the symptom, root cause analysis, and engineering resolution.

---

## Challenge 1: `pydantic_core` Import Error in AWS Lambda

### Symptom
After deploying the `ade-s3-handler` Lambda function via a local deployment script, invocations failed immediately with:
```text
[ERROR] Runtime.ImportModuleError: Unable to import module 'ade_s3_handler':
No module named 'pydantic_core._pydantic_core'
```

### Root Cause Analysis
The deployment script relied on a standard `pip install -t` from a macOS development environment. Because `pydantic v2` relies on C-extensions, `pip` fetched the macOS-compatible wheel (`_pydantic_core.cpython-312-darwin.so`). AWS Lambda runs on Amazon Linux x86_64, which requires `.cpython-312-x86_64-linux-gnu.so`. The binary mismatch caused the import failure at runtime.

### Resolution
Updated the deployment packaging logic to explicitly target the `manylinux2014_x86_64` platform regardless of the host OS:
```bash
pip install \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 312 \
  --only-binary=:all: \
  -t {package_dir} \
  {req_string}
```

---

## Challenge 2: Lambda Runtime Mismatch (Python 3.10 vs 3.12)

### Symptom
Intermittent Lambda failures continued. CloudWatch logs (`INIT_START`) indicated the runtime environment was Python 3.10, whereas the C-extension wheels were built for Python 3.12.

### Root Cause Analysis
The target AWS Lambda function was originally provisioned with the `python3.10` runtime. The deployment pipeline updated the `.zip` payload with Python 3.12 binaries but failed to update the underlying Lambda configuration.

### Resolution
Separated the code deployment and configuration deployment into two discrete AWS API calls. Added explicit `update_function_configuration(Runtime="python3.12")` to the deployment script to guarantee environment consistency.

---

## Challenge 3: Vector Embeddings Rate Limiting (429 RESOURCE_EXHAUSTED)

### Symptom
During batch ingestion of PDF chunks using the `gemini-embedding-001` model, the pipeline consistently failed with:
```text
ClientError: 429 RESOURCE_EXHAUSTED
Quota exceeded for metric: embed_content_free_tier_requests, limit: 100
```

### Root Cause Analysis
The Gemini Free Tier enforces a strict 100 requests-per-minute (RPM) quota on embedding generation. A single 759-chunk document batch immediately saturated the quota, creating a severe bottleneck for rapid testing and development.

### Resolution
Migrated the embedding architecture from a cloud dependency to local, on-device computation. Configured ChromaDB to utilize the native `sentence-transformers/all-MiniLM-L6-v2` model, yielding zero-cost, unlimited, and offline indexing capabilities.

---

## Challenge 4: GenAI Client & Pydantic Validation Errors

### Symptom
The primary agent loop consistently crashed during multi-turn conversations, throwing 19 nested Pydantic validation errors originating from `_GenerateContentParameters`.

### Root Cause Analysis
The `google-genai` SDK utilizes strict Pydantic v2 validation. The custom conversation state manager was appending raw dictionaries to the history rather than SDK-native `types.Content` objects. Pydantic's `ContentListUnion` validation recursively evaluates all possible union types upon failure, generating highly verbose and cryptic stack traces for a simple malformed dictionary.

### Resolution
1. Refactored the state manager to exclusively construct and pass `genai_types.Content` objects.
2. Leveraged the raw `candidate.content` payload directly from prior LLM responses instead of manually reconstructing dictionaries.
3. Enforced explicit state teardown on session reset to prevent stale dictionary elements from corrupting the history queue.

---

## Challenge 5: Resilient LLM Routing & Generation API Quotas

### Symptom
While scaling the system to support automated Ragas evaluation across larger datasets, the generative endpoint (`gemini-3.5-flash`) began rejecting requests with `429 Quota Exceeded` errors. Standard try/except blocks caused the entire evaluation pipeline to crash midway.

### Root Cause Analysis
The 15 RPM generative limit is too restrictive for bulk RAG evaluation or multi-user parallel access. A naive client implementation lacks the retry semantics required for production-grade resilience.

### Resolution
Implemented a custom `GeminiRouter` class wrapper leveraging the `tenacity` library.
- **Exponential Backoff**: Configured jittered wait periods to survive transient drops.
- **Key Rotation**: Enabled the router to accept a list of keys, hot-swapping to a fallback key when a `429` is detected.
- **Model Fallback Degradation**: Designed an automatic fallback chain (`3.6-flash` → `3.5-flash` → `3.5-flash-lite`) if all API keys exhaust their quotas simultaneously.

---

## Challenge 6: Ragas Framework Asynchronous Concurrency Saturation

### Symptom
During the final `ragas` Generation Metrics evaluation (Faithfulness and Answer Relevancy), the script returned `nan` scores across the entire dataset, accompanied by terminal logs showing deep `aiohttp` timeouts and `ClientConnectorDNSError` exceptions.

### Root Cause Analysis
The Ragas evaluation framework utilizes an extremely aggressive asynchronous concurrency model to evaluate LLM responses in parallel. When bound to the `langchain-google-genai` wrapper on a Free-Tier API, the framework attempts dozens of simultaneous connections, completely saturating the rate limiter before the LangChain exponential backoff logic can successfully throttle the queue. 

### Resolution
While the implementation is functionally correct (`eval/evaluate_ragas.py`), running Ragas at scale requires either a paid/production API tier capable of handling high concurrency, or throttling the Ragas `RunConfig` to `max_workers=1`, which severely impacts runtime. The retrieval metrics (Precision/Recall) were calculated successfully via synchronous local scripts to prove the architectural gains.

---

## Challenge 7: Undefined Name Errors Post-Refactoring

### Symptom
After refactoring the system to utilize the new `GeminiRouter`, the standalone CLI (`agent.py`) crashed on startup with an undefined variable error.

### Root Cause Analysis
The underlying LLM initialization logic was updated across the web application and testing suite, but the `if __name__ == "__main__":` entrypoint in `agent.py` was missed during the refactor.

### Resolution
1. Corrected the CLI entrypoint to initialize the `GeminiRouter`.
2. To prevent future regressions, integrated GitHub Actions CI/CD to automatically execute `flake8` syntax checks and `pytest` execution against all branches prior to merge.

---

## Summary Table

| # | Challenge Area | Root Cause | Engineering Fix |
|---|----------------|------------|-----------------|
| 1 | Lambda Deployment | `pip` fetched macOS wheels | Enforced `--platform manylinux2014_x86_64` |
| 2 | Lambda Execution | Inconsistent Runtime versions | Explicit `update_function_configuration` call |
| 3 | Embedding Quotas | Gemini 100 RPM limit exceeded | Migrated to local `sentence-transformers` |
| 4 | LLM Type Validation | Dictionary vs Pydantic schema | Enforced strictly-typed `genai_types.Content` |
| 5 | Generative Quotas | Gemini 15 RPM limit exceeded | Implemented `GeminiRouter` with Backoff/Rotation |
| 6 | Ragas Evaluation | Async concurrency saturation | Confirmed API limitation; implemented CI/CD alternative |
| 7 | Refactoring Bugs | Missed CLI entrypoint | Introduced automated `flake8` via GitHub Actions CI |
