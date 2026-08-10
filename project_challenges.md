# Agentic-RAG — Project Challenges & Resolutions

A detailed log of every major technical challenge encountered while building the **Agentic RAG System with Gemini + ChromaDB**, how each was diagnosed, what the root cause was, and how it was fixed — including how AI-assisted debugging was used throughout.

---

## Challenge 1: `pydantic_core` Import Error in AWS Lambda

### What Was the Issue?

After deploying the Lambda function (`ade-s3-handler`) that processes medical PDFs using LandingAI ADE, CloudWatch logs showed this error every time the function was invoked:

```
[ERROR] Runtime.ImportModuleError: Unable to import module 'ade_s3_handler':
No module named 'pydantic_core._pydantic_core'
```

The Lambda function crashed immediately at import time — before processing a single PDF.

### What Was Happening?

`landingai-ade` (and many modern Python packages) depend on `pydantic v2`, which contains a C-extension module called `_pydantic_core.cpython-312-darwin.so`. The key word is **darwin** — that's a macOS binary.

The deployment workflow was using a plain `pip install -t ./package/` to collect dependencies. On a Mac, `pip` naturally fetches macOS-compatible wheels. These work locally, but **Lambda runs on Amazon Linux x86_64**, which needs a `.cpython-312-x86_64-linux-gnu.so` file instead.

When Lambda tried to import the macOS `.so` binary, it found the file physically present but the OS couldn't load it — hence "No module named `pydantic_core._pydantic_core`".

### How Was It Identified?

1. The error message `No module named 'pydantic_core._pydantic_core'` was a hint — the package *was* there (otherwise it would say "No module named `pydantic_core`"), but its C extension was wrong.
2. The AI agent (Antigravity) inspected the zip file contents using `unzip -l ade_lambda.zip | grep pydantic_core` and found `_pydantic_core.cpython-312-darwin.so` — the macOS binary.
3. Cross-referencing this with Lambda's Linux runtime confirmed the platform mismatch.

### How Was It Solved?

The `pip install` command in [lambda_helpers.py](./lambda_helpers.py) was updated to explicitly target Linux:

```python
# BEFORE (installs macOS binaries on Mac — wrong for Lambda):
pip install -t {package_dir} {req_string}

# AFTER (forces Linux x86_64 wheels regardless of host OS):
pip install \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 312 \
  --only-binary=:all: \
  -t {package_dir} \
  {req_string}
```

**What each flag does:**
- `--platform manylinux2014_x86_64` → fetch the Linux x86_64 wheel
- `--implementation cp` → CPython (what Lambda uses)
- `--python-version 312` → match Lambda's Python 3.12 runtime
- `--only-binary=:all:` → never compile from source (source builds also produce local OS binaries)

After rebuilding the zip and redeploying, the Lambda's zip contained `_pydantic_core.cpython-312-x86_64-linux-gnu.so` ✅

### What We Learned

- `pip install` always fetches binaries matching the **host OS** by default
- Cross-platform Lambda deployments always need the `--platform` flags for any package with C extensions (pydantic, numpy, cryptography, etc.)
- Checking the zip contents (`unzip -l`) is the fastest way to diagnose platform mismatches

---

## Challenge 2: Lambda Runtime Mismatch (Python 3.10 vs 3.12)

### What Was the Issue?

Even after the platform fix, the Lambda still failed intermittently, and the AWS console showed the runtime was Python 3.10 while local development was using Python 3.12 (Anaconda).

### What Was Happening?

The Lambda function was originally deployed with `python3.10` as the runtime. When the `pip install` was updated to target `--python-version 312`, it built wheels for Python 3.12. But the Lambda was still running Python 3.10 — the two are **binary-incompatible** for C extensions.

Additionally, the notebook's deploy step had a hardcoded `Runtime: "python3.10"` in the cell.

### How Was It Identified?

CloudWatch log's `INIT_START` entry showed `"Runtime: python3.10"`. The AI agent cross-referenced this with the `--python-version 312` in the deployment script and caught the mismatch.

### How Was It Solved?

Two changes were made:
1. Updated the Lambda runtime in the deployment cell from `"python3.10"` → `"python3.12"`
2. Updated the [deploy_lambda_function](./lambda_helpers.py#L163-L228) call to also push the runtime configuration change (not just the code zip), using `lambda_client.update_function_configuration(Runtime="python3.12")`

### What We Learned

- Updating Lambda code (the zip) and updating Lambda configuration (the runtime) are **two separate API calls**
- Always match `--python-version` in the pip flags to the actual Lambda `Runtime`

---

## Challenge 3: Gemini Embedding API Rate Limit (429 RESOURCE_EXHAUSTED)

### What Was the Issue?

After successfully indexing 759 medical document chunks into ChromaDB using Gemini embeddings (`gemini-embedding-001`), re-running Step 9b raised:

```
ClientError: 429 RESOURCE_EXHAUSTED
Quota exceeded for metric: embed_content_free_tier_requests, limit: 100
Please retry in 48s.
```

### What Was Happening?

The Gemini free tier allows only **100 embedding requests per minute**. Each batch was 50 texts = 2 requests per batch × 16 batches = ~32 requests for 759 chunks. However, the test search (Step 9c) also called `embed_content` for the query itself. With rapid re-runs (testing, debugging), the quota was exhausted quickly.

More fundamentally, using a **cloud API for embeddings** during local development is fragile — quotas reset on a schedule, and hitting them blocks all subsequent work.

### How Was It Identified?

The error code `429 RESOURCE_EXHAUSTED` and the specific metric name `embed_content_free_tier_requests` made it immediately clear. The AI agent explained that the free tier limit is 100 calls/minute and proposed switching to local embeddings.

### How Was It Solved?

Switched from Gemini API embeddings to **ChromaDB's built-in local sentence-transformers** (`all-MiniLM-L6-v2`), which runs entirely on-device with no API calls:

**[gemini_helpers.py](./gemini_helpers.py) — [embed_and_index_chunks](./gemini_helpers.py#L203-L288) (before):**
```python
embeddings = embed_texts_with_gemini(gemini_client, texts, "RETRIEVAL_DOCUMENT")
collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
```

**After:**
```python
# ChromaDB auto-embeds using sentence-transformers locally — no API calls
collection.add(ids=ids, documents=texts, metadatas=metadatas)
# Just omit the embeddings= argument — ChromaDB handles it automatically
```

**[gemini_helpers.py](./gemini_helpers.py) — [search_chroma](./gemini_helpers.py#L294-L362) (before):**
```python
embedding = embed_texts_with_gemini(gemini_client, [query], "RETRIEVAL_QUERY")
results = collection.query(query_embeddings=embedding, ...)
```

**After:**
```python
results = collection.query(query_texts=[query], ...)
# ChromaDB auto-embeds the query with the same model used for documents
```

### What We Learned

- For local/dev work, local embedding models are almost always better: free, fast, no rate limits, and offline
- ChromaDB's built-in embedding is a zero-config way to get sentence-transformers working
- The same model must be used for both indexing and querying — ChromaDB guarantees this automatically when using `documents=` / `query_texts=`

---

## Challenge 4: Agent Loop — 19 Pydantic Validation Errors

### What Was the Issue?

The final agent chat cell reliably failed on the first question with:

```
❌ Error: 19 validation errors for _GenerateContentParameters
contents.Content
  Input should be a valid dictionary or object to extract fields from
  [input_value=[{'role': 'user', 'parts': [{'text': 'what are the symptoms of cold?'}]}], input_type=list]
contents.str
  Input should be a valid string ...
```

19 errors for a single API call. The agent would immediately crash.

### What Was Happening?

The `conversation_history` was built as a **list of plain Python dicts**:

```python
conversation_history.append({
    "role": "user",
    "parts": [{"text": user_message}]     # plain string ← original bug
})
```

The `google-genai` SDK v1.66.0 uses **pydantic v2.9** for strict request validation. The `contents` parameter has a complex union type (`ContentListUnion`). Pydantic tries every possible union alternative and reports failures for all of them — hence 19 errors.

Two root causes were entangled:

1. **`"parts": [user_message]`** — original code put a raw string directly in `parts` instead of `{"text": user_message}`
2. **Stale `conversation_history`** — re-running the cell without a kernel restart accumulated entries from previous (failed) runs. Two consecutive `user` messages in a row — with no `model` response in between — is invalid for Gemini's multi-turn format, triggering the most cryptic form of this error.

### How Was It Identified?

The AI agent (Antigravity) ran a series of isolation tests directly from the command line:

```python
# Test 1: dict format, no config → WORKS
# Test 2: types.Content format → WORKS
# Test A: dict + full GENERATION_CONFIG with tools → WORKS
```

All tests passed in isolation, confirming the format itself wasn't the issue — something **in the notebook's runtime state** was. The key insight: `conversation_history = []` was only initialized when the setup cell first ran. Re-running the chat cell without restarting added a second user message to the existing list.

### How Was It Solved?

**Fix 1 — Use `types.Content` objects explicitly throughout:**

```python
# BEFORE: plain dicts (fragile — mixes Part objects with dict format)
conversation_history.append({"role": "user", "parts": [user_message]})

# AFTER: explicit types.Content objects (always valid)
conversation_history.append(
    genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)])
)
```

**Fix 2 — Reuse `candidate.content` directly for model turns:**

```python
# BEFORE: rebuild a dict from response_parts
conversation_history.append({"role": "model", "parts": response_parts})

# AFTER: the SDK gives us a types.Content object directly — use it
conversation_history.append(candidate.content)  # already types.Content ✅
```

**Fix 3 — Reset history at the start of every chat session:**

```python
# At the top of the chat cell:
conversation_history = []  # Reset so every run starts clean
```

### What We Learned

- **pydantic v2 union validation** reports ALL failures across ALL alternatives — 19 errors for 1 item means there are ~19 possible types in the union, all of which failed
- **Notebook state persistence** is a common trap: variables from a previous cell run survive across re-runs unless the kernel is restarted or the variable is explicitly reset
- Always use the SDK's typed objects (`types.Content`, `types.Part`) rather than plain dicts when working with complex pydantic-validated APIs
- `candidate.content` from a Gemini response is already a `types.Content` object — there's no need to rebuild it

---

## Challenge 5: VS Code Notebook Caching Overwriting File Changes

### What Was the Issue?

After the AI agent programmatically fixed the notebook file ([Lab-6-Gemini.ipynb](./Lab-6-Gemini.ipynb)) on disk using Python — replacing the dict-based `conversation_history` with `types.Content` objects — the notebook still ran the old broken code. Even after a kernel restart.

### What Was Happening?

VS Code keeps an **in-memory representation** of any open [.ipynb](./Lab-6.ipynb) file. When the file is modified externally (e.g., by a Python script), VS Code may either:
- Prompt "File changed on disk — reload?" (if the user dismisses this, the old version stays active)
- Or on save, write its in-memory version **back over** the disk file

This meant the programmatic fix written to disk was silently overwritten by VS Code's cached version every time the user saved or VS Code auto-saved.

### How Was It Identified?

The agent verified via `grep` that the correct `genai_types.Content` code was on disk:
```bash
grep -c "genai_types.Content" Lab-6-Gemini.ipynb  # returned 1 → fix IS on disk
```

But the notebook still ran old code. This ruled out a code issue and pointed to a VS Code caching problem.

### How Was It Solved?

The user **closed the notebook file** in VS Code and **reopened it**. This forced VS Code to reload from disk, picking up the corrected version. The fix then worked correctly on the next run.

### What We Learned

- Never rely on programmatic [.ipynb](./Lab-6.ipynb) file edits taking effect while the notebook is open in VS Code
- The reliable workflow: make the edit → close the file in VS Code → reopen it
- For Python helper files ([.py](./lambda_helpers.py)), programmatic edits work fine since VS Code doesn't cache their execution state the same way
- When a notebook fix "doesn't stick," always verify what's actually on disk vs. what VS Code is displaying

---

## Summary Table

| # | Challenge | Root Cause | Fix |
|---|-----------|------------|-----|
| 1 | Lambda `pydantic_core` import error | `pip` fetched macOS wheels instead of Linux | `--platform manylinux2014_x86_64 --only-binary=:all:` flags |
| 2 | Lambda Python version mismatch | Runtime was `python3.10`, code built for `3.12` | Updated Lambda runtime to `python3.12` |
| 3 | Gemini embedding 429 rate limit | Free tier: 100 calls/min, exceeded during development | Switched to local ChromaDB sentence-transformers |
| 4 | Agent loop 19 pydantic validation errors | Plain dicts + stale `conversation_history` | `types.Content` objects + `conversation_history = []` reset |
| 5 | VS Code notebook caching | VS Code in-memory state overwrote disk edits | Close and reopen notebook to force reload |

---

## How AI-Assisted Debugging Helped

Throughout these challenges, the AI agent (Antigravity) accelerated debugging in several ways:

- **Binary inspection**: `unzip -l ade_lambda.zip | grep pydantic_core` immediately revealed the macOS vs. Linux binary mismatch without needing to deploy and wait for CloudWatch logs
- **Isolation testing**: Running the exact `generate_content` call with different input formats from the terminal confirmed the SDK accepted both formats, pointing to notebook state as the true culprit
- **Systematic hypothesis testing**: Ruling out format issues via live API tests before investigating VS Code caching
- **Root cause explanation**: The pydantic union validation behavior ("19 errors = tries all union alternatives") was explained clearly before any code was changed
- **Inline documentation**: Each fix was accompanied by comments explaining the *why* — making the code self-documenting for future debugging

The general pattern that proved most effective: **reproduce the issue in the simplest possible environment first** (command line, not notebook), then narrow down what's different between that environment and the failing one.
