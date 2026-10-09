# Stage 1: Build dependencies
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies if required (e.g. for building wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
# This standalone image serves CPU retrieval/reranking. HF ZeroGPU uses its
# Gradio environment separately. Select the CPU torch wheel explicitly so
# resolving sentence-transformers does not pull multi-GB CUDA dependencies.
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels \
        --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels \
        /app/wheels/torch-*.whl -r requirements-prod.txt

# Stage 2: Final runtime image
FROM python:3.12-slim

WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements-prod.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements-prod.txt

# Copy only runtime source. Never bake local indexes, documents, memory,
# notebooks, generated archives, or environment files into a public image.
COPY app.py agent.py access_control.py config.py gemini_helpers.py \
     supabase_backend.py private_sessions.py \
     hybrid_search.py llm_router.py live_guardrail.py query_optimizer.py \
     upload_handler.py visual_grounding_helper.py ./
COPY utils/logger.py ./utils/logger.py

# Expose Gradio default port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py", "--port", "7860" ]
