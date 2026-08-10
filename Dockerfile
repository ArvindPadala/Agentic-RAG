# Stage 1: Build dependencies
FROM python:3.12-slim as builder

WORKDIR /app

# Install system dependencies if required (e.g. for building wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements-prod.txt

# Stage 2: Final runtime image
FROM python:3.12-slim

WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements-prod.txt .
RUN pip install --no-cache /wheels/*

# Copy application code
COPY . .

# Expose Gradio default port
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py", "--port", "7860", "--share"]
