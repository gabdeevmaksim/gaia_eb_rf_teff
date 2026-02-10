# Multi-stage Dockerfile for Gaia EB Teff Prediction
# Optimized for prediction workloads (~500MB)

# Stage 1: Builder - Install dependencies
FROM python:3.11-slim-bullseye AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy and install Python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Stage 2: Runtime - Minimal production image
FROM python:3.11-slim-bullseye

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-base \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY pipeline.py .
COPY scripts/download_datasets.py ./scripts/

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create required directories
RUN mkdir -p data/{raw,processed,cache} models/ reports/figures/

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/app/models \
    DATA_CACHE_DIR=/app/data/processed

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["--help"]
