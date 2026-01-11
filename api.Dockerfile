# LexDPR Embedding API Server - Optimized Build
# Target size: ~6-7GB (down from 11GB+)

# ============================================================
# Stage 1: Builder - Install dependencies
# ============================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Create virtualenv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Install API dependencies only
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# ============================================================
# Stage 2: Runtime - Minimal image
# ============================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Python 3.12 runtime only (no dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    curl \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy only necessary application code (model weights are mounted externally)
COPY lex_dpr/__init__.py ./lex_dpr/
COPY lex_dpr/models/*.py ./lex_dpr/models/
COPY entrypoint_api.py .

# Default environment variables
ENV MODEL_PATH="/model" \
    TEMPLATE="bge" \
    HOST="0.0.0.0" \
    PORT="8009" \
    DEVICE="" \
    PEFT_ADAPTER=""

EXPOSE 8009

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the API server
CMD python entrypoint_api.py \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    ${DEVICE:+--device "${DEVICE}"} \
    ${PEFT_ADAPTER:+--peft-adapter "${PEFT_ADAPTER}"}