# LexDPR Embedding API Server
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.4 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install Python 3.12 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    curl \
    git \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml poetry.lock* ./

RUN rm -f poetry.lock
RUN poetry lock

# Install dependencies (without dev dependencies)
RUN poetry install --no-root --no-dev --no-interaction

# Copy application code
COPY . .

# Install the project itself
RUN poetry install --only-root --no-interaction

# Default environment variables
ENV MODEL_PATH="checkpoint/lexdpr/bi_encoder" \
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
