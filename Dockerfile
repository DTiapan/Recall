# Multi-stage security-hardened Dockerfile for Recall
# Stage 1: Build virtual environment and wheels
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache .

# Stage 2: Minimal runtime image
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    RAG_ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app

# Create unprivileged service user
RUN groupadd -g 10001 recall && \
    useradd -u 10001 -g recall -s /bin/bash -m recall && \
    mkdir -p /app/data /app/storage && \
    chown -R recall:recall /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=recall:recall config.yaml ./
COPY --chown=recall:recall src/ ./src/

USER recall

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

ENTRYPOINT ["python", "-m", "recall.cli", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
