# syntax=docker/dockerfile:1
# IAD platform — multi-stage production image.
#
# Targets:
#   streamlit  — UI (default)
#   api        — FastAPI backend
#
# Build examples:
#   docker build -f docker/Dockerfile --target streamlit -t iad-streamlit .
#   docker build -f docker/Dockerfile --target api -t iad-api .

# ---------------------------------------------------------------------------
# Builder — compile wheels / install package
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt pyproject.toml README.md ./
COPY iad ./iad
COPY src ./src
COPY app.py ./
COPY pages ./pages
COPY alembic.ini ./

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements-prod.txt \
    && pip install --no-deps .

# ---------------------------------------------------------------------------
# Runtime base — shared OS libraries for ML stacks
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_RUN_ON_SAVE=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin iad

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY iad ./iad
COPY src ./src
COPY app.py ./
COPY pages ./pages
COPY alembic.ini ./
COPY .streamlit ./.streamlit
COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh \
    && mkdir -p data models logs reports exports \
    && chown -R iad:iad /app

USER iad

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://127.0.0.1:${HEALTH_PORT:-8501}/_stcore/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
FROM runtime-base AS streamlit

ENV HEALTH_PORT=8501 \
    IAD_ENVIRONMENT=production \
    IAD_DEBUG=false

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.fileWatcherType=none"]

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
FROM runtime-base AS api

ENV HEALTH_PORT=8000 \
    IAD_ENVIRONMENT=production \
    IAD_DEBUG=false \
    IAD_AUTO_CREATE_DB=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "iad.backend.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
