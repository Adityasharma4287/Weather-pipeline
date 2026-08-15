# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: builder — install dependencies into a virtualenv so the final
# image doesn't carry build toolchains.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime — slim image, non-root user, only the venv + app code.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root user so the container never runs the API as root.
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AUDIT_LOG_PATH=/data/weather_pipeline_audit.log

COPY src ./src
COPY static ./static
COPY run_pipeline.py interactive_cli.py ./

# Writable volume mount point for the append-only audit log — keep it
# outside the read-only application code layer.
RUN mkdir -p /data && chown -R app:app /data /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=3).status == 200 else sys.exit(1)"

# --workers can be raised for production; kept at a modest default here.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
