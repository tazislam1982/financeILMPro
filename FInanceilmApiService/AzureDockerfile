FROM python:3.11-bookworm

# System deps (curl for HEALTHCHECK, sqlite3 if you use it)
RUN apt-get update \
 && apt-get install -y --no-install-recommends sqlite3 curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Dependencies first (cache-friendly)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# App code
COPY . /app

# Startup script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
 && chown -R appuser:appuser /app /entrypoint.sh

USER appuser

# Runtime config
ENV PORT=8000 \
    WORKERS=4 \
    TIMEOUT=120 \
    APP_MODULE=app:app

EXPOSE 8000

# Exec-form ENTRYPOINT (no shell), JSON args = no hadolint warning
ENTRYPOINT ["/entrypoint.sh"]

# Healthcheck: prefer /healthz, fallback to /
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl --fail "http://127.0.0.1:$PORT/healthz" || curl --fail "http://127.0.0.1:$PORT/" || exit 1
