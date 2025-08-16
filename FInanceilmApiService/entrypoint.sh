#!/usr/bin/env sh
set -e

# Defaults (can be overridden with env vars)
: "${PORT:=8000}"
: "${WORKERS:=4}"
: "${TIMEOUT:=120}"
: "${APP_MODULE:=app:app}"

# Hand off to gunicorn (exec = proper signal handling)
exec gunicorn -k uvicorn.workers.UvicornWorker \
  -w "$WORKERS" \
  --bind "0.0.0.0:$PORT" \
  --timeout "$TIMEOUT" \
  "$APP_MODULE"
