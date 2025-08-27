#!/usr/bin/env bash
set -euo pipefail

# --- Config ---
API_URL="http://localhost:9090"
MLFLOW_UI="http://localhost:5500"
MINIO_CONSOLE="http://localhost:8800"
JUPYTER_URL="http://localhost:8880"

echo "==> 0) Ensure Docker Desktop is running (Windows tray)."

echo "==> 1) Start core services (DB, MinIO, MLflow)"
docker compose up -d db minio mlflow

echo "==> 2) Wait for MLflow to answer..."
for i in {1..60}; do
  if curl -fsS "$MLFLOW_UI" >/dev/null 2>&1; then echo "MLflow UI OK"; break; fi
  sleep 1
done

echo "==> 3) Start API, retrainer, and (optional) Jupyter"
docker compose up -d api retrainer jupyter-spark || true

echo "==> 4) Show status"
docker compose ps

echo "==> 5) Health check"
curl -fsS "$API_URL/health" || true
echo

echo "==> Tips:"
echo " MLflow UI:      $MLFLOW_UI"
echo " MinIO Console:  $MINIO_CONSOLE (login: minio / minio123)"
echo " Jupyter-Spark:  $JUPYTER_URL"
