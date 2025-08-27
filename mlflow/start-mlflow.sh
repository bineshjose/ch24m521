#!/bin/sh
set -e
python /mlf/wait_for_db.py
exec mlflow server   --backend-store-uri "$MLFLOW_BACKEND_URI"   --default-artifact-root "$MLFLOW_ARTIFACT_URI"   --host 0.0.0.0 --port 5000
