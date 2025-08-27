#!/usr/bin/env bash
set -euo pipefail
# Initialize DVC and set MinIO S3 remote (host ports).
dvc init -f
dvc remote add -d s3 s3://mlflow
dvc remote modify s3 endpointurl http://localhost:8801
dvc remote modify s3 access_key_id "${MINIO_ROOT_USER:-minio}"
dvc remote modify s3 secret_access_key "${MINIO_ROOT_PASSWORD:-minio123}"
# Track Titanic csv
if [ -f data/titanic.csv ]; then
  dvc add data/titanic.csv
  echo "Added data/titanic.csv to DVC. Use 'dvc push' to upload to MinIO."
else
  echo "WARNING: data/titanic.csv not found yet; add it and run 'dvc add data/titanic.csv'."
fi
# Create preprocess stage
cat > dvc.yaml <<'YAML'
stages:
  preprocess_titanic:
    cmd: docker compose run --rm trainer-titanic python preprocess_spark.py
    deps:
      - data/titanic.csv
      - training_titanic/preprocess_spark.py
    outs:
      - data/processed/titanic.parquet
YAML
echo "DVC initialized. Remote set to MinIO http://localhost:8801"
