# mlflow/Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# MLflow + S3 (boto3) + Postgres driver
RUN pip install --no-cache-dir \
      mlflow==3.3.1 \
      boto3==1.34.145 \
      psycopg2-binary==2.9.9

# Helpful default (can be overridden in compose)
ENV AWS_DEFAULT_REGION=us-east-1

# Keep your existing envs used by start-mlflow.sh
ENV MLFLOW_BACKEND_URI="" \
    MLFLOW_ARTIFACT_URI="" \
    MLFLOW_DB_HOST=db \
    MLFLOW_DB_PORT=5432

WORKDIR /mlf
COPY wait_for_db.py .
COPY start-mlflow.sh /usr/local/bin/start-mlflow.sh
RUN chmod +x /usr/local/bin/start-mlflow.sh

EXPOSE 5000
CMD ["/usr/local/bin/start-mlflow.sh"]
