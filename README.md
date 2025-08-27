# Titanic MLOps — Full E2E (Your Ports)

Ports:
- **MLflow UI** → http://localhost:5500
- **FastAPI** → http://localhost:9090  (Swagger: `/docs`)
- **Jupyter** → http://localhost:8880
- **MinIO Console** → http://localhost:8800
- **MinIO S3 API** → http://localhost:8801

Covers all tasks from the project brief: Spark preprocessing, DVC, distributed training (Spark MLlib),
HPO, MLflow Tracking & Model Registry (with auto-promotion), REST API with tests, drift detection (PSI)
and automated retraining, plus Spark resource profiling.

## Quick Start
1. Place Kaggle Titanic CSV at `./data/titanic.csv`.
2. `docker compose up -d --build`
3. `docker compose run --rm trainer-titanic`
4. Open MLflow: http://localhost:5500 → confirm `titanic_spark_model` is in **Production**.
5. API: http://localhost:9090/docs → try `/predict_titanic`.  
6. Drift monitor: `docker compose up -d retrainer`.

## DVC (optional but included)
From host/WSL:
```bash
pip install "dvc[s3]"
bash scripts/init_dvc.sh
dvc push
```

Notes:
- Inside Docker, services use `http://mlflow:5000` and `http://minio:9000`.
- Auto-promotion can be turned off in `.env` (`AUTO_PROMOTE_TITANIC=false`).
