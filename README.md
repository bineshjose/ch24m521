# Building a Robust MLOps Pipeline  
_End-to-end Spark + MLflow + MinIO + FastAPI on WSL/Docker_

**Author : Dr Binesh Jose , IIT Madras**

> This repository contains a **complete, reproducible MLOps pipeline** for the classic **Titanic survival** prediction task. It demonstrates distributed preprocessing and training in **Apache Spark**, experiment tracking and **Model Registry** with **MLflow**, S3-style artifact storage on **MinIO**, **automated promotion** of the best model, **drift checks**, and a production-style **FastAPI** service for inference.  
> Everything is containerized and orchestrated with **Docker Compose** on **WSL2** (Windows 10/11).

---

## Components

- **Data pipeline**: fault-tolerant preprocessing + feature engineering in Spark.  
- **Training**: LR / RF / GBT with small **HPO** grids; logs metrics, artifacts, and versions to MLflow.  
- **Model management**: auto-register & auto-promote best AUC champion to **Production** in the **MLflow Model Registry**.  
- **Serving**: FastAPI microservice that loads the Production model at startup and exposes `/predict_titanic`.  
- **Storage**: PostgreSQL (MLflow backend) and **MinIO** (S3 artifacts).  
- **Ops niceties**: `run_after_reboot.sh`, `smoke.sh`, data/version utilities, drift baseline & monitoring, exportable figures/tables for reports.

**Default ports (chosen for this project):**

| Service | URL | Notes |
|---|---|---|
| **API** (FastAPI) | http://localhost:9090 | `/health`, `/reload`, `/predict_titanic` |
| **MLflow UI** | http://localhost:5500 | Experiments, Models, Artifacts |
| **MinIO Console** | http://localhost:8800 | S3 browser |
| **MinIO S3 API** | http://localhost:8801 | Endpoint for `aws s3 …` / `mc …` |
| **Jupyter (Spark)** | http://localhost:8880 | Optional exploration |

---

## 1) Repository layout

```
ch24m521/
├─ api/                          # FastAPI service (serving)
│  ├─ app.py
│  └─ requirements.txt
├─ mlflow/                       # MLflow server image
│  ├─ Dockerfile
│  └─ start-mlflow.sh
├─ training_titanic/             # Spark training code & env
│  ├─ train_titanic.py
│  ├─ preprocess_spark.py
│  ├─ drift_titanic.py
│  └─ environment.yml            # micromamba base env for trainer
├─ data/                         # data under DVC (Git stores pointer)
│  └─ titanic.csv                # tracked via DVC, not Git LFS
├─ scripts/                      # utilities (see Section 9)
│  ├─ test_api.py
│  ├─ smoke.sh
│  ├─ run_after_reboot.sh
│  ├─ capture_env.py
│  ├─ summarize_data.py
│  ├─ collect_mlflow_results.py
│  └─ fetch_minio.py
├─ reports/                      # generated tables & figures
│  ├─ tables/
│  └─ figures/
├─ docker-compose.yml
├─ .env.example                  # copy to .env and edit
├─ .gitattributes                # enforce LF endings for sh/yml/py
└─ README.md
```

---

## 2) Prerequisites

- **Windows 10/11** with **WSL2** and Ubuntu installed.
- **Docker Desktop** (enable WSL integration for your Ubuntu distro).
- **Git** (on Windows and/or WSL).  
- Optional: **conda/mamba** on host (not required; everything runs in containers).
- Open ports: **9090, 5500, 8800, 8801, 8880**.

> _Project path examples_: `F:\mlops\ch24m521` (Windows) == `/mnt/f/mlops/ch24m521` (WSL).

---

## 3) First-time setup

### 3.1 Clone and fix line endings (important on Windows)
This project uses shell scripts. Ensure **LF** endings to avoid `bash\r` errors:

```bash
# Inside WSL, in the repo root:
git config core.autocrlf false
sudo apt-get update && sudo apt-get install -y dos2unix
find . -type f \( -name "*.sh" -o -name "*.yml" -o -name "*.yaml" -o -name "*.py" \) -print0 | xargs -0 dos2unix
chmod +x scripts/*.sh mlflow/start-mlflow.sh
```

### 3.2 Create your `.env`
Copy the template and edit as needed:

```bash
cp .env.example .env
```

Key entries (defaults shown):

```ini
API_PORT=9090
MLFLOW_PORT=5500
MINIO_CONSOLE_PORT=8800
MINIO_API_PORT=8801
JUPYTER_PORT=8880

MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=minio123
MLFLOW_S3_BUCKET=mlflow

MLFLOW_BACKEND_URI=postgresql+psycopg2://mlflow:mlflow@db:5432/mlflow
MLFLOW_ARTIFACT_URI=s3://mlflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

POSTGRES_DB=mlflow
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow

AUTO_PROMOTE_TITANIC=true
PSI_THRESHOLD=0.2

AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123
AWS_DEFAULT_REGION=us-east-1
```

> **Security**: change the MinIO / Postgres credentials if you expose these services.

### 3.3 Data versioning with DVC (one-time)

```bash
# init DVC (if not yet)
dvc init
git add .dvc .dvcignore
git commit -m "init DVC"

# stop tracking CSV with Git, track with DVC
git rm -r --cached data/titanic.csv
dvc add data/titanic.csv
git add data/titanic.csv.dvc data/.gitignore
git commit -m "Track Titanic CSV via DVC"
```

> If you use a remote (S3/Azure/GS), add it via `dvc remote add` and `dvc push`.

---

## 4) Build & start the stack

### 4.1 Build images

```bash
docker compose build
```

This creates images for: `mlflow`, `trainer-titanic`, `api`, and pulls `jupyter/pyspark-notebook` if you use Jupyter.

### 4.2 Start core infrastructure

```bash
docker compose up -d db minio mlflow
```

- **MLflow UI** → http://localhost:5500  
- **MinIO Console** → http://localhost:8800 (login: minio/minio123)  
- **MinIO S3 API** → http://localhost:8801 (use with `aws` / `mc`)

---

## 5) Train & evaluate

### 5.1 One command training

```bash
docker compose run --rm trainer-titanic
```

What it does:
- Runs the Spark preprocessing pipeline.
- Trains **Logistic Regression**, **Random Forest**, **Gradient-Boosted Trees** with small HPO grids.
- Logs **parameters**, **metrics** (AUC, confusion matrices, timings), and **artifacts** (feature importances, plots) to MLflow.
- **Registers** each best model to MLflow **Model Registry** and (optionally) **promotes** the champion to **Production** (if `AUTO_PROMOTE_TITANIC=true`).

Inspect results in **MLflow UI** → Experiment **titanic-spark**.

### 5.2 Optional: generate publication-ready outputs

```bash
# Environment snapshot (host + images)
python scripts/capture_env.py

# Dataset overview + missingness tables
python scripts/summarize_data.py

# Leaderboard & tables from MLflow
docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000   -w /app trainer-titanic python scripts/collect_mlflow_results.py
```

Outputs land in `reports/tables/` & `reports/figures/`.

---

## 6) Serve the model (FastAPI)

### 6.1 Start the API

```bash
docker compose up -d api
```

The service resolves the **Production** version of the model from MLflow at startup.

### 6.2 Health, reload & prediction

```bash
# health
curl -s http://localhost:9090/health
# => {"status":"ok","mlflow_uri":"http://mlflow:5000","model_loaded":true,...}

# reload the newest Production model (after a new promotion)
curl -X POST http://localhost:9090/reload

# predict
curl -s -X POST http://localhost:9090/predict_titanic   -H "Content-Type: application/json"   -d '{"Pclass":3,"Sex":"male","Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":"S"}'
```

Or run the test script:

```bash
python scripts/test_api.py
```

---

## 7) Artifacts on MinIO (S3)

### 7.1 Browse
Open **http://localhost:8800** and explore the bucket **mlflow**.

### 7.2 Download via AWS CLI

```bash
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
export AWS_DEFAULT_REGION=us-east-1

# Example: pull a confusion matrix from a run
aws --endpoint-url http://localhost:8801 s3 cp   s3://mlflow/<exp_id>/<run_id>/artifacts/eval/confusion_matrix.csv   reports/tables/confusion_matrix.csv
```

### 7.3 Download via MinIO Client (`mc`)

```bash
mc alias set localminio http://localhost:8801 minio minio123
mc cp localminio/mlflow/<exp_id>/<run_id>/artifacts/eval/confusion_matrix.csv       reports/tables/confusion_matrix.csv
```

---

## 8) Drift detection & automated retraining (optional but included)

### 8.1 Create a PSI baseline

```bash
docker compose run --rm -w /app trainer-titanic python drift_titanic.py
# exit 0 → baseline created / no drift; exit 42 → drift detected
```

The baseline file is stored under `data/` and used to compare incoming inference distributions.

### 8.2 Scheduled retraining (sample)

The `retrainer` service (if enabled in `docker-compose.yml`) periodically:
- Computes PSI vs. baseline.
- If `PSI_THRESHOLD` is exceeded, triggers a fresh training run.
- Optionally promotes the new model to **Production**.

Wire the same logic to your scheduler (cron, GitHub Actions, etc.) for production.

---

## 9) Utilities & scripts (don’t skip!)

> These scripts are what make this repo turnkey for new users.

- **`scripts/run_after_reboot.sh`**  
  Bring the environment up **after a restart**:
  ```bash
  ./scripts/run_after_reboot.sh
  ```
  It starts core services and the API, then pings `/reload` so the service picks up the latest **Production** model.  
  **If you ever see** `/usr/bin/env: 'bash\r': No such file or directory`, convert endings:
  ```bash
  dos2unix scripts/run_after_reboot.sh && chmod +x scripts/run_after_reboot.sh
  ```

- **`scripts/smoke.sh`**  
  One-shot **smoke test** that:
  1) checks **MLflow** and **API** health, 2) executes a **sample prediction**, and 3) prints a pass/fail summary.  
  Run it anytime you want to verify the deployment:
  ```bash
  ./scripts/smoke.sh
  ```

- **`scripts/test_api.py`**  
  Minimal client for `/predict_titanic` (great for CI).

- **`scripts/capture_env.py`**  
  Writes `reports/tables/environment.json/csv` with host + container versions for reproducibility.

- **`scripts/summarize_data.py`**  
  Parses the raw CSV / Parquet and writes dataset summary and missingness tables to `reports/tables/`.

- **`scripts/collect_mlflow_results.py`**  
  Queries MLflow for top runs (AUC) and generates a leaderboard table.  
  Recommended to run inside the trainer container:
  ```bash
  docker compose run --rm -e MLFLOW_TRACKING_URI=http://mlflow:5000     -w /app trainer-titanic python scripts/collect_mlflow_results.py
  ```

- **`scripts/fetch_minio.py`**  
  Python + boto3 example to fetch artifacts from MinIO via S3 API.

---

## 10) Everyday workflows

### 10.1 Fresh clone / new machine

```bash
# 1) Fix line endings, set executable bits
dos2unix $(git ls-files '*.sh' '*.yml' '*.yaml' '*.py') || true
chmod +x scripts/*.sh mlflow/start-mlflow.sh

# 2) Configure .env
cp .env.example .env && edit .env

# 3) Build & start infra
docker compose build
docker compose up -d db minio mlflow

# 4) Train and promote
docker compose run --rm trainer-titanic

# 5) Serve
docker compose up -d api
./scripts/smoke.sh
```

### 10.2 After reboot
```bash
./scripts/run_after_reboot.sh
# If API reports model_loaded=false, do:
curl -X POST http://localhost:9090/reload
```

### 10.3 Clean up
```bash
docker compose down           # stop
docker compose down -v        # stop + remove volumes (DB + artifacts!)
```

---

## 11) Troubleshooting (field-tested)

**Port already in use**  
Use `netstat` (Windows) or `ss` (WSL) and free the port, or change it in `.env` and `docker compose up -d`.

**API restarts with MLflow “current_stage invalid”**  
Use the shipped API code (we already avoid filtering by `current_stage` in the search query for older servers). If you modified it, revert to the version that enumerates versions and selects Production safely.

**MLflow “Loading artifact failed: INTERNAL_ERROR”**  
The MLflow image must include `boto3` for S3/MinIO. Rebuild:
```bash
docker compose build mlflow && docker compose up -d mlflow
```

**Jupyter asks for token**  
Get it from logs:
```bash
docker compose logs --tail 200 jupyter-spark | grep -i token
```
Then set a password on the landing page.

**Running Python scripts on host complains about PySpark/pyarrow**  
Prefer running analysis scripts **inside the trainer container** (ensures versions match). If you must run on the host, install:  
`pip install pyspark==3.4.1 pyarrow fastparquet mlflow boto3`.

**`/usr/bin/env 'bash\r'` or scripts won’t run**  
Convert to LF and make executable:
```bash
dos2unix scripts/*.sh mlflow/start-mlflow.sh
chmod +x scripts/*.sh mlflow/start-mlflow.sh
```

**API says `model_loaded:false`**  
It started before RL promotion completed. Call:
```bash
curl -X POST http://localhost:9090/reload
```
Ensure the registry has **at least one Production** version.

---

## 12) Reproducibility & data governance

- **DVC** keeps raw data out of Git; only `.dvc` pointers are committed.  
- **MLflow** logs params/metrics/artifacts; **MinIO** stores models and evaluation assets.  
- **Environment capture** (scripts/capture_env.py) records versions to support like-for-like reruns.

---

## 13) License & attribution

- Code: IITM License terms and conditions for distribution  
- Data: Titanic dataset © Kaggle / public domain sources; ensure you comply with their terms when distributing.

---

## 14) Credits

This project was engineered to be **teachable, reproducible, and production-minded**. It showcases strong defaults and pragmatic trade-offs so that anyone cloning the repo can build, run, and extend a robust MLOps pipeline with minimal friction.

> If you hit any snag running this on a fresh machine, open an issue with the exact command + output and we’ll keep the README and scripts bullet-proof.
