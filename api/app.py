# api/app.py
import os, csv, pathlib, time
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Titanic MLOps API")

class Passenger(BaseModel):
    Pclass:int; Sex:str; Age:float; SibSp:int; Parch:int; Fare:float; Embarked:str

MODEL_NAME = "titanic_spark_model"
_model = None
_last_error = None

def _choose_model_uri(name: str) -> str:
    """Pick Production if present; else latest — filter client-side (no current_stage in server query)."""
    c = mlflow.tracking.MlflowClient()
    vers = c.search_model_versions(f"name='{name}'")
    if not vers:
        raise RuntimeError(f"Model '{name}' not found in registry.")
    prod = [v for v in vers if getattr(v, "current_stage", None) == "Production"]
    v = prod[0] if prod else sorted(vers, key=lambda x: int(x.version))[-1]
    return f"models:/{name}/{v.version}"

def _load_model(retries: int = 6, delay: float = 2.0):
    """Lazy-load with small retry so MLflow can come up; record last_error."""
    global _model, _last_error
    _model = None
    _last_error = None
    for _ in range(retries):
        try:
            uri = _choose_model_uri(MODEL_NAME)
            _model = mlflow.pyfunc.load_model(uri)
            return
        except Exception as e:
            _last_error = str(e)
            time.sleep(delay)
    raise RuntimeError(_last_error or "Unknown model load error")

def get_model():
    global _model
    if _model is None:
        _load_model()
    return _model

LOG = pathlib.Path("/app/data/infer_log_titanic.csv")
LOG.parent.mkdir(parents=True, exist_ok=True)

@app.get("/health")
def health():
    return {
        "status":"ok",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "model_loaded": _model is not None,
        "last_error": _last_error,
    }

@app.post("/reload")
def reload_model():
    try:
        _load_model()
        return {"reloaded": True, "error": None}
    except Exception as e:
        # return 200 with error message so you can see the cause quickly
        return {"reloaded": False, "error": str(e)}

@app.post("/predict_titanic")
def predict_titanic(p: Passenger):
    try:
        model = get_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")

    df = pd.DataFrame([p.dict()])
    y = model.predict(df)
    pred = int(pd.Series(y).iloc[0])

    with LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=df.columns)
        if LOG.stat().st_size == 0:
            w.writeheader()
        w.writerow(df.iloc[0].to_dict())
    return {"prediction": pred}
