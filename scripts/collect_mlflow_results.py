# scripts/collect_mlflow_results.py
import os, json, pathlib, mlflow
import pandas as pd

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5500"))
client = mlflow.tracking.MlflowClient()

EXP_NAME = "titanic_experiment"    # adjust if you used a different name
MODEL_NAME = "titanic_spark_model"

out_tab = pathlib.Path("reports/tables"); out_fig = pathlib.Path("reports/figures")
out_tab.mkdir(parents=True, exist_ok=True); out_fig.mkdir(parents=True, exist_ok=True)

# Find experiment
exp = client.get_experiment_by_name(EXP_NAME)
if exp is None:
    raise SystemExit(f"Experiment {EXP_NAME} not found")

# Pull all runs (completed) and basic fields
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)
# Expect metrics: auc, accuracy, precision, recall, f1 (if logged), plus params.*
runs = runs.sort_values(["metrics.auc"], ascending=False)

# Save a compact leaderboard
cols = [c for c in runs.columns if c.startswith("params.") or c.startswith("metrics.") or c in ["run_id","start_time","end_time","tags.mlflow.runName"]]
leader = runs[cols].copy()
leader.rename(columns={"tags.mlflow.runName":"run_name"}, inplace=True)
leader_path = out_tab/"leaderboard.csv"
leader.to_csv(leader_path, index=False)

# Find current Production model
versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
prod = [v for v in versions if getattr(v, "current_stage", "") == "Production"]
prod = prod[0] if prod else None

# Pull confusion matrix artifact if present for the best run and for Production
def download_confusion(run_id, dest):
    try:
        local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="confusion_matrix.csv")
        pd.read_csv(local_dir).to_csv(dest, index=False)
        return True
    except Exception:
        return False

best_run_id = leader.iloc[0]["run_id"] if len(leader) else None
if best_run_id:
    download_confusion(best_run_id, out_tab/"confusion_best.csv")
if prod is not None:
    download_confusion(prod.run_id, out_tab/"confusion_production.csv")

print(f"Wrote {leader_path}")
if best_run_id: print("Wrote confusion_best.csv (if logged)")
if prod is not None: print("Wrote confusion_production.csv (if logged)")
