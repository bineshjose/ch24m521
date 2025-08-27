# scripts/plot_hpo.py
import os, pathlib, mlflow, pandas as pd
import matplotlib.pyplot as plt

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5500"))
client = mlflow.tracking.MlflowClient()
EXP_NAME = "titanic_experiment"

out = pathlib.Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
exp = client.get_experiment_by_name(EXP_NAME); assert exp
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=1000)

# Example: AUC vs numTrees for RF
rf = runs[runs["tags.mlflow.runName"].str.contains("titanic-rf", na=False)]
if "params.numTrees" in rf and "metrics.auc" in rf:
    plt.figure()
    (rf.groupby("params.numTrees")["metrics.auc"].max().astype(float)
       .sort_index().plot(marker="o", title="RF: AUC vs numTrees"))
    plt.xlabel("numTrees"); plt.ylabel("AUC"); plt.tight_layout()
    plt.savefig(out/"hpo_rf_auc_numTrees.png")

# Example: AUC vs maxDepth for GBT
gbt = runs[runs["tags.mlflow.runName"].str.contains("titanic-gbt", na=False)]
if "params.maxDepth" in gbt and "metrics.auc" in gbt:
    plt.figure()
    (gbt.groupby("params.maxDepth")["metrics.auc"].max().astype(float)
        .sort_index().plot(marker="o", title="GBT: AUC vs maxDepth"))
    plt.xlabel("maxDepth"); plt.ylabel("AUC"); plt.tight_layout()
    plt.savefig(out/"hpo_gbt_auc_maxDepth.png")

# Example: AUC vs regParam for LR
lr = runs[runs["tags.mlflow.runName"].str.contains("titanic-lr", na=False)]
if "params.regParam" in lr and "metrics.auc" in lr:
    plt.figure()
    (lr.groupby("params.regParam")["metrics.auc"].max().astype(float)
        .sort_index().plot(marker="o", title="LR: AUC vs regParam"))
    plt.xlabel("regParam"); plt.ylabel("AUC"); plt.tight_layout()
    plt.savefig(out/"hpo_lr_auc_regParam.png")

print("Wrote HPO plots (if params available).")
