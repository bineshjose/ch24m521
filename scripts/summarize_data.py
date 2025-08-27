# scripts/summarize_data.py
import pandas as pd, json, pathlib
from datetime import datetime
parquet = pathlib.Path("data/processed/titanic.parquet")
raw_csv = pathlib.Path("data/titanic.csv")

out_tab = pathlib.Path("reports/tables"); out_fig = pathlib.Path("reports/figures")
out_tab.mkdir(parents=True, exist_ok=True); out_fig.mkdir(parents=True, exist_ok=True)

# Raw summary
raw = pd.read_csv(raw_csv)
raw_summary = {
    "n_rows": len(raw),
    "n_cols": raw.shape[1],
    "target_positive": int((raw["Survived"]==1).sum()),
    "target_negative": int((raw["Survived"]==0).sum()),
    "missing_counts": raw.isna().sum().to_dict(),
}
path1 = out_tab/"dataset_raw_summary.json"
path1.write_text(json.dumps(raw_summary, indent=2))

# Processed parquet (as used by Spark)
df = pd.read_parquet(parquet)
proc_summary = {"n_rows": len(df), "n_cols": df.shape[1], "columns": list(df.columns)}
(out_tab/"dataset_processed_summary.json").write_text(json.dumps(proc_summary, indent=2))

# Simple bar for class balance
ax = raw["Survived"].value_counts().sort_index().rename({0:"Died",1:"Survived"}).plot(kind="bar", title="Class Balance (Raw)")
fig = ax.get_figure(); fig.tight_layout(); fig.savefig(out_fig/"class_balance.png"); fig.clf()
print("Wrote dataset summaries and class_balance.png")
