import os, sys, json, math, pathlib
import numpy as np
import pandas as pd

# -------- Settings --------
DATA_DIR = pathlib.Path("data")
BASELINE_PATH = DATA_DIR / "drift_baseline_titanic.json"
INFER_LOG = DATA_DIR / "infer_log_titanic.csv"

PSI_THRESHOLD = float(os.environ.get("PSI_THRESHOLD", "0.2"))
MIN_INFER_ROWS = int(os.environ.get("MIN_INFER_ROWS", "50"))

NUMERIC_FEATS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEG_FEATS  = ["Sex", "Embarked"]
OTHER = "__OTHER__"

# -------- Helpers --------
def _safe_hist_probs(counts, eps=1e-6):
    probs = (counts + eps).astype(float)
    probs = probs / probs.sum()
    return probs

def psi(expected, actual):
    """Population Stability Index between two probability arrays (same length)."""
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual,   dtype=float)
    return float(np.sum((actual - expected) * np.log(actual / expected)))

def make_numeric_bins(s: pd.Series, n_bins=10):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = list(np.unique(np.nanquantile(s.dropna(), qs)))
    if len(edges) < 3:
        # fallback to min/median/max style
        mn, mx = float(s.min()), float(s.max())
        edges = [mn, (mn + mx) / 2.0, mx]
    edges[0]  = -float("inf")
    edges[-1] =  float("inf")
    return edges

def baseline_from_training(csv_path="data/titanic.csv"):
    df = pd.read_csv(csv_path)
    base = {"numeric": {}, "categorical": {}, "n": int(len(df))}
    # numeric
    for f in NUMERIC_FEATS:
        edges = make_numeric_bins(df[f])
        counts, _ = np.histogram(df[f], bins=edges)
        probs = _safe_hist_probs(counts)
        base["numeric"][f] = {"bins": edges, "probs": probs.tolist()}
    # categorical
    for f in CATEG_FEATS:
        cats = list(df[f].astype("category").cat.remove_unused_categories().cat.categories)
        if OTHER not in cats:
            cats.append(OTHER)
        counts = pd.Series(0, index=cats, dtype=float)
        obs = df[f].astype(str).value_counts()
        counts.loc[obs.index.intersection(counts.index)] = obs
        probs = _safe_hist_probs(counts.values)
        base["categorical"][f] = {"cats": cats, "probs": probs.tolist()}
    return base

def ensure_baseline():
    if BASELINE_PATH.exists():
        with open(BASELINE_PATH, "r") as f:
            return json.load(f), False
    base = baseline_from_training()
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(base, f)
    print(f"[drift] Baseline created at {BASELINE_PATH}. (First run) No drift check performed.")
    return base, True

def actual_from_infer_log(base):
    if not INFER_LOG.exists():
        raise FileNotFoundError(f"Inference log not found: {INFER_LOG}")
    df = pd.read_csv(INFER_LOG)
    if len(df) < MIN_INFER_ROWS:
        print(f"[drift] Not enough inference rows ({len(df)}<{MIN_INFER_ROWS}). No drift.")
        return None
    act = {"numeric": {}, "categorical": {}, "n": int(len(df))}
    # numeric
    for f, meta in base["numeric"].items():
        edges = meta["bins"]
        counts, _ = np.histogram(df[f], bins=edges)
        probs = _safe_hist_probs(counts)
        act["numeric"][f] = {"probs": probs.tolist()}
    # categorical
    for f, meta in base["categorical"].items():
        cats = list(meta["cats"])
        # include OTHER in both baselines
        if OTHER not in cats:
            cats.append(OTHER)
        counts = pd.Series(0, index=cats, dtype=float)
        obs = df[f].astype(str)
        obs = obs.where(obs.isin(cats), OTHER)
        vc = obs.value_counts()
        counts.loc[vc.index] = vc.values
        probs = _safe_hist_probs(counts.values)
        act["categorical"][f] = {"probs": probs.tolist()}
    return act

# -------- Main --------
def main():
    base, just_created = ensure_baseline()
    if just_created:
        # First-time baseline creation => exit success, no drift
        sys.exit(0)

    act = actual_from_infer_log(base)
    if act is None:
        sys.exit(0)

    psi_report = {}
    max_psi = 0.0

    for f, meta in base["numeric"].items():
        p = np.array(meta["probs"])
        q = np.array(act["numeric"][f]["probs"])
        v = psi(p, q)
        psi_report[f] = v
        max_psi = max(max_psi, v)

    for f, meta in base["categorical"].items():
        p = np.array(meta["probs"])
        q = np.array(act["categorical"][f]["probs"])
        v = psi(p, q)
        psi_report[f] = v
        max_psi = max(max_psi, v)

    print("[drift] PSI by feature:")
    for k, v in psi_report.items():
        print(f"  - {k:10s}: {v:.4f}")

    if max_psi >= PSI_THRESHOLD:
        print(f"[drift] DRIFT_DETECTED (max PSI={max_psi:.4f} >= {PSI_THRESHOLD}).")
        # Exit code 42 signals the retrainer to kick in
        sys.exit(42)
    else:
        print(f"[drift] No material drift (max PSI={max_psi:.4f} < {PSI_THRESHOLD}).")
        sys.exit(0)

if __name__ == "__main__":
    main()
