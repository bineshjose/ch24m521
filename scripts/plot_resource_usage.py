# scripts/plot_resource_usage.py
import pandas as pd, matplotlib.pyplot as plt, pathlib
p = pathlib.Path("reports/tables/docker_stats_training.csv")
df = pd.read_csv(p)
# normalize percentages
df["cpu_perc"] = df["cpu_perc"].str.rstrip("%").astype(float)
df["mem_perc"] = df["mem_perc"].str.rstrip("%").astype(float)
# basic line plots
out = pathlib.Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
for col in ["cpu_perc","mem_perc"]:
    ax = df[col].plot(title=f"Trainer {col} over time")
    ax.set_xlabel("sample"); ax.set_ylabel(col)
    ax.get_figure().tight_layout(); ax.get_figure().savefig(out/f"trainer_{col}.png"); ax.get_figure().clf()
print("Wrote trainer_cpu_perc.png, trainer_mem_perc.png")
