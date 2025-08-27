# scripts/capture_env.py
import json, platform, subprocess, sys, pathlib
from datetime import datetime, timezone

def sh(args):
    try:
        return subprocess.check_output(args, text=True).strip()
    except Exception as e:
        return f"ERR: {e}"

def compose_exec(service, cmd):
    """
    Try `docker compose exec -T` (requires the service to be running).
    If that fails, fall back to `docker compose run --rm -T` to start a
    one-shot container just to query the version.
    """
    try:
        return sh(["bash","-lc", f"docker compose exec -T {service} {cmd}"])
    except Exception:
        return sh(["bash","-lc", f"docker compose run --rm -T {service} {cmd}"])

outdir = pathlib.Path("reports/tables"); outdir.mkdir(parents=True, exist_ok=True)

report = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "host_platform": platform.platform(),
    "host_python": sys.version.replace("\n"," "),
    "cpu": sh(["bash","-lc","lscpu | sed -n 's/Model name:\\s*//p' | head -n1"]),
    "mem_total_kb": sh(["bash","-lc","grep MemTotal /proc/meminfo | awk '{print $2}'"]),
    "docker": sh(["bash","-lc","docker --version"]),
    "docker_compose": sh(["bash","-lc","docker compose version"]),

    # Versions from containers (no need for host to have pyspark/mlflow)
    "java_in_trainer": compose_exec("trainer-titanic", "java -version 2>&1 | head -n1"),
    "spark_in_trainer": compose_exec("trainer-titanic",
        "python -c \"import pyspark; print(pyspark.__version__)\""),
    "spark_in_jupyter": compose_exec("jupyter-spark",
        "python -c \"import pyspark; print(pyspark.__version__)\""),
    "mlflow_in_server": compose_exec("mlflow",
        "python -c \"import mlflow, pkgutil; "
        "print(getattr(mlflow,'__version__', None) or "
        "__import__('pkg_resources').get_distribution('mlflow').version)\""),
}

(outdir/"environment.json").write_text(json.dumps(report, indent=2))
print(f"Wrote {outdir/'environment.json'}")
