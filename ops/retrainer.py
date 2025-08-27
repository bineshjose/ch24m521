import os, time, subprocess, sys, requests

INTERVAL=int(os.environ.get("RETRAIN_CHECK_INTERVAL_SEC","900"))
os.environ["PSI_THRESHOLD"]=os.environ.get("PSI_THRESHOLD","0.2")

def run(cmd, cwd=None):
    print(">", " ".join(cmd))
    return subprocess.call(cmd, cwd=cwd)

def api_health():
    try:
        r = requests.get("http://api:8000/health", timeout=5)
        print("API health:", r.json())
    except Exception as e:
        print("API health check failed:", e)

def main():
    print("Auto-retrainer started. Interval:", INTERVAL, "sec")
    while True:
        code = run([sys.executable, "training_titanic/drift_titanic.py"], cwd="/ops")
        if code==42:
            print("Drift detected -> retraining Titanic...")
            run([sys.executable, "training_titanic/train_titanic.py"], cwd="/ops")
            api_health()
        else:
            print("No drift or not enough data yet.")
        time.sleep(INTERVAL)

if __name__=="__main__":
    main()
