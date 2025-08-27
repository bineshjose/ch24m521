# scripts/test_api.py (enhanced)
import requests, time, statistics, json, pathlib
cases = [
  {"Pclass":3,"Sex":"male","Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":"S"},
  {"Pclass":1,"Sex":"female","Age":38,"SibSp":1,"Parch":0,"Fare":71.2833,"Embarked":"C"},
  {"Pclass":3,"Sex":"female","Age":26,"SibSp":0,"Parch":0,"Fare":7.925,"Embarked":"S"},
]
lat = []
for i,p in enumerate(cases,1):
    t0 = time.perf_counter()
    r = requests.post("http://localhost:9090/predict_titanic", json=p, timeout=10)
    dt = (time.perf_counter()-t0)*1000
    lat.append(dt)
    print(f"Case {i}: {r.status_code} → {r.json()}  ({dt:.1f} ms)")
print("latency_ms:", lat, "p50:", statistics.median(lat), "p95:", sorted(lat)[int(0.95*(len(lat)-1))])
# Save for the report
path = pathlib.Path("reports/tables/api_smoke.json"); path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"cases":cases, "latency_ms":lat}, indent=2))
print(f"Wrote {path}")
