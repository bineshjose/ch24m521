# scripts/bench_api.py
import concurrent.futures, requests, time, random, statistics, json, pathlib
N=200; CONCURRENCY=10
def sample():
    # random but reasonable feature generator
    return {
        "Pclass": random.choice([1,2,3]),
        "Sex": random.choice(["male","female"]),
        "Age": max(0.4, random.gauss(30,12)),
        "SibSp": random.choice([0,1,2]),
        "Parch": random.choice([0,1,2]),
        "Fare": max(0.0, random.gauss(30,20)),
        "Embarked": random.choice(["S","C","Q"]),
    }

def one_call():
    p = sample()
    t0 = time.perf_counter()
    r = requests.post("http://localhost:9090/predict_titanic", json=p, timeout=5)
    dt = (time.perf_counter()-t0)*1000
    return dt, r.status_code

lat = []
with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
    for dt,code in ex.map(lambda _: one_call(), range(N)):
        if code==200: lat.append(dt)

lat.sort()
p50 = statistics.median(lat); p95 = lat[int(0.95*(len(lat)-1))]; p99 = lat[int(0.99*(len(lat)-1))]
out = {"N":N,"concurrency":CONCURRENCY,"p50_ms":p50,"p95_ms":p95,"p99_ms":p99,"mean_ms":statistics.mean(lat)}
path = pathlib.Path("reports/tables/api_bench.json"); path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2))
print("Results:", out)
