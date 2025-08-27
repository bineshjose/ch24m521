#!/usr/bin/env bash
set -euo pipefail
API="http://localhost:9090"

echo "==> /health"
curl -fsS "$API/health" || true
echo -e "\n"

echo "==> Warm model (reload)"
curl -fsS -X POST "$API/reload" || true
echo -e "\n"

echo "==> 3 prediction cases"
python - <<'PY'
import requests, json
API="http://localhost:9090/predict_titanic"
cases=[
 {"Pclass":3,"Sex":"male","Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":"S"},
 {"Pclass":1,"Sex":"female","Age":38,"SibSp":1,"Parch":0,"Fare":71.2833,"Embarked":"C"},
 {"Pclass":3,"Sex":"female","Age":26,"SibSp":0,"Parch":0,"Fare":7.925,"Embarked":"S"},
]
for i,p in enumerate(cases,1):
    r=requests.post(API,json=p,timeout=10)
    print(f"Case {i}: {r.status_code} -> {r.text}")
PY
