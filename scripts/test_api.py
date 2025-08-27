import requests

print("Testing Titanic endpoint...")
r = requests.post("http://localhost:9090/predict_titanic", json={
    "Pclass":3,"Sex":"male","Age":22,"SibSp":1,"Parch":0,"Fare":7.25,"Embarked":"S"
}, timeout=10)
print("Titanic:", r.status_code, r.text)

print("Health check...")
print(requests.get("http://localhost:9090/health", timeout=5).text)
