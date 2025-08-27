import os, time, sys
import psycopg2
host = os.environ.get("MLFLOW_DB_HOST","db")
port = int(os.environ.get("MLFLOW_DB_PORT","5432"))
user = os.environ.get("POSTGRES_USER","mlflow")
pwd = os.environ.get("POSTGRES_PASSWORD","mlflowpass")
db  = os.environ.get("POSTGRES_DB","mlflow")
for i in range(60):
    try:
        psycopg2.connect(host=host, port=port, user=user, password=pwd, dbname=db).close()
        sys.exit(0)
    except Exception as e:
        print("DB not ready:", e)
        time.sleep(2)
print("DB wait timeout"); sys.exit(1)
