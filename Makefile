
COMPOSE = docker compose

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down -v

logs:
	$(COMPOSE) logs -f

urls:
	@echo "MLflow:  http://localhost:5500"
	@echo "API:     http://localhost:9090 (Swagger /docs)"
	@echo "MinIO:   http://localhost:8800 (Console)  |  S3 API: http://localhost:8801"
	@echo "Jupyter: http://localhost:8880"

train-titanic:
	$(COMPOSE) run --rm trainer-titanic

profile-spark:
	$(COMPOSE) run --rm trainer-titanic python profile_spark.py

drift-check:
	$(COMPOSE) run --rm trainer-titanic python drift_titanic.py || true

api-test:
	python scripts/test_api.py

retrainer-up:
	$(COMPOSE) up -d retrainer

retrainer-down:
	$(COMPOSE) stop retrainer
