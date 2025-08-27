import os
import time
import tempfile
from urllib.parse import urlparse

import psutil
import mlflow
import mlflow.spark
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, Imputer
)
from pyspark.ml.classification import (
    RandomForestClassifier, GBTClassifier, LogisticRegression
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


# -----------------------------
# MLflow configuration
# -----------------------------
AUTO_PROMOTE = os.environ.get("AUTO_PROMOTE_TITANIC", "true").lower() == "true"
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment("titanic-spark")


# -----------------------------
# SparkSession with S3/MinIO support
# -----------------------------
def build_spark(app_name: str) -> SparkSession:
    s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    ep = urlparse(s3_endpoint)
    endpoint_hostport = f"{ep.hostname}:{ep.port or (443 if ep.scheme == 'https' else 80)}"
    ssl_enabled = "true" if ep.scheme == "https" else "false"

    aws_key = os.environ.get("AWS_ACCESS_KEY_ID", "minio")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")

    spark_packages = os.environ.get(
        "SPARK_PACKAGES",
        "org.apache.hadoop:hadoop-aws:3.3.2,com.amazonaws:aws-java-sdk-bundle:1.12.262",
    )

    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.jars.packages", spark_packages)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3.impl",   "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.s3a.impl", "org.apache.hadoop.fs.s3a.S3A")
        .config("spark.hadoop.fs.s3a.endpoint", endpoint_hostport)
        .config("spark.hadoop.fs.s3a.access.key", aws_key)
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", ssl_enabled)
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .getOrCreate()
    )


spark = build_spark("titanic-train")

# -----------------------------
# Load data (prefer processed parquet if present)
# -----------------------------
if os.path.exists("data/processed/titanic.parquet"):
    df = spark.read.parquet("data/processed/titanic.parquet")
else:
    df = spark.read.csv("data/titanic.csv", header=True, inferSchema=True)

# -----------------------------
# Pipeline builders
# -----------------------------
def build_pipeline(algo: str = "rf"):
    im = Imputer(strategy="median", inputCols=["Age", "Fare"], outputCols=["Age", "Fare"])
    sx = StringIndexer(inputCol="Sex", outputCol="SexIdx", handleInvalid="keep")
    em = StringIndexer(inputCol="Embarked", outputCol="EmbarkedIdx", handleInvalid="keep")
    ohe = OneHotEncoder(inputCols=["SexIdx", "EmbarkedIdx"], outputCols=["SexOH", "EmbarkedOH"])
    va = VectorAssembler(
        inputCols=["Pclass", "Age", "SibSp", "Parch", "Fare", "SexOH", "EmbarkedOH"],
        outputCol="features",
    )

    if algo == "rf":
        clf = RandomForestClassifier(labelCol="Survived", featuresCol="features", seed=42)
    elif algo == "gbt":
        clf = GBTClassifier(labelCol="Survived", featuresCol="features", seed=42)
    else:
        clf = LogisticRegression(labelCol="Survived", featuresCol="features", maxIter=100)

    pipe = Pipeline(stages=[im, sx, em, ohe, va, clf])
    return pipe, clf


def hpo_and_log(algo: str = "rf"):
    train, val = df.randomSplit([0.8, 0.2], seed=42)
    pipe, clf = build_pipeline(algo)

    # Grid from the classifier object (do NOT index pipe.stages)
    if algo == "rf":
        grid = (ParamGridBuilder()
                .addGrid(clf.getParam("numTrees"), [100, 200, 400])
                .addGrid(clf.getParam("maxDepth"), [5, 7, 10])
                .build())
    elif algo == "gbt":
        grid = (ParamGridBuilder()
                .addGrid(clf.getParam("maxDepth"), [3, 5, 7])
                .addGrid(clf.getParam("maxIter"), [50, 100])
                .build())
    else:  # lr
        grid = (ParamGridBuilder()
                .addGrid(clf.getParam("regParam"), [0.0, 0.01, 0.1])
                .addGrid(clf.getParam("elasticNetParam"), [0.0, 0.5, 1.0])
                .build())

    evaluator = BinaryClassificationEvaluator(labelCol="Survived", metricName="areaUnderROC")
    tvs = TrainValidationSplit(
        estimator=pipe,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        trainRatio=0.8,
        parallelism=2,
    )

    process = psutil.Process(os.getpid())
    with mlflow.start_run(run_name=f"titanic-{algo}") as run:
        t0 = time.time()
        model = tvs.fit(train)
        train_time = time.time() - t0

        # Metrics
        auc = evaluator.evaluate(model.bestModel.transform(val))
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("train_time_sec", train_time)
        mlflow.log_metric("rss_mb", process.memory_info().rss / 1e6)

        # ---- Extra artifacts for Task 3 compliance ----
        preds = model.bestModel.transform(val).select(
            col("Survived").cast("int").alias("y_true"),
            col("prediction").cast("int").alias("y_pred"),
        )
        cm_pdf = (
            preds.groupBy("y_true", "y_pred").count()
            .toPandas()
            .pivot(index="y_true", columns="y_pred", values="count")
            .fillna(0).astype(int)
        )

        tmpdir = tempfile.mkdtemp()
        cm_path = os.path.join(tmpdir, "confusion_matrix.csv")
        cm_pdf.to_csv(cm_path, index=True)
        mlflow.log_artifact(cm_path, artifact_path="eval")

        bst = model.bestModel.stages[-1]
        fi_df = None
        if hasattr(bst, "featureImportances"):
            vec = np.array(bst.featureImportances.toArray())
            fi_df = pd.DataFrame({"feature": [f"f{i}" for i in range(len(vec))], "importance": vec})
        elif hasattr(bst, "coefficients"):
            vec = np.array(bst.coefficients)
            fi_df = pd.DataFrame({"feature": [f"f{i}" for i in range(len(vec))], "coefficient": vec})
        if fi_df is not None:
            fi_path = os.path.join(tmpdir, "feature_importances.csv")
            fi_df.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path, artifact_path="eval")

        # Log best algo/params (safe)
        try:
            algo_name = type(bst).__name__
            mlflow.log_param("best_algo", algo_name)
            for p in ["numTrees", "maxDepth", "maxIter", "regParam", "elasticNetParam"]:
                if bst.hasParam(p):
                    mlflow.log_param(f"best_{p}", bst.getOrDefault(p))
        except Exception:
            pass
        # ---- End artifacts ----

        # Log & register
        mlflow.spark.log_model(model.bestModel, "model",
                               registered_model_name="titanic_spark_model")
        run_id = run.info.run_id
    return run_id, auc


# Run all three algorithms and collect best
candidates = [hpo_and_log(a) for a in ["rf", "gbt", "lr"]]

# -----------------------------
# Auto-promote best to Production (server-agnostic)
# -----------------------------
if AUTO_PROMOTE:
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    name = "titanic_spark_model"

    best_run, best_auc = max(candidates, key=lambda x: x[1])
    versions = [v for v in client.search_model_versions(f"name='{name}'") if v.run_id == best_run]
    if versions:
        best_v = sorted(versions, key=lambda v: int(v.version))[-1]

        existing = client.search_model_versions(f"name='{name}'")
        prod_versions = [v for v in existing if getattr(v, "current_stage", None) == "Production"]
        for v in prod_versions:
            client.transition_model_version_stage(name, v.version, "Archived")
        client.transition_model_version_stage(name, best_v.version, "Production")
        print(f"Promoted {name} v{best_v.version} (AUC={best_auc:.4f}) to Production")

spark.stop()
print("Titanic training complete.")
