import os
from urllib.parse import urlparse

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when


mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

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

spark = build_spark("titanic-preprocess")

# Read raw CSV (Kaggle Titanic)
df = spark.read.csv("data/titanic.csv", header=True, inferSchema=True)

# Basic cleaning
df = (
    df.withColumn("Sex", when(col("Sex").isNull(), "unknown").otherwise(col("Sex")))
      .withColumn("Embarked", when(col("Embarked").isNull(), "S").otherwise(col("Embarked")))
)

# Write processed parquet (snappy)
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")
df.write.mode("overwrite").parquet("data/processed/titanic.parquet")

print("Wrote data/processed/titanic.parquet")
spark.stop()
