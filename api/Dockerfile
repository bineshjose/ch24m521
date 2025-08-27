# api/Dockerfile
FROM mambaorg/micromamba:1.5.10

USER root
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app

# Environment: Java 17 + Python + PySpark + MLflow + FastAPI + boto3
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean -a -y

ENV JAVA_HOME=/opt/conda
ENV PATH=/opt/conda/bin:$PATH

# S3 support for Spark when needed
ENV SPARK_PACKAGES="org.apache.hadoop:hadoop-aws:3.3.2,com.amazonaws:aws-java-sdk-bundle:1.12.262"
ENV PYSPARK_SUBMIT_ARGS="--packages ${SPARK_PACKAGES} pyspark-shell"

COPY --chown=$MAMBA_USER:$MAMBA_USER app.py /app/app.py

EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
