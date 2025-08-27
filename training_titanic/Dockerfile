# training_titanic/Dockerfile
FROM mambaorg/micromamba:1.5.10

USER root
ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /app

# Create env (Java 17 + PySpark + MLflow + essentials)
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean -a -y

ENV JAVA_HOME=/opt/conda
ENV PATH=/opt/conda/bin:$PATH

# Spark S3 support
ENV SPARK_PACKAGES="org.apache.hadoop:hadoop-aws:3.3.2,com.amazonaws:aws-java-sdk-bundle:1.12.262"
ENV PYSPARK_SUBMIT_ARGS="--packages ${SPARK_PACKAGES} pyspark-shell"

# *** This is the part you were trying to "run" — it belongs here ***
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

CMD ["python","train_titanic.py"]
