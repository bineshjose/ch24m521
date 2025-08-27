FROM continuumio/miniconda3:latest

WORKDIR /ops
# Conda installs JRE 17 and Python; avoids apt
RUN conda install -y -c conda-forge python=3.10 openjdk=17 && conda clean -afy
ENV JAVA_HOME=/opt/conda
ENV PATH="/opt/conda/bin:${PATH}"
RUN pip install --no-cache-dir mlflow pyspark pandas numpy requests
COPY retrainer.py .
CMD ["python","retrainer.py"]
