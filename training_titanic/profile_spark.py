import os,time, mlflow, mlflow.spark, psutil
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, Imputer
from pyspark.ml.evaluation import BinaryClassificationEvaluator

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","http://mlflow:5000"))
mlflow.set_experiment("titanic-profiling")

def run_with_conf(exec_mem, drv_mem, parallelism):
    spark=(SparkSession.builder.appName("profile")
           .config("spark.executor.memory", exec_mem)
           .config("spark.driver.memory", drv_mem)
           .config("spark.default.parallelism", str(parallelism))
           .getOrCreate())
    df=spark.read.csv("data/titanic.csv", header=True, inferSchema=True)
    im=Imputer(strategy="median", inputCols=["Age","Fare"], outputCols=["Age","Fare"])
    sx=StringIndexer(inputCol="Sex", outputCol="SexIdx", handleInvalid="keep")
    em=StringIndexer(inputCol="Embarked", outputCol="EmbarkedIdx", handleInvalid="keep")
    ohe=OneHotEncoder(inputCols=["SexIdx","EmbarkedIdx"], outputCols=["SexOH","EmbarkedOH"])
    va=VectorAssembler(inputCols=["Pclass","Age","SibSp","Parch","Fare","SexOH","EmbarkedOH"], outputCol="features")
    rf=RandomForestClassifier(labelCol="Survived", featuresCol="features", numTrees=200, maxDepth=7, seed=42)
    pipe=Pipeline(stages=[im,sx,em,ohe,va,rf])
    tr,te=df.randomSplit([0.8,0.2], seed=42)
    ps=psutil.Process(os.getpid())
    with mlflow.start_run(run_name=f"profile-{exec_mem}-{drv_mem}-{parallelism}"):
        t0=time.time(); model=pipe.fit(tr); t=time.time()-t0
        auc=BinaryClassificationEvaluator(labelCol="Survived").evaluate(model.transform(te))
        mlflow.log_params({"exec_mem":exec_mem,"drv_mem":drv_mem,"parallelism":parallelism})
        mlflow.log_metrics({"AUC":auc,"train_time_sec":t,"rss_mb":ps.memory_info().rss/1e6})
        print(exec_mem, drv_mem, parallelism, "=> AUC", auc, "time", t)
    spark.stop()

for exec_mem in ["1g","2g"]:
    for drv_mem in ["1g","2g"]:
        for par in [2,4,8]:
            run_with_conf(exec_mem, drv_mem, par)
