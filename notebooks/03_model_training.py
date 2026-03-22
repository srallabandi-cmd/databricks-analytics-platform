# Databricks notebook source

# MAGIC %md
# MAGIC # 03 — Model Training
# MAGIC
# MAGIC Train an XGBoost classification model using features from the Feature Store.
# MAGIC Full MLflow experiment tracking with autologging, hyperparameter logging,
# MAGIC cross-validation, and model registration.
# MAGIC
# MAGIC **Input:** Feature Store table `analytics_platform.gold.feature_table`
# MAGIC **Output:** Registered model in MLflow Model Registry

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup & Configuration

# COMMAND ----------

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import mlflow
from mlflow.models.signature import infer_signature
import xgboost as xgb
from xgboost.spark import SparkXGBClassifier
import numpy as np
import logging
import json
from datetime import datetime

# COMMAND ----------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_training")

# Configuration
CATALOG = "analytics_platform"
GOLD_SCHEMA = "gold"
FEATURE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.feature_table"
EXPERIMENT_NAME = "/Shared/analytics_platform/purchase_prediction"
MODEL_NAME = "purchase_prediction_model"
RANDOM_SEED = 42

spark = SparkSession.builder.getOrCreate()
spark.sql(f"USE CATALOG {CATALOG}")

fe_client = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Features from Feature Store

# COMMAND ----------

# Define training set from Feature Store
# The label table contains user_id, event_date, and the target label
label_df = (
    spark.read.table(f"{CATALOG}.{GOLD_SCHEMA}.training_labels")
    .select("user_id", "event_date", "label")
)

feature_lookups = [
    FeatureLookup(
        table_name=FEATURE_TABLE,
        lookup_key=["user_id", "event_date"],
    )
]

training_set = fe_client.create_training_set(
    df=label_df,
    feature_lookups=feature_lookups,
    label="label",
    exclude_columns=["user_id", "event_date"],
)

training_df = training_set.load_df()
logger.info("Training set loaded: %d rows, %d columns", training_df.count(), len(training_df.columns))

# Label distribution
label_dist = training_df.groupBy("label").count()
display(label_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Train / Test Split with Stratification

# COMMAND ----------

def stratified_split(df: DataFrame, label_col: str, train_ratio: float = 0.8, seed: int = 42):
    """Split DataFrame with stratification to maintain label distribution."""
    positive = df.filter(F.col(label_col) == 1.0)
    negative = df.filter(F.col(label_col) == 0.0)

    pos_train, pos_test = positive.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    neg_train, neg_test = negative.randomSplit([train_ratio, 1 - train_ratio], seed=seed)

    train_df = pos_train.unionByName(neg_train)
    test_df = pos_test.unionByName(neg_test)

    return train_df, test_df


train_df, test_df = stratified_split(training_df, "label", train_ratio=0.8, seed=RANDOM_SEED)

train_count = train_df.count()
test_count = test_df.count()
logger.info("Train: %d rows | Test: %d rows", train_count, test_count)

# Verify stratification
print("--- Train label distribution ---")
display(train_df.groupBy("label").count().withColumn("pct", F.round(F.col("count") / F.lit(train_count) * 100, 2)))
print("--- Test label distribution ---")
display(test_df.groupBy("label").count().withColumn("pct", F.round(F.col("count") / F.lit(test_count) * 100, 2)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature Assembly

# COMMAND ----------

# Identify numeric feature columns (exclude label and metadata)
exclude_cols = {"label", "user_id", "event_date", "numeric_features_scaled"}
feature_cols = [
    c for c in training_df.columns
    if c not in exclude_cols
    and training_df.schema[c].dataType.typeName() in ("double", "integer", "long", "float")
]

logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols[:10])

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="keep",
)

train_assembled = assembler.transform(train_df).select("features", "label")
test_assembled = assembler.transform(test_df).select("features", "label")

# Cache for faster iteration
train_assembled.cache()
test_assembled.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. MLflow Experiment Setup

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.autolog(log_models=True, log_input_examples=True, log_model_signatures=True)

logger.info("MLflow experiment: %s", EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. XGBoost Model Training

# COMMAND ----------

xgb_params = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1.0,
    "eval_metric": "auc",
    "random_state": RANDOM_SEED,
}

with mlflow.start_run(run_name="xgboost_baseline") as run:
    xgb_run_id = run.info.run_id

    # Log parameters explicitly for clarity
    mlflow.log_params(xgb_params)
    mlflow.log_param("num_features", len(feature_cols))
    mlflow.log_param("train_rows", train_count)
    mlflow.log_param("test_rows", test_count)
    mlflow.set_tag("model_type", "xgboost")
    mlflow.set_tag("training_date", datetime.now().isoformat())

    # SparkXGBClassifier for distributed training
    xgb_classifier = SparkXGBClassifier(
        features_col="features",
        label_col="label",
        prediction_col="prediction",
        probability_col="probability",
        raw_prediction_col="rawPrediction",
        max_depth=xgb_params["max_depth"],
        learning_rate=xgb_params["learning_rate"],
        n_estimators=xgb_params["n_estimators"],
        subsample=xgb_params["subsample"],
        colsample_bytree=xgb_params["colsample_bytree"],
        min_child_weight=xgb_params["min_child_weight"],
        gamma=xgb_params["gamma"],
        reg_alpha=xgb_params["reg_alpha"],
        reg_lambda=xgb_params["reg_lambda"],
        scale_pos_weight=xgb_params["scale_pos_weight"],
        eval_metric=xgb_params["eval_metric"],
        random_state=RANDOM_SEED,
        num_workers=4,
        use_gpu=False,
    )

    logger.info("Training XGBoost model...")
    xgb_model = xgb_classifier.fit(train_assembled)

    # Predictions
    train_preds = xgb_model.transform(train_assembled)
    test_preds = xgb_model.transform(test_assembled)

    # Evaluation
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    pr_evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
    )
    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="weightedRecall"
    )
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    metrics = {
        "train_auc": auc_evaluator.evaluate(train_preds),
        "test_auc": auc_evaluator.evaluate(test_preds),
        "test_pr_auc": pr_evaluator.evaluate(test_preds),
        "test_accuracy": accuracy_evaluator.evaluate(test_preds),
        "test_precision": precision_evaluator.evaluate(test_preds),
        "test_recall": recall_evaluator.evaluate(test_preds),
        "test_f1": f1_evaluator.evaluate(test_preds),
    }

    mlflow.log_metrics(metrics)
    logger.info("XGBoost metrics: %s", json.dumps({k: round(v, 4) for k, v in metrics.items()}))

    # Confusion matrix
    confusion = (
        test_preds
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
    )
    display(confusion)

    # Log confusion matrix as artifact
    confusion_pd = confusion.toPandas()
    confusion_path = "/tmp/confusion_matrix.json"
    confusion_pd.to_json(confusion_path, orient="records")
    mlflow.log_artifact(confusion_path, "evaluation")

    # Log feature importance
    feature_importance_dict = {
        feature_cols[i]: float(xgb_model.get_feature_importances()[i])
        for i in range(len(feature_cols))
        if i < len(xgb_model.get_feature_importances())
    }
    mlflow.log_dict(feature_importance_dict, "feature_importance.json")

    logger.info("XGBoost baseline run complete: %s", xgb_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. GBT Classifier Comparison

# COMMAND ----------

with mlflow.start_run(run_name="spark_gbt_baseline") as gbt_run:
    gbt_run_id = gbt_run.info.run_id

    gbt_classifier = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        maxDepth=8,
        maxIter=200,
        stepSize=0.05,
        subsamplingRate=0.8,
        featureSubsetStrategy="0.8",
        minInstancesPerNode=5,
        seed=RANDOM_SEED,
    )

    mlflow.log_params({
        "model_type": "spark_gbt",
        "maxDepth": 8,
        "maxIter": 200,
        "stepSize": 0.05,
        "subsamplingRate": 0.8,
    })

    logger.info("Training Spark GBT model...")
    gbt_model = gbt_classifier.fit(train_assembled)

    gbt_test_preds = gbt_model.transform(test_assembled)

    gbt_metrics = {
        "test_auc": auc_evaluator.evaluate(gbt_test_preds),
        "test_pr_auc": pr_evaluator.evaluate(gbt_test_preds),
        "test_accuracy": accuracy_evaluator.evaluate(gbt_test_preds),
        "test_precision": precision_evaluator.evaluate(gbt_test_preds),
        "test_recall": recall_evaluator.evaluate(gbt_test_preds),
        "test_f1": f1_evaluator.evaluate(gbt_test_preds),
    }

    mlflow.log_metrics(gbt_metrics)
    logger.info("GBT metrics: %s", json.dumps({k: round(v, 4) for k, v in gbt_metrics.items()}))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Cross-Validation (XGBoost)

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_cross_validation") as cv_run:
    cv_run_id = cv_run.info.run_id

    xgb_cv = SparkXGBClassifier(
        features_col="features",
        label_col="label",
        max_depth=8,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        num_workers=4,
    )

    param_grid = (
        ParamGridBuilder()
        .addGrid(xgb_cv.max_depth, [6, 8, 10])
        .addGrid(xgb_cv.learning_rate, [0.01, 0.05, 0.1])
        .addGrid(xgb_cv.n_estimators, [200, 500])
        .build()
    )

    cv = CrossValidator(
        estimator=xgb_cv,
        estimatorParamMaps=param_grid,
        evaluator=auc_evaluator,
        numFolds=5,
        parallelism=4,
        seed=RANDOM_SEED,
    )

    logger.info("Running 5-fold cross-validation with %d parameter combinations...",
                len(param_grid))

    cv_model = cv.fit(train_assembled)

    # Best model results
    best_model = cv_model.bestModel
    cv_test_preds = cv_model.transform(test_assembled)

    cv_metrics = {
        "cv_best_test_auc": auc_evaluator.evaluate(cv_test_preds),
        "cv_best_test_accuracy": accuracy_evaluator.evaluate(cv_test_preds),
        "cv_best_test_f1": f1_evaluator.evaluate(cv_test_preds),
        "cv_avg_metrics": float(np.mean(cv_model.avgMetrics)),
        "cv_std_metrics": float(np.std(cv_model.avgMetrics)),
    }

    mlflow.log_metrics(cv_metrics)

    # Log best parameters
    best_params = {
        "best_max_depth": best_model.getOrDefault("max_depth"),
        "best_learning_rate": best_model.getOrDefault("learning_rate"),
        "best_n_estimators": best_model.getOrDefault("n_estimators"),
    }
    mlflow.log_params(best_params)

    logger.info("Cross-validation complete. Best AUC: %.4f", cv_metrics["cv_best_test_auc"])
    logger.info("Best params: %s", best_params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Select & Register Best Model

# COMMAND ----------

# Compare all runs
all_results = {
    "xgboost_baseline": {"run_id": xgb_run_id, "auc": metrics["test_auc"]},
    "spark_gbt": {"run_id": gbt_run_id, "auc": gbt_metrics["test_auc"]},
    "xgboost_cv": {"run_id": cv_run_id, "auc": cv_metrics["cv_best_test_auc"]},
}

best_model_name = max(all_results, key=lambda k: all_results[k]["auc"])
best_result = all_results[best_model_name]
logger.info("Best model: %s (AUC=%.4f, run_id=%s)", best_model_name, best_result["auc"], best_result["run_id"])

print(f"\nBest Model: {best_model_name}")
print(f"  AUC: {best_result['auc']:.4f}")
print(f"  Run ID: {best_result['run_id']}")

# COMMAND ----------

# Register the best model using Feature Engineering client
best_run_uri = f"runs:/{best_result['run_id']}/model"

with mlflow.start_run(run_id=best_result["run_id"]):
    # Log the model with the Feature Store for lineage
    fe_client.log_model(
        model=cv_model.bestModel if best_model_name == "xgboost_cv" else xgb_model,
        artifact_path="model",
        flavor=mlflow.spark,
        training_set=training_set,
        registered_model_name=MODEL_NAME,
    )

logger.info("Model registered as '%s'", MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Transition to Staging

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

# Get latest version
latest_versions = client.get_latest_versions(MODEL_NAME)
latest_version = max(latest_versions, key=lambda v: int(v.version))

# Transition to Staging
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=latest_version.version,
    stage="Staging",
    archive_existing_versions=True,
)

# Add description
client.update_model_version(
    name=MODEL_NAME,
    version=latest_version.version,
    description=(
        f"Purchase prediction model ({best_model_name}). "
        f"Test AUC: {best_result['auc']:.4f}. "
        f"Trained on {train_count} samples with {len(feature_cols)} features. "
        f"Promoted to Staging on {datetime.now().isoformat()}"
    ),
)

logger.info("Model version %s transitioned to Staging.", latest_version.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Summary
# MAGIC
# MAGIC | Model | Test AUC | Status |
# MAGIC |---|---|---|
# MAGIC | XGBoost Baseline | see metrics above | Evaluated |
# MAGIC | Spark GBT | see metrics above | Evaluated |
# MAGIC | XGBoost + Cross-Validation | see metrics above | **Best -> Staging** |
# MAGIC
# MAGIC **Next:** Validate the Staging model in `04_model_inference.py`, then promote to Production.
