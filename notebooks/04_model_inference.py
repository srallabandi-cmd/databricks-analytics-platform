# Databricks notebook source

# MAGIC %md
# MAGIC # 04 — Model Inference
# MAGIC
# MAGIC Load the registered model from MLflow Model Registry and run batch and real-time
# MAGIC inference. Includes prediction monitoring and drift detection.
# MAGIC
# MAGIC **Model:** `purchase_prediction_model` (Production stage)
# MAGIC **Output:** `analytics_platform.gold.predictions`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup & Imports

# COMMAND ----------

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, ArrayType, StructType, StructField, StringType
from pyspark.sql.window import Window
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from mlflow import MlflowClient
import mlflow
import json
import logging
from datetime import datetime, timedelta

# COMMAND ----------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_inference")

# Configuration
CATALOG = "analytics_platform"
GOLD_SCHEMA = "gold"
FEATURE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.feature_table"
PREDICTIONS_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.predictions"
DRIFT_METRICS_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.drift_metrics"
MODEL_NAME = "purchase_prediction_model"

spark = SparkSession.builder.getOrCreate()
spark.sql(f"USE CATALOG {CATALOG}")

fe_client = FeatureEngineeringClient()
mlflow_client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Model from Registry

# COMMAND ----------

def load_production_model(model_name: str):
    """Load the Production-stage model from MLflow Model Registry."""
    production_versions = mlflow_client.get_latest_versions(model_name, stages=["Production"])

    if not production_versions:
        # Fall back to Staging if no Production model exists
        logger.warning("No Production model found. Falling back to Staging.")
        production_versions = mlflow_client.get_latest_versions(model_name, stages=["Staging"])

    if not production_versions:
        raise ValueError(f"No Production or Staging model found for '{model_name}'")

    model_version = production_versions[0]
    model_uri = f"models:/{model_name}/{model_version.version}"

    logger.info("Loading model: %s (version %s, stage: %s)",
                model_name, model_version.version, model_version.current_stage)

    model = mlflow.spark.load_model(model_uri)
    return model, model_version


model, model_version_info = load_production_model(MODEL_NAME)
logger.info("Model loaded successfully. Run ID: %s", model_version_info.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Batch Inference

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Load New Data for Scoring

# COMMAND ----------

def load_scoring_data(lookback_days: int = 1) -> DataFrame:
    """Load new data that needs scoring, joining with Feature Store."""
    cutoff_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    # Load users that need scoring (e.g., active in the last N days)
    scoring_keys = (
        spark.read.table(f"{CATALOG}.{GOLD_SCHEMA}.feature_table")
        .filter(F.col("event_date") >= cutoff_date)
        .select("user_id", "event_date")
        .distinct()
    )

    logger.info("Scoring data: %d user-date combinations since %s",
                scoring_keys.count(), cutoff_date)
    return scoring_keys


scoring_keys = load_scoring_data(lookback_days=1)
display(scoring_keys.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Apply Feature Engineering & Generate Predictions

# COMMAND ----------

def run_batch_inference(
    model_name: str,
    scoring_keys: DataFrame,
    feature_table: str,
) -> DataFrame:
    """Run batch inference using Feature Store for automatic feature lookup."""
    feature_lookups = [
        FeatureLookup(
            table_name=feature_table,
            lookup_key=["user_id", "event_date"],
        )
    ]

    # Score using Feature Engineering client (handles feature lookup automatically)
    scored_df = fe_client.score_batch(
        model_uri=f"models:/{model_name}/Production",
        df=scoring_keys,
    )

    # Enrich predictions with metadata
    scored_df = (
        scored_df
        .withColumn("prediction_timestamp", F.current_timestamp())
        .withColumn("model_name", F.lit(model_name))
        .withColumn("model_version", F.lit(model_version_info.version))
        .withColumn(
            "prediction_label",
            F.when(F.col("prediction") == 1.0, "will_purchase")
            .otherwise("will_not_purchase")
        )
    )

    return scored_df


predictions_df = run_batch_inference(MODEL_NAME, scoring_keys, FEATURE_TABLE)
logger.info("Batch inference complete: %d predictions generated.", predictions_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Write Predictions to Delta Table

# COMMAND ----------

# Merge predictions into the predictions table (upsert)
predictions_df.createOrReplaceTempView("new_predictions")

spark.sql(f"""
    MERGE INTO {PREDICTIONS_TABLE} AS target
    USING new_predictions AS source
    ON target.user_id = source.user_id
       AND target.event_date = source.event_date
    WHEN MATCHED THEN UPDATE SET
        target.prediction = source.prediction,
        target.probability = source.probability,
        target.prediction_label = source.prediction_label,
        target.prediction_timestamp = source.prediction_timestamp,
        target.model_name = source.model_name,
        target.model_version = source.model_version
    WHEN NOT MATCHED THEN INSERT *
""")

logger.info("Predictions written to %s", PREDICTIONS_TABLE)

# COMMAND ----------

# Prediction distribution summary
prediction_summary = (
    predictions_df
    .groupBy("prediction_label")
    .agg(
        F.count("*").alias("count"),
        F.avg("probability").alias("avg_probability"),
    )
)
display(prediction_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Real-Time Inference Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Model Serving Endpoint Configuration

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    AutoCaptureConfigInput,
)

def create_serving_endpoint(model_name: str, model_version: str):
    """Create or update a Databricks Model Serving endpoint."""
    w = WorkspaceClient()
    endpoint_name = model_name.replace("_", "-")

    endpoint_config = EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=model_name,
                entity_version=model_version,
                workload_size="Small",
                scale_to_zero_enabled=True,
                environment_vars={
                    "DATABRICKS_TOKEN": "{{secrets/ml-serving/databricks-token}}",
                },
            )
        ],
        auto_capture_config=AutoCaptureConfigInput(
            catalog_name=CATALOG,
            schema_name=GOLD_SCHEMA,
            table_name_prefix="serving_",
            enabled=True,
        ),
    )

    try:
        existing = w.serving_endpoints.get(endpoint_name)
        logger.info("Updating existing endpoint: %s", endpoint_name)
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=endpoint_config.served_entities,
            auto_capture_config=endpoint_config.auto_capture_config,
        )
    except Exception:
        logger.info("Creating new endpoint: %s", endpoint_name)
        w.serving_endpoints.create(
            name=endpoint_name,
            config=endpoint_config,
        )

    return endpoint_name


endpoint_name = create_serving_endpoint(MODEL_NAME, model_version_info.version)
logger.info("Serving endpoint configured: %s", endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Request / Response Format

# COMMAND ----------

# Example: query the serving endpoint
import requests

def query_serving_endpoint(endpoint_name: str, features: dict) -> dict:
    """Send a prediction request to the model serving endpoint."""
    workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
    token = dbutils.secrets.get(scope="ml-serving", key="databricks-token")

    url = f"https://{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {
        "dataframe_records": [features]
    }

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


# Example request format
example_request = {
    "user_id": "user_12345",
    "event_date": "2025-01-15",
    "daily_event_count": 12,
    "event_count_7d": 78,
    "event_count_30d": 245,
    "total_spend": 1250.50,
    "avg_transaction_value": 62.52,
    "is_weekend": 0,
    "hour_of_day": 14,
}

print("Example request format:")
print(json.dumps(example_request, indent=2))

# Uncomment to test:
# result = query_serving_endpoint(endpoint_name, example_request)
# print("Prediction:", result)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Monitor Predictions

# COMMAND ----------

def monitor_prediction_distribution(
    predictions_table: str,
    lookback_hours: int = 24,
) -> DataFrame:
    """Monitor recent prediction distributions for anomalies."""
    cutoff = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()

    recent_preds = (
        spark.read.table(predictions_table)
        .filter(F.col("prediction_timestamp") >= cutoff)
    )

    if recent_preds.count() == 0:
        logger.warning("No predictions found in the last %d hours.", lookback_hours)
        return None

    monitoring = recent_preds.agg(
        F.count("*").alias("total_predictions"),
        F.sum(F.when(F.col("prediction") == 1.0, 1).otherwise(0)).alias("positive_predictions"),
        F.sum(F.when(F.col("prediction") == 0.0, 1).otherwise(0)).alias("negative_predictions"),
        F.avg("probability").alias("avg_probability"),
        F.stddev("probability").alias("stddev_probability"),
        F.min("prediction_timestamp").alias("window_start"),
        F.max("prediction_timestamp").alias("window_end"),
    ).withColumn(
        "positive_rate",
        F.round(F.col("positive_predictions") / F.col("total_predictions") * 100, 2)
    )

    return monitoring


prediction_monitor = monitor_prediction_distribution(PREDICTIONS_TABLE, lookback_hours=24)
if prediction_monitor:
    display(prediction_monitor)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Drift Detection

# COMMAND ----------

def compute_feature_drift(
    reference_df: DataFrame,
    current_df: DataFrame,
    numeric_cols: list,
    method: str = "psi",
) -> DataFrame:
    """
    Compute Population Stability Index (PSI) for feature drift detection.

    PSI < 0.1  -> No significant drift
    PSI 0.1-0.2 -> Moderate drift (monitor)
    PSI > 0.2  -> Significant drift (retrain)
    """
    drift_results = []
    num_buckets = 10

    for col_name in numeric_cols:
        ref_col = reference_df.select(col_name).dropna()
        cur_col = current_df.select(col_name).dropna()

        if ref_col.count() == 0 or cur_col.count() == 0:
            continue

        # Compute percentile-based buckets from reference distribution
        percentiles = [i / num_buckets for i in range(1, num_buckets)]
        ref_quantiles = ref_col.approxQuantile(col_name, percentiles, 0.01)

        if not ref_quantiles:
            continue

        # Bucket both distributions
        boundaries = [-float("inf")] + ref_quantiles + [float("inf")]

        ref_counts = []
        cur_counts = []
        ref_total = ref_col.count()
        cur_total = cur_col.count()

        for i in range(len(boundaries) - 1):
            lower = boundaries[i]
            upper = boundaries[i + 1]
            ref_count = ref_col.filter(
                (F.col(col_name) > lower) & (F.col(col_name) <= upper)
            ).count()
            cur_count = cur_col.filter(
                (F.col(col_name) > lower) & (F.col(col_name) <= upper)
            ).count()
            ref_counts.append(max(ref_count / ref_total, 0.0001))
            cur_counts.append(max(cur_count / cur_total, 0.0001))

        # PSI calculation
        import math
        psi = sum(
            (cur_counts[i] - ref_counts[i]) * math.log(cur_counts[i] / ref_counts[i])
            for i in range(len(ref_counts))
        )

        drift_status = (
            "no_drift" if psi < 0.1
            else "moderate_drift" if psi < 0.2
            else "significant_drift"
        )

        drift_results.append({
            "feature": col_name,
            "psi": round(psi, 6),
            "drift_status": drift_status,
            "reference_mean": float(ref_col.agg(F.mean(col_name)).collect()[0][0] or 0),
            "current_mean": float(cur_col.agg(F.mean(col_name)).collect()[0][0] or 0),
        })

    return spark.createDataFrame(drift_results)


# Reference: training data distribution; Current: latest scoring data
reference_features = spark.read.table(FEATURE_TABLE).filter(
    F.col("event_date") < F.date_sub(F.current_date(), 30)
)
current_features = spark.read.table(FEATURE_TABLE).filter(
    F.col("event_date") >= F.date_sub(F.current_date(), 7)
)

numeric_drift_cols = [
    "daily_event_count", "event_count_7d", "event_count_30d",
    "total_spend", "avg_transaction_value", "total_transactions",
]

drift_report = compute_feature_drift(reference_features, current_features, numeric_drift_cols)
display(drift_report)

# COMMAND ----------

# Save drift metrics
(
    drift_report
    .withColumn("analysis_timestamp", F.current_timestamp())
    .withColumn("model_version", F.lit(model_version_info.version))
    .write
    .format("delta")
    .mode("append")
    .saveAsTable(DRIFT_METRICS_TABLE)
)

logger.info("Drift metrics saved to %s", DRIFT_METRICS_TABLE)

# COMMAND ----------

# Alert on significant drift
significant_drift = drift_report.filter(F.col("drift_status") == "significant_drift")
if significant_drift.count() > 0:
    drifted_features = [row.feature for row in significant_drift.collect()]
    logger.warning(
        "ALERT: Significant drift detected in features: %s. Consider retraining.",
        drifted_features,
    )
    # In production, trigger notification via webhook or Databricks notification
else:
    logger.info("No significant feature drift detected.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference Summary
# MAGIC
# MAGIC | Component | Status |
# MAGIC |---|---|
# MAGIC | Model Loading | Loaded from Registry (Production/Staging) |
# MAGIC | Batch Inference | Predictions written to Delta |
# MAGIC | Real-Time Endpoint | Configured with auto-capture |
# MAGIC | Prediction Monitoring | Distribution tracked |
# MAGIC | Drift Detection | PSI computed for key features |
