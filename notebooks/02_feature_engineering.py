# Databricks notebook source

# MAGIC %md
# MAGIC # 02 — Feature Engineering
# MAGIC
# MAGIC Build ML-ready features from the Silver layer using PySpark ML pipelines.
# MAGIC Features include time-based, aggregation (rolling windows), categorical encodings,
# MAGIC and scaled numeric features. Output is registered with the Databricks Feature Store.
# MAGIC
# MAGIC **Input:** `analytics_platform.silver.events`, `analytics_platform.silver.transactions`
# MAGIC **Output:** `analytics_platform.gold.feature_table`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup & Imports

# COMMAND ----------

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
)
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import mlflow
import logging

# COMMAND ----------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("feature_engineering")

# Configuration
CATALOG = "analytics_platform"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"
FEATURE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.feature_table"

spark = SparkSession.builder.getOrCreate()
spark.sql(f"USE CATALOG {CATALOG}")

fe_client = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Cleaned Silver Data

# COMMAND ----------

events_df = spark.read.table(f"{CATALOG}.{SILVER_SCHEMA}.events")
transactions_df = spark.read.table(f"{CATALOG}.{SILVER_SCHEMA}.transactions")
users_df = spark.read.table(f"{CATALOG}.{SILVER_SCHEMA}.users")

logger.info("Loaded Silver tables: events=%d, transactions=%d, users=%d",
            events_df.count(), transactions_df.count(), users_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Time-Based Features

# COMMAND ----------

def create_time_features(df: DataFrame, timestamp_col: str) -> DataFrame:
    """Extract temporal features from a timestamp column."""
    return (
        df
        .withColumn("hour_of_day", F.hour(F.col(timestamp_col)))
        .withColumn("day_of_week", F.dayofweek(F.col(timestamp_col)))
        .withColumn("day_of_month", F.dayofmonth(F.col(timestamp_col)))
        .withColumn("month", F.month(F.col(timestamp_col)))
        .withColumn("quarter", F.quarter(F.col(timestamp_col)))
        .withColumn("year", F.year(F.col(timestamp_col)))
        .withColumn(
            "is_weekend",
            F.when(F.dayofweek(F.col(timestamp_col)).isin(1, 7), 1).otherwise(0)
        )
        .withColumn(
            "is_business_hours",
            F.when(
                (F.hour(F.col(timestamp_col)) >= 9) & (F.hour(F.col(timestamp_col)) < 17),
                1
            ).otherwise(0)
        )
        .withColumn(
            "part_of_day",
            F.when(F.hour(F.col(timestamp_col)) < 6, "night")
            .when(F.hour(F.col(timestamp_col)) < 12, "morning")
            .when(F.hour(F.col(timestamp_col)) < 18, "afternoon")
            .otherwise("evening")
        )
    )


events_with_time = create_time_features(events_df, "event_timestamp")
logger.info("Time-based features created.")

# COMMAND ----------

# Days since last event per user
user_event_window = Window.partitionBy("user_id").orderBy("event_timestamp")

events_with_recency = (
    events_with_time
    .withColumn("prev_event_ts", F.lag("event_timestamp").over(user_event_window))
    .withColumn(
        "days_since_last_event",
        F.datediff(F.col("event_timestamp"), F.col("prev_event_ts"))
    )
    .withColumn(
        "seconds_since_last_event",
        F.unix_timestamp("event_timestamp") - F.unix_timestamp("prev_event_ts")
    )
    .fillna({"days_since_last_event": -1, "seconds_since_last_event": -1})
    .drop("prev_event_ts")
)

logger.info("Recency features created.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Aggregation Features (Rolling Windows)

# COMMAND ----------

def create_rolling_aggregation_features(df: DataFrame, user_col: str, date_col: str) -> DataFrame:
    """Build rolling window aggregation features at the user level."""
    # Create a date column for windowing
    df_with_date = df.withColumn("event_date", F.to_date(F.col(date_col)))

    # Daily user-level aggregation base
    daily_user = (
        df_with_date
        .groupBy(user_col, "event_date")
        .agg(
            F.count("*").alias("daily_event_count"),
            F.countDistinct("event_type").alias("daily_unique_event_types"),
            F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0))
            .alias("daily_purchase_count"),
        )
    )

    # Rolling windows: 7d, 30d, 90d
    for window_days in [7, 30, 90]:
        window_spec = (
            Window
            .partitionBy(user_col)
            .orderBy(F.col("event_date").cast("long"))
            .rangeBetween(-(window_days - 1) * 86400, 0)
        )

        daily_user = (
            daily_user
            .withColumn(
                f"event_count_{window_days}d",
                F.sum("daily_event_count").over(window_spec)
            )
            .withColumn(
                f"unique_event_types_{window_days}d",
                F.sum("daily_unique_event_types").over(window_spec)
            )
            .withColumn(
                f"purchase_count_{window_days}d",
                F.sum("daily_purchase_count").over(window_spec)
            )
            .withColumn(
                f"avg_daily_events_{window_days}d",
                F.round(F.avg("daily_event_count").over(window_spec), 4)
            )
        )

    return daily_user


rolling_features = create_rolling_aggregation_features(
    events_with_recency, "user_id", "event_timestamp"
)
logger.info("Rolling aggregation features created.")
display(rolling_features.limit(20))

# COMMAND ----------

# Transaction-based aggregation features
def create_transaction_features(txn_df: DataFrame, user_col: str) -> DataFrame:
    """Aggregate transaction features per user."""
    return (
        txn_df
        .groupBy(user_col)
        .agg(
            F.count("*").alias("total_transactions"),
            F.sum("amount").alias("total_spend"),
            F.avg("amount").alias("avg_transaction_value"),
            F.max("amount").alias("max_transaction_value"),
            F.min("amount").alias("min_transaction_value"),
            F.stddev("amount").alias("stddev_transaction_value"),
            F.countDistinct("product_category").alias("unique_categories_purchased"),
            F.datediff(F.max("transaction_timestamp"), F.min("transaction_timestamp"))
            .alias("customer_tenure_days"),
            F.count(
                F.when(F.col("transaction_status") == "refunded", 1)
            ).alias("refund_count"),
        )
        .withColumn(
            "refund_rate",
            F.round(F.col("refund_count") / F.col("total_transactions"), 4)
        )
    )


txn_features = create_transaction_features(transactions_df, "user_id")
logger.info("Transaction aggregation features created.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Join Feature Sets

# COMMAND ----------

# Take the latest snapshot per user from rolling features
latest_window = Window.partitionBy("user_id").orderBy(F.col("event_date").desc())
latest_rolling = (
    rolling_features
    .withColumn("rn", F.row_number().over(latest_window))
    .filter(F.col("rn") == 1)
    .drop("rn")
)

# Join all feature sets
feature_df = (
    latest_rolling
    .join(txn_features, on="user_id", how="left")
    .join(
        users_df.select("user_id", "signup_date", "user_segment", "country", "platform"),
        on="user_id",
        how="left",
    )
)

# Derived features from user data
feature_df = (
    feature_df
    .withColumn(
        "account_age_days",
        F.datediff(F.col("event_date"), F.col("signup_date"))
    )
    .fillna(0, subset=[
        "total_transactions", "total_spend", "avg_transaction_value",
        "max_transaction_value", "min_transaction_value", "refund_count",
        "refund_rate", "unique_categories_purchased",
    ])
)

logger.info("Feature sets joined. Total feature columns: %d", len(feature_df.columns))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Categorical Encoding

# COMMAND ----------

categorical_cols = ["user_segment", "country", "platform", "part_of_day"]
# Filter to only columns that exist
categorical_cols = [c for c in categorical_cols if c in feature_df.columns]

indexers = [
    StringIndexer(
        inputCol=col, outputCol=f"{col}_index", handleInvalid="keep"
    )
    for col in categorical_cols
]

encoders = [
    OneHotEncoder(
        inputCol=f"{col}_index", outputCol=f"{col}_ohe", dropLast=True
    )
    for col in categorical_cols
]

encoding_pipeline = Pipeline(stages=indexers + encoders)
encoding_model = encoding_pipeline.fit(feature_df)
feature_df_encoded = encoding_model.transform(feature_df)

logger.info("Categorical encoding complete. Columns encoded: %s", categorical_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Feature Scaling

# COMMAND ----------

numeric_feature_cols = [
    "daily_event_count", "daily_unique_event_types",
    "event_count_7d", "event_count_30d", "event_count_90d",
    "purchase_count_7d", "purchase_count_30d", "purchase_count_90d",
    "avg_daily_events_7d", "avg_daily_events_30d", "avg_daily_events_90d",
    "total_transactions", "total_spend", "avg_transaction_value",
    "max_transaction_value", "stddev_transaction_value",
    "unique_categories_purchased", "customer_tenure_days",
    "refund_rate", "account_age_days",
    "hour_of_day", "day_of_week", "is_weekend", "is_business_hours",
    "days_since_last_event",
]

# Filter to only existing columns
numeric_feature_cols = [c for c in numeric_feature_cols if c in feature_df_encoded.columns]

assembler = VectorAssembler(
    inputCols=numeric_feature_cols,
    outputCol="numeric_features_raw",
    handleInvalid="keep",
)

scaler = StandardScaler(
    inputCol="numeric_features_raw",
    outputCol="numeric_features_scaled",
    withMean=True,
    withStd=True,
)

scaling_pipeline = Pipeline(stages=[assembler, scaler])
scaling_model = scaling_pipeline.fit(feature_df_encoded)
feature_df_scaled = scaling_model.transform(feature_df_encoded)

logger.info("Feature scaling complete. %d numeric features scaled.", len(numeric_feature_cols))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Feature Importance Analysis (Preliminary)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler as VA

# Assemble all numeric features for importance ranking
# Use a simple target: did the user make a purchase in the last 7 days?
importance_df = (
    feature_df_encoded
    .withColumn(
        "label",
        F.when(F.col("purchase_count_7d") > 0, 1.0).otherwise(0.0)
    )
    .fillna(0)
)

importance_assembler = VA(
    inputCols=numeric_feature_cols,
    outputCol="importance_features",
    handleInvalid="skip",
)

importance_assembled = importance_assembler.transform(importance_df).select(
    "importance_features", "label"
)

rf = RandomForestClassifier(
    featuresCol="importance_features",
    labelCol="label",
    numTrees=50,
    maxDepth=8,
    seed=42,
)

rf_model = rf.fit(importance_assembled)

# Extract feature importances
importances = rf_model.featureImportances.toArray()
feature_importance_data = [
    {"feature": col, "importance": round(float(imp), 6)}
    for col, imp in zip(numeric_feature_cols, importances)
]
feature_importance_df = (
    spark.createDataFrame(feature_importance_data)
    .orderBy(F.col("importance").desc())
)

print("--- Feature Importance (Random Forest) ---")
display(feature_importance_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Save Feature Table to Delta

# COMMAND ----------

# Select final features for the output table
output_cols = (
    ["user_id", "event_date"]
    + numeric_feature_cols
    + [f"{c}_index" for c in categorical_cols]
    + ["numeric_features_scaled"]
)
output_cols = [c for c in output_cols if c in feature_df_scaled.columns]

final_features = feature_df_scaled.select(output_cols)

(
    final_features
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.autoOptimize.optimizeWrite", "true")
    .option("delta.autoOptimize.autoCompact", "true")
    .saveAsTable(FEATURE_TABLE)
)

logger.info("Feature table saved to %s with %d rows.", FEATURE_TABLE, final_features.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Register with Feature Store

# COMMAND ----------

# Create the feature table in the Feature Store
fe_client.create_table(
    name=FEATURE_TABLE,
    primary_keys=["user_id", "event_date"],
    timestamp_keys=["event_date"],
    df=final_features,
    description="User-level features for ML models. Includes time-based, rolling aggregation, "
                "transaction, and encoded categorical features.",
    tags={"team": "data-science", "project": "analytics-platform", "version": "1.0"},
)

logger.info("Feature table registered with Feature Store: %s", FEATURE_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Feature Category | Count | Examples |
# MAGIC |---|---|---|
# MAGIC | Time-based | 9 | hour_of_day, day_of_week, is_weekend |
# MAGIC | Recency | 2 | days_since_last_event, seconds_since_last_event |
# MAGIC | Rolling Aggregation | 12 | event_count_7d/30d/90d, purchase_count_* |
# MAGIC | Transaction | 9 | total_spend, avg_transaction_value, refund_rate |
# MAGIC | Categorical (encoded) | 4+ | user_segment, country, platform |
# MAGIC | Scaled Numeric | 1 vector | numeric_features_scaled |
# MAGIC
# MAGIC **Next:** Use these features in `03_model_training.py` for model training.
