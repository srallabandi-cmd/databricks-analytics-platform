# Databricks notebook source

# MAGIC %md
# MAGIC # 01 — Exploratory Data Analysis
# MAGIC
# MAGIC This notebook performs comprehensive EDA on the raw and cleaned datasets stored in
# MAGIC Delta Lake. We inspect schemas, assess data quality, compute statistical summaries,
# MAGIC analyze distributions and correlations, and identify temporal trends.
# MAGIC
# MAGIC **Data source:** `analytics_platform.silver.events`
# MAGIC
# MAGIC **Outputs:** Summary statistics tables written to `analytics_platform.gold.eda_summaries`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup & Configuration

# COMMAND ----------

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, TimestampType, LongType, BooleanType,
)
from pyspark.sql.window import Window
import json

# COMMAND ----------

# Configuration
CATALOG = "analytics_platform"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"
EVENTS_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.events"
USERS_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.users"
TRANSACTIONS_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.transactions"
EDA_OUTPUT_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.eda_summaries"

spark = SparkSession.builder.getOrCreate()
spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Data from Delta Lake

# COMMAND ----------

events_df = spark.read.table(EVENTS_TABLE)
users_df = spark.read.table(USERS_TABLE)
transactions_df = spark.read.table(TRANSACTIONS_TABLE)

print(f"Events table loaded: {EVENTS_TABLE}")
print(f"Users table loaded: {USERS_TABLE}")
print(f"Transactions table loaded: {TRANSACTIONS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Schema Inspection

# COMMAND ----------

def inspect_schema(df: DataFrame, table_name: str) -> None:
    """Print schema details and basic metadata for a DataFrame."""
    print(f"\n{'='*60}")
    print(f"Schema for: {table_name}")
    print(f"{'='*60}")
    df.printSchema()

    schema_summary = []
    for field in df.schema.fields:
        schema_summary.append({
            "column_name": field.name,
            "data_type": str(field.dataType),
            "nullable": field.nullable,
        })

    schema_df = spark.createDataFrame(schema_summary)
    display(schema_df)


inspect_schema(events_df, EVENTS_TABLE)
inspect_schema(users_df, USERS_TABLE)
inspect_schema(transactions_df, TRANSACTIONS_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Row Counts & Partitioning

# COMMAND ----------

tables = {
    "events": events_df,
    "users": users_df,
    "transactions": transactions_df,
}

row_counts = []
for name, df in tables.items():
    count = df.count()
    num_cols = len(df.columns)
    num_partitions = df.rdd.getNumPartitions()
    row_counts.append({
        "table": name,
        "row_count": count,
        "column_count": num_cols,
        "num_partitions": num_partitions,
    })
    print(f"{name}: {count:,} rows | {num_cols} columns | {num_partitions} partitions")

row_counts_df = spark.createDataFrame(row_counts)
display(row_counts_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Null Analysis

# COMMAND ----------

def null_analysis(df: DataFrame, table_name: str) -> DataFrame:
    """Compute null counts and percentages for every column."""
    total_rows = df.count()
    if total_rows == 0:
        print(f"WARNING: {table_name} has zero rows.")
        return spark.createDataFrame([], schema="column_name STRING, null_count LONG, null_pct DOUBLE")

    null_exprs = []
    for col_name in df.columns:
        null_exprs.append(
            F.sum(F.when(F.col(col_name).isNull(), 1).otherwise(0)).alias(col_name)
        )

    null_counts_row = df.select(null_exprs).collect()[0]

    results = []
    for col_name in df.columns:
        null_count = null_counts_row[col_name]
        null_pct = round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        results.append({
            "table": table_name,
            "column_name": col_name,
            "null_count": int(null_count),
            "null_pct": null_pct,
            "total_rows": total_rows,
        })

    result_df = spark.createDataFrame(results)
    return result_df


print("--- Events Null Analysis ---")
events_nulls = null_analysis(events_df, "events")
display(events_nulls.filter(F.col("null_pct") > 0).orderBy(F.col("null_pct").desc()))

print("--- Users Null Analysis ---")
users_nulls = null_analysis(users_df, "users")
display(users_nulls.filter(F.col("null_pct") > 0).orderBy(F.col("null_pct").desc()))

print("--- Transactions Null Analysis ---")
txn_nulls = null_analysis(transactions_df, "transactions")
display(txn_nulls.filter(F.col("null_pct") > 0).orderBy(F.col("null_pct").desc()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Statistical Summaries

# COMMAND ----------

def detailed_statistics(df: DataFrame, numeric_cols: list) -> DataFrame:
    """Compute extended statistics including percentiles for numeric columns."""
    if not numeric_cols:
        print("No numeric columns provided.")
        return None

    stats_rows = []
    for col_name in numeric_cols:
        col_stats = df.select(
            F.lit(col_name).alias("column"),
            F.count(col_name).alias("count"),
            F.mean(col_name).alias("mean"),
            F.stddev(col_name).alias("stddev"),
            F.min(col_name).alias("min_val"),
            F.expr(f"percentile_approx({col_name}, 0.25)").alias("p25"),
            F.expr(f"percentile_approx({col_name}, 0.50)").alias("median"),
            F.expr(f"percentile_approx({col_name}, 0.75)").alias("p75"),
            F.expr(f"percentile_approx({col_name}, 0.90)").alias("p90"),
            F.expr(f"percentile_approx({col_name}, 0.95)").alias("p95"),
            F.expr(f"percentile_approx({col_name}, 0.99)").alias("p99"),
            F.max(col_name).alias("max_val"),
            F.skewness(col_name).alias("skewness"),
            F.kurtosis(col_name).alias("kurtosis"),
        ).collect()[0]

        stats_rows.append(col_stats.asDict())

    return spark.createDataFrame(stats_rows)


# Identify numeric columns in events
events_numeric_cols = [
    f.name for f in events_df.schema.fields
    if isinstance(f.dataType, (IntegerType, LongType, DoubleType))
]
print(f"Numeric columns in events: {events_numeric_cols}")

events_stats = detailed_statistics(events_df, events_numeric_cols)
display(events_stats)

# COMMAND ----------

# Transactions statistics
txn_numeric_cols = [
    f.name for f in transactions_df.schema.fields
    if isinstance(f.dataType, (IntegerType, LongType, DoubleType))
]
print(f"Numeric columns in transactions: {txn_numeric_cols}")

txn_stats = detailed_statistics(transactions_df, txn_numeric_cols)
display(txn_stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Distribution Analysis

# COMMAND ----------

# Categorical column distributions
def categorical_distribution(df: DataFrame, col_name: str, top_n: int = 20) -> DataFrame:
    """Compute value counts and percentages for a categorical column."""
    total = df.count()
    dist = (
        df.groupBy(col_name)
        .agg(F.count("*").alias("count"))
        .withColumn("percentage", F.round((F.col("count") / F.lit(total)) * 100, 2))
        .orderBy(F.col("count").desc())
        .limit(top_n)
    )
    return dist


# Event type distribution
print("--- Event Type Distribution ---")
event_type_dist = categorical_distribution(events_df, "event_type")
display(event_type_dist)

# COMMAND ----------

# Numeric distribution — bucketed histograms
def numeric_histogram(df: DataFrame, col_name: str, num_buckets: int = 20) -> DataFrame:
    """Create a histogram by bucketing a numeric column."""
    col_min, col_max = df.select(
        F.min(col_name), F.max(col_name)
    ).collect()[0]

    if col_min is None or col_max is None:
        print(f"Column {col_name} has no data.")
        return None

    bucket_width = (col_max - col_min) / num_buckets if col_max != col_min else 1

    histogram = (
        df.withColumn(
            "bucket",
            F.floor((F.col(col_name) - F.lit(col_min)) / F.lit(bucket_width))
        )
        .withColumn(
            "bucket",
            F.when(F.col("bucket") >= num_buckets, num_buckets - 1).otherwise(F.col("bucket"))
        )
        .withColumn(
            "bucket_lower",
            F.round(F.lit(col_min) + F.col("bucket") * F.lit(bucket_width), 2)
        )
        .withColumn(
            "bucket_upper",
            F.round(F.lit(col_min) + (F.col("bucket") + 1) * F.lit(bucket_width), 2)
        )
        .groupBy("bucket", "bucket_lower", "bucket_upper")
        .agg(F.count("*").alias("count"))
        .orderBy("bucket")
    )
    return histogram


# Example: histogram of transaction amounts
if "amount" in transactions_df.columns:
    print("--- Transaction Amount Distribution ---")
    amount_hist = numeric_histogram(transactions_df, "amount", num_buckets=25)
    display(amount_hist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Correlation Analysis

# COMMAND ----------

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

def compute_correlation_matrix(df: DataFrame, numeric_cols: list, method: str = "pearson") -> DataFrame:
    """Compute a pairwise correlation matrix for numeric columns."""
    clean_df = df.select(numeric_cols).dropna()

    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
    vector_df = assembler.transform(clean_df).select("features")

    corr_matrix = Correlation.corr(vector_df, "features", method).collect()[0][0]
    corr_array = corr_matrix.toArray()

    rows = []
    for i, col_i in enumerate(numeric_cols):
        row = {"column": col_i}
        for j, col_j in enumerate(numeric_cols):
            row[col_j] = round(float(corr_array[i][j]), 4)
        rows.append(row)

    return spark.createDataFrame(rows)


if len(txn_numeric_cols) >= 2:
    print("--- Transaction Correlation Matrix (Pearson) ---")
    txn_corr = compute_correlation_matrix(transactions_df, txn_numeric_cols, "pearson")
    display(txn_corr)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Temporal Trends

# COMMAND ----------

# Daily event volume trends
daily_events = (
    events_df
    .withColumn("event_date", F.to_date("event_timestamp"))
    .groupBy("event_date")
    .agg(
        F.count("*").alias("event_count"),
        F.countDistinct("user_id").alias("unique_users"),
    )
    .orderBy("event_date")
)

# Add 7-day moving averages
window_7d = Window.orderBy("event_date").rowsBetween(-6, 0)
daily_events = (
    daily_events
    .withColumn("event_count_7d_avg", F.round(F.avg("event_count").over(window_7d), 2))
    .withColumn("unique_users_7d_avg", F.round(F.avg("unique_users").over(window_7d), 2))
)

print("--- Daily Event Volume with 7-Day Moving Average ---")
display(daily_events)

# COMMAND ----------

# Hourly patterns
hourly_pattern = (
    events_df
    .withColumn("hour_of_day", F.hour("event_timestamp"))
    .groupBy("hour_of_day")
    .agg(
        F.count("*").alias("event_count"),
        F.countDistinct("user_id").alias("unique_users"),
    )
    .orderBy("hour_of_day")
)

print("--- Hourly Event Pattern ---")
display(hourly_pattern)

# COMMAND ----------

# Day-of-week patterns
dow_pattern = (
    events_df
    .withColumn("day_of_week", F.dayofweek("event_timestamp"))
    .withColumn(
        "day_name",
        F.when(F.col("day_of_week") == 1, "Sunday")
        .when(F.col("day_of_week") == 2, "Monday")
        .when(F.col("day_of_week") == 3, "Tuesday")
        .when(F.col("day_of_week") == 4, "Wednesday")
        .when(F.col("day_of_week") == 5, "Thursday")
        .when(F.col("day_of_week") == 6, "Friday")
        .when(F.col("day_of_week") == 7, "Saturday")
    )
    .groupBy("day_of_week", "day_name")
    .agg(
        F.count("*").alias("event_count"),
        F.countDistinct("user_id").alias("unique_users"),
    )
    .orderBy("day_of_week")
)

print("--- Day-of-Week Pattern ---")
display(dow_pattern)

# COMMAND ----------

# Revenue trends (from transactions)
if "amount" in transactions_df.columns and "transaction_timestamp" in transactions_df.columns:
    daily_revenue = (
        transactions_df
        .withColumn("txn_date", F.to_date("transaction_timestamp"))
        .groupBy("txn_date")
        .agg(
            F.sum("amount").alias("daily_revenue"),
            F.count("*").alias("transaction_count"),
            F.avg("amount").alias("avg_transaction_value"),
            F.countDistinct("user_id").alias("paying_users"),
        )
        .orderBy("txn_date")
    )

    window_7d_rev = Window.orderBy("txn_date").rowsBetween(-6, 0)
    daily_revenue = daily_revenue.withColumn(
        "revenue_7d_avg", F.round(F.avg("daily_revenue").over(window_7d_rev), 2)
    )

    print("--- Daily Revenue Trend ---")
    display(daily_revenue)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Data Quality Assessment

# COMMAND ----------

def data_quality_report(df: DataFrame, table_name: str) -> DataFrame:
    """Generate a comprehensive data quality report."""
    total_rows = df.count()
    total_cols = len(df.columns)

    # Duplicate check
    distinct_rows = df.distinct().count()
    duplicate_rows = total_rows - distinct_rows
    duplicate_pct = round((duplicate_rows / total_rows) * 100, 2) if total_rows > 0 else 0.0

    # Completeness per column
    quality_metrics = []
    for col_name in df.columns:
        field = [f for f in df.schema.fields if f.name == col_name][0]

        null_count = df.filter(F.col(col_name).isNull()).count()
        completeness = round(((total_rows - null_count) / total_rows) * 100, 2) if total_rows > 0 else 0.0
        distinct_count = df.select(col_name).distinct().count()
        uniqueness = round((distinct_count / total_rows) * 100, 2) if total_rows > 0 else 0.0

        quality_metrics.append({
            "table": table_name,
            "column_name": col_name,
            "data_type": str(field.dataType),
            "total_rows": total_rows,
            "null_count": null_count,
            "completeness_pct": completeness,
            "distinct_count": distinct_count,
            "uniqueness_pct": uniqueness,
        })

    quality_df = spark.createDataFrame(quality_metrics)
    return quality_df


for name, df in tables.items():
    print(f"\n--- Data Quality Report: {name} ---")
    quality_report = data_quality_report(df, name)
    display(quality_report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Visualization-Ready Summaries

# COMMAND ----------

# Save EDA summary statistics to Gold table for downstream dashboards
all_nulls = events_nulls.unionByName(users_nulls).unionByName(txn_nulls)

(
    all_nulls
    .withColumn("eda_run_timestamp", F.current_timestamp())
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(EDA_OUTPUT_TABLE)
)

print(f"EDA summaries written to {EDA_OUTPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Key Findings Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Total Events | `events_df.count()` |
# MAGIC | Total Users | `users_df.count()` |
# MAGIC | Total Transactions | `transactions_df.count()` |
# MAGIC | Columns with >5% Nulls | See null analysis above |
# MAGIC | Duplicate Rate | See quality report above |
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Address columns with high null rates before feature engineering
# MAGIC - Investigate outliers identified in percentile analysis
# MAGIC - Use temporal patterns to inform time-based feature creation
# MAGIC - Feed correlation findings into feature selection
