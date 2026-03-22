"""
Silver Layer — Transform & Clean

Read from Bronze Delta tables, apply data quality checks, type casting,
deduplication, PII masking, and SCD Type 2 merge logic. Write validated
records to Silver Delta tables with OPTIMIZE and ZORDER.

Usage:
    spark-submit delta-lake/silver/transform_clean.py --config delta-lake/pipeline_config.yml
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    DoubleType, IntegerType, LongType, TimestampType, BooleanType, StringType,
)
import logging
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("silver.transform")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATALOG = "analytics_platform"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"


# ---------------------------------------------------------------------------
# Data Quality Checks
# ---------------------------------------------------------------------------

class DataQualityChecker:
    """Run data quality validations and collect metrics."""

    def __init__(self, spark: SparkSession, table_name: str):
        self.spark = spark
        self.table_name = table_name
        self.metrics: List[Dict] = []

    def check_nulls(self, df: DataFrame, columns: List[str], max_null_pct: float = 5.0) -> DataFrame:
        """Validate that null percentage is within threshold for specified columns."""
        total = df.count()
        if total == 0:
            logger.warning("Empty DataFrame — skipping null checks.")
            return df

        failed_cols = []
        for col_name in columns:
            null_count = df.filter(F.col(col_name).isNull()).count()
            null_pct = (null_count / total) * 100

            passed = null_pct <= max_null_pct
            self.metrics.append({
                "table": self.table_name,
                "check_type": "null_check",
                "column": col_name,
                "null_count": null_count,
                "null_pct": round(null_pct, 2),
                "threshold": max_null_pct,
                "passed": passed,
                "timestamp": datetime.now().isoformat(),
            })

            if not passed:
                failed_cols.append(col_name)
                logger.warning(
                    "NULL CHECK FAILED: %s.%s — %.2f%% nulls (threshold: %.2f%%)",
                    self.table_name, col_name, null_pct, max_null_pct,
                )

        if failed_cols:
            logger.warning("Columns failing null check: %s", failed_cols)

        return df

    def check_range(
        self, df: DataFrame, col_name: str, min_val: float, max_val: float,
    ) -> DataFrame:
        """Validate that values fall within an expected range."""
        out_of_range = df.filter(
            (F.col(col_name) < min_val) | (F.col(col_name) > max_val)
        ).count()

        total = df.filter(F.col(col_name).isNotNull()).count()
        oor_pct = (out_of_range / total * 100) if total > 0 else 0

        passed = out_of_range == 0
        self.metrics.append({
            "table": self.table_name,
            "check_type": "range_check",
            "column": col_name,
            "out_of_range_count": out_of_range,
            "out_of_range_pct": round(oor_pct, 2),
            "min_allowed": min_val,
            "max_allowed": max_val,
            "passed": passed,
            "timestamp": datetime.now().isoformat(),
        })

        if not passed:
            logger.warning(
                "RANGE CHECK FAILED: %s.%s — %d values outside [%.2f, %.2f]",
                self.table_name, col_name, out_of_range, min_val, max_val,
            )

        return df

    def check_referential_integrity(
        self, df: DataFrame, fk_col: str, reference_df: DataFrame, pk_col: str,
    ) -> DataFrame:
        """Validate that foreign key values exist in the reference table."""
        orphans = df.join(reference_df, df[fk_col] == reference_df[pk_col], "left_anti")
        orphan_count = orphans.count()
        total = df.count()

        passed = orphan_count == 0
        self.metrics.append({
            "table": self.table_name,
            "check_type": "referential_integrity",
            "column": fk_col,
            "orphan_count": orphan_count,
            "total_rows": total,
            "passed": passed,
            "timestamp": datetime.now().isoformat(),
        })

        if not passed:
            logger.warning(
                "REFERENTIAL INTEGRITY FAILED: %s.%s — %d orphan records",
                self.table_name, fk_col, orphan_count,
            )

        return df

    def check_schema(self, df: DataFrame, expected_columns: List[str]) -> DataFrame:
        """Validate that all expected columns exist."""
        actual_cols = set(df.columns)
        missing = [c for c in expected_columns if c not in actual_cols]

        passed = len(missing) == 0
        self.metrics.append({
            "table": self.table_name,
            "check_type": "schema_check",
            "column": "N/A",
            "missing_columns": missing,
            "passed": passed,
            "timestamp": datetime.now().isoformat(),
        })

        if not passed:
            logger.error("SCHEMA CHECK FAILED: missing columns %s", missing)

        return df

    def get_metrics_df(self) -> DataFrame:
        """Return all quality metrics as a DataFrame."""
        if not self.metrics:
            return None
        return self.spark.createDataFrame(
            [{k: str(v) for k, v in m.items()} for m in self.metrics]
        )


# ---------------------------------------------------------------------------
# Transformation functions
# ---------------------------------------------------------------------------

def cast_columns(df: DataFrame, type_map: Dict[str, str]) -> DataFrame:
    """Cast columns to specified types."""
    for col_name, target_type in type_map.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(target_type))
    return df


def standardize_strings(df: DataFrame, columns: List[str]) -> DataFrame:
    """Trim whitespace and lowercase string columns."""
    for col_name in columns:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.lower(F.trim(F.col(col_name))))
    return df


def deduplicate_records(df: DataFrame, primary_key: str, order_col: str) -> DataFrame:
    """Keep only the latest record per primary key using a window function."""
    window = Window.partitionBy(primary_key).orderBy(F.col(order_col).desc())
    return (
        df
        .withColumn("_dedup_rank", F.row_number().over(window))
        .filter(F.col("_dedup_rank") == 1)
        .drop("_dedup_rank")
    )


def mask_pii(df: DataFrame, pii_columns: List[str]) -> DataFrame:
    """Hash PII columns using SHA-256 for pseudonymization."""
    for col_name in pii_columns:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.sha2(F.col(col_name).cast("string"), 256))
    return df


def apply_scd_type2_merge(
    spark: SparkSession,
    target_table: str,
    source_df: DataFrame,
    primary_key: str,
    change_columns: List[str],
) -> None:
    """
    Apply SCD Type 2 merge logic: close existing records and insert new versions
    when changes are detected.
    """
    source_df.createOrReplaceTempView("source_updates")

    # Build the change detection condition
    change_condition = " OR ".join(
        [f"target.{c} <> source.{c}" for c in change_columns]
    )

    merge_sql = f"""
        MERGE INTO {target_table} AS target
        USING (
            SELECT *, current_timestamp() AS _effective_from
            FROM source_updates
        ) AS source
        ON target.{primary_key} = source.{primary_key}
           AND target._is_current = true

        -- Close existing record when data has changed
        WHEN MATCHED AND ({change_condition}) THEN UPDATE SET
            target._is_current = false,
            target._effective_to = source._effective_from,
            target._updated_at = current_timestamp()

        -- Insert new or changed records
        WHEN NOT MATCHED THEN INSERT (
            {', '.join(source_df.columns)},
            _effective_from,
            _effective_to,
            _is_current,
            _updated_at
        ) VALUES (
            {', '.join(['source.' + c for c in source_df.columns])},
            source._effective_from,
            CAST(NULL AS TIMESTAMP),
            true,
            current_timestamp()
        )
    """

    spark.sql(merge_sql)
    logger.info("SCD Type 2 merge applied to %s", target_table)

    # Insert new versions for changed records
    insert_sql = f"""
        INSERT INTO {target_table}
        SELECT
            source.*,
            current_timestamp() AS _effective_from,
            CAST(NULL AS TIMESTAMP) AS _effective_to,
            true AS _is_current,
            current_timestamp() AS _updated_at
        FROM source_updates source
        INNER JOIN {target_table} target
            ON source.{primary_key} = target.{primary_key}
            AND target._is_current = false
            AND target._updated_at = (
                SELECT MAX(_updated_at)
                FROM {target_table}
                WHERE {primary_key} = source.{primary_key}
            )
        WHERE NOT EXISTS (
            SELECT 1 FROM {target_table} t2
            WHERE t2.{primary_key} = source.{primary_key}
              AND t2._is_current = true
        )
    """
    spark.sql(insert_sql)
    logger.info("New SCD2 versions inserted for changed records in %s", target_table)


# ---------------------------------------------------------------------------
# Silver transformations per table
# ---------------------------------------------------------------------------

class SilverTransformations:
    """Transform Bronze data into clean Silver tables."""

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def transform_events(self) -> Tuple[DataFrame, DataQualityChecker]:
        """Transform events from Bronze to Silver."""
        source_table = f"{CATALOG}.{BRONZE_SCHEMA}.events_raw"
        target_table = f"{CATALOG}.{SILVER_SCHEMA}.events"

        logger.info("Transforming events: %s -> %s", source_table, target_table)
        df = self.spark.read.table(source_table)

        # Quality checks
        qc = DataQualityChecker(self.spark, "events")
        qc.check_schema(df, ["event_id", "user_id", "event_type", "event_timestamp"])
        qc.check_nulls(df, ["event_id", "user_id", "event_timestamp"], max_null_pct=1.0)

        # Type casting
        df = cast_columns(df, {
            "event_timestamp": "timestamp",
        })

        # Standardize strings
        df = standardize_strings(df, ["event_type", "device_type", "os", "browser", "country"])

        # Deduplicate
        df = deduplicate_records(df, "event_id", "_ingestion_timestamp")

        # Mask PII
        df = mask_pii(df, ["ip_address"])

        # Filter out records with null primary keys
        df = df.filter(F.col("event_id").isNotNull() & F.col("user_id").isNotNull())

        # Add processing metadata
        df = (
            df
            .withColumn("_processed_at", F.current_timestamp())
            .withColumn("_silver_version", F.lit(1))
        )

        # Drop Bronze-only audit columns
        drop_cols = ["_rescued_data", "_batch_id", "_source_name", "_ingestion_date",
                     "year", "month", "day"]
        df = df.drop(*[c for c in drop_cols if c in df.columns])

        # Range check on timestamp
        qc.check_range(
            df.withColumn("_ts_epoch", F.unix_timestamp("event_timestamp")),
            "_ts_epoch",
            min_val=1_577_836_800,  # 2020-01-01
            max_val=2_000_000_000,  # ~2033
        )

        # Write to Silver
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(target_table)
        logger.info("Events Silver table written: %d rows", df.count())

        return df, qc

    def transform_transactions(self, users_df: Optional[DataFrame] = None) -> Tuple[DataFrame, DataQualityChecker]:
        """Transform transactions from Bronze to Silver."""
        source_table = f"{CATALOG}.{BRONZE_SCHEMA}.transactions_raw"
        target_table = f"{CATALOG}.{SILVER_SCHEMA}.transactions"

        logger.info("Transforming transactions: %s -> %s", source_table, target_table)
        df = self.spark.read.table(source_table)

        qc = DataQualityChecker(self.spark, "transactions")
        qc.check_schema(df, ["transaction_id", "user_id", "amount", "transaction_timestamp"])
        qc.check_nulls(df, ["transaction_id", "user_id", "amount"], max_null_pct=1.0)

        # Type casting
        df = cast_columns(df, {
            "amount": "double",
            "quantity": "integer",
            "transaction_timestamp": "timestamp",
        })

        # Range validation
        qc.check_range(df, "amount", min_val=0.0, max_val=1_000_000.0)
        qc.check_range(df, "quantity", min_val=1, max_val=10_000)

        # Referential integrity with users
        if users_df is not None:
            qc.check_referential_integrity(df, "user_id", users_df, "user_id")

        # Standardize
        df = standardize_strings(df, ["transaction_status", "payment_method", "product_category", "currency"])

        # Deduplicate
        df = deduplicate_records(df, "transaction_id", "_ingestion_timestamp")

        # Filter nulls in primary key
        df = df.filter(F.col("transaction_id").isNotNull() & F.col("user_id").isNotNull())

        # Remove clearly invalid records
        df = df.filter(F.col("amount") >= 0)

        df = (
            df
            .withColumn("_processed_at", F.current_timestamp())
            .withColumn("_silver_version", F.lit(1))
        )

        drop_cols = ["_rescued_data", "_batch_id", "_source_name", "_ingestion_date",
                     "year", "month"]
        df = df.drop(*[c for c in drop_cols if c in df.columns])

        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(target_table)
        logger.info("Transactions Silver table written: %d rows", df.count())

        return df, qc

    def transform_users(self) -> Tuple[DataFrame, DataQualityChecker]:
        """Transform users from Bronze to Silver with SCD Type 2."""
        source_table = f"{CATALOG}.{BRONZE_SCHEMA}.users_raw"
        target_table = f"{CATALOG}.{SILVER_SCHEMA}.users"

        logger.info("Transforming users: %s -> %s", source_table, target_table)
        df = self.spark.read.table(source_table)

        qc = DataQualityChecker(self.spark, "users")
        qc.check_schema(df, ["user_id", "email", "signup_date"])
        qc.check_nulls(df, ["user_id", "email"], max_null_pct=0.5)

        # Type casting
        df = cast_columns(df, {
            "signup_date": "date",
            "is_active": "boolean",
        })

        # Standardize
        df = standardize_strings(df, ["user_segment", "country", "platform", "age_group"])

        # Deduplicate
        df = deduplicate_records(df, "user_id", "_ingestion_timestamp")

        # Mask PII
        df = mask_pii(df, ["email", "full_name"])

        df = df.filter(F.col("user_id").isNotNull())

        df = (
            df
            .withColumn("_processed_at", F.current_timestamp())
            .withColumn("_silver_version", F.lit(1))
        )

        drop_cols = ["_rescued_data", "_batch_id", "_source_name", "_ingestion_date"]
        df = df.drop(*[c for c in drop_cols if c in df.columns])

        # Check if target table exists for SCD Type 2
        try:
            self.spark.read.table(target_table)
            table_exists = True
        except Exception:
            table_exists = False

        if table_exists:
            apply_scd_type2_merge(
                spark=self.spark,
                target_table=target_table,
                source_df=df,
                primary_key="user_id",
                change_columns=["user_segment", "country", "platform", "is_active"],
            )
        else:
            scd_df = (
                df
                .withColumn("_effective_from", F.current_timestamp())
                .withColumn("_effective_to", F.lit(None).cast(TimestampType()))
                .withColumn("_is_current", F.lit(True))
                .withColumn("_updated_at", F.current_timestamp())
            )
            scd_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(target_table)

        logger.info("Users Silver table written.")
        return df, qc


def optimize_silver_tables(spark: SparkSession) -> None:
    """Run OPTIMIZE and ZORDER on Silver tables for query performance."""
    optimize_configs = {
        f"{CATALOG}.{SILVER_SCHEMA}.events": ["event_timestamp", "user_id"],
        f"{CATALOG}.{SILVER_SCHEMA}.transactions": ["transaction_timestamp", "user_id"],
        f"{CATALOG}.{SILVER_SCHEMA}.users": ["user_id"],
    }

    for table, zorder_cols in optimize_configs.items():
        try:
            zorder_clause = ", ".join(zorder_cols)
            spark.sql(f"OPTIMIZE {table} ZORDER BY ({zorder_clause})")
            logger.info("OPTIMIZE + ZORDER completed for %s on (%s)", table, zorder_clause)
        except Exception as exc:
            logger.error("Failed to optimize %s: %s", table, exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run Silver transformations for all tables."""
    spark = SparkSession.builder.appName("SilverTransform").getOrCreate()
    spark.sql(f"USE CATALOG {CATALOG}")

    transformer = SilverTransformations(spark)

    # Transform users first (needed for referential integrity checks)
    logger.info("=" * 60)
    logger.info("Transforming users")
    logger.info("=" * 60)
    users_df, users_qc = transformer.transform_users()

    # Transform events
    logger.info("=" * 60)
    logger.info("Transforming events")
    logger.info("=" * 60)
    events_df, events_qc = transformer.transform_events()

    # Transform transactions (with referential integrity to users)
    logger.info("=" * 60)
    logger.info("Transforming transactions")
    logger.info("=" * 60)
    txn_df, txn_qc = transformer.transform_transactions(users_df=users_df)

    # Log quality metrics
    quality_metrics_table = f"{CATALOG}.{SILVER_SCHEMA}.data_quality_metrics"
    for qc in [users_qc, events_qc, txn_qc]:
        metrics_df = qc.get_metrics_df()
        if metrics_df:
            metrics_df.write.format("delta").mode("append").saveAsTable(quality_metrics_table)

    logger.info("Data quality metrics written to %s", quality_metrics_table)

    # Optimize tables
    logger.info("=" * 60)
    logger.info("Optimizing Silver tables")
    logger.info("=" * 60)
    optimize_silver_tables(spark)

    logger.info("Silver transformation pipeline complete.")


if __name__ == "__main__":
    main()
