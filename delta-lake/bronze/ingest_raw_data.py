"""
Bronze Layer — Raw Data Ingestion

Ingest raw data into Delta Lake Bronze tables using Databricks Auto Loader
(cloudFiles) for streaming ingestion. Supports JSON, CSV, and Parquet sources
with schema evolution, bad-record quarantine, deduplication, and exactly-once
checkpointing.

Usage:
    spark-submit delta-lake/bronze/ingest_raw_data.py --config delta-lake/pipeline_config.yml
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    TimestampType,
    LongType,
    IntegerType,
)
from pyspark.sql.window import Window
import logging
import uuid
import yaml
import sys
import os
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bronze.ingest")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATALOG = "analytics_platform"
BRONZE_SCHEMA = "bronze"
CHECKPOINT_BASE = "/mnt/checkpoints/bronze"

# Expected schemas per source
EVENTS_SCHEMA = StructType([
    StructField("event_id", StringType(), nullable=False),
    StructField("user_id", StringType(), nullable=False),
    StructField("event_type", StringType(), nullable=True),
    StructField("event_timestamp", StringType(), nullable=True),
    StructField("session_id", StringType(), nullable=True),
    StructField("page_url", StringType(), nullable=True),
    StructField("referrer", StringType(), nullable=True),
    StructField("device_type", StringType(), nullable=True),
    StructField("os", StringType(), nullable=True),
    StructField("browser", StringType(), nullable=True),
    StructField("ip_address", StringType(), nullable=True),
    StructField("country", StringType(), nullable=True),
    StructField("city", StringType(), nullable=True),
    StructField("properties", StringType(), nullable=True),
])

TRANSACTIONS_SCHEMA = StructType([
    StructField("transaction_id", StringType(), nullable=False),
    StructField("user_id", StringType(), nullable=False),
    StructField("amount", DoubleType(), nullable=True),
    StructField("currency", StringType(), nullable=True),
    StructField("transaction_timestamp", StringType(), nullable=True),
    StructField("transaction_status", StringType(), nullable=True),
    StructField("payment_method", StringType(), nullable=True),
    StructField("product_id", StringType(), nullable=True),
    StructField("product_category", StringType(), nullable=True),
    StructField("quantity", IntegerType(), nullable=True),
])

USERS_SCHEMA = StructType([
    StructField("user_id", StringType(), nullable=False),
    StructField("email", StringType(), nullable=True),
    StructField("full_name", StringType(), nullable=True),
    StructField("signup_date", StringType(), nullable=True),
    StructField("user_segment", StringType(), nullable=True),
    StructField("country", StringType(), nullable=True),
    StructField("platform", StringType(), nullable=True),
    StructField("age_group", StringType(), nullable=True),
    StructField("is_active", StringType(), nullable=True),
])

SOURCE_CONFIGS = {
    "events": {
        "schema": EVENTS_SCHEMA,
        "primary_key": "event_id",
        "partition_cols": ["year", "month", "day"],
        "format": "json",
    },
    "transactions": {
        "schema": TRANSACTIONS_SCHEMA,
        "primary_key": "transaction_id",
        "partition_cols": ["year", "month"],
        "format": "json",
    },
    "users": {
        "schema": USERS_SCHEMA,
        "primary_key": "user_id",
        "partition_cols": [],
        "format": "csv",
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_pipeline_config(config_path: str) -> dict:
    """Load pipeline configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Pipeline config loaded from %s", config_path)
        return config
    except FileNotFoundError:
        logger.warning("Config file not found at %s. Using defaults.", config_path)
        return {}
    except yaml.YAMLError as e:
        logger.error("Failed to parse config file: %s", e)
        raise


def add_audit_columns(df: DataFrame, source_name: str) -> DataFrame:
    """Add audit / lineage columns to ingested data."""
    batch_id = str(uuid.uuid4())
    return (
        df
        .withColumn("_ingestion_timestamp", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
        .withColumn("_batch_id", F.lit(batch_id))
        .withColumn("_source_name", F.lit(source_name))
        .withColumn("_ingestion_date", F.current_date())
    )


def add_partition_columns(df: DataFrame, timestamp_col: str) -> DataFrame:
    """Add year/month/day partition columns derived from a timestamp."""
    return (
        df
        .withColumn("year", F.year(F.to_timestamp(F.col(timestamp_col))))
        .withColumn("month", F.month(F.to_timestamp(F.col(timestamp_col))))
        .withColumn("day", F.dayofmonth(F.to_timestamp(F.col(timestamp_col))))
    )


def deduplicate(df: DataFrame, primary_key: str, order_col: str = "_ingestion_timestamp") -> DataFrame:
    """Remove duplicates within a micro-batch, keeping the latest record."""
    window = Window.partitionBy(primary_key).orderBy(F.col(order_col).desc())
    return (
        df
        .withColumn("_row_num", F.row_number().over(window))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )


# ---------------------------------------------------------------------------
# Auto Loader ingestion
# ---------------------------------------------------------------------------

class BronzeIngestion:
    """Manages streaming ingestion of raw data into Bronze Delta tables."""

    def __init__(self, spark: SparkSession, catalog: str, schema: str, checkpoint_base: str):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.checkpoint_base = checkpoint_base

    def _target_table(self, source_name: str) -> str:
        return f"{self.catalog}.{self.schema}.{source_name}_raw"

    def _checkpoint_path(self, source_name: str) -> str:
        return f"{self.checkpoint_base}/{source_name}"

    def ingest_with_autoloader(
        self,
        source_path: str,
        source_name: str,
        source_format: str,
        source_schema: StructType,
        primary_key: str,
        partition_cols: list,
        timestamp_col: Optional[str] = None,
    ) -> None:
        """
        Start a streaming Auto Loader job to ingest data from cloud storage
        into a Bronze Delta table.
        """
        target_table = self._target_table(source_name)
        checkpoint = self._checkpoint_path(source_name)

        logger.info(
            "Starting Auto Loader ingestion: %s -> %s (format=%s)",
            source_path, target_table, source_format,
        )

        reader_options = {
            "cloudFiles.format": source_format,
            "cloudFiles.schemaLocation": f"{checkpoint}/schema",
            "cloudFiles.inferColumnTypes": "true",
            "cloudFiles.schemaEvolutionMode": "addNewColumns",
            "cloudFiles.schemaHints": "",
            "rescuedDataColumn": "_rescued_data",
        }

        if source_format == "csv":
            reader_options["header"] = "true"
            reader_options["multiLine"] = "true"
            reader_options["escape"] = '"'

        if source_format == "json":
            reader_options["multiLine"] = "true"

        try:
            raw_stream = (
                self.spark.readStream
                .format("cloudFiles")
                .options(**reader_options)
                .schema(source_schema)
                .load(source_path)
            )

            # Apply transformations
            def process_micro_batch(batch_df: DataFrame, batch_id: int) -> None:
                """Process each micro-batch: audit columns, dedup, write."""
                if batch_df.isEmpty():
                    logger.info("Batch %d is empty for %s — skipping.", batch_id, source_name)
                    return

                logger.info(
                    "Processing batch %d for %s: %d rows",
                    batch_id, source_name, batch_df.count(),
                )

                # Add audit columns
                enriched = add_audit_columns(batch_df, source_name)

                # Add partition columns if a timestamp column exists
                if timestamp_col and timestamp_col in enriched.columns:
                    enriched = add_partition_columns(enriched, timestamp_col)

                # Deduplicate within the micro-batch
                deduped = deduplicate(enriched, primary_key)

                # Write to Delta table
                writer = (
                    deduped.write
                    .format("delta")
                    .mode("append")
                    .option("mergeSchema", "true")
                )

                if partition_cols:
                    existing_partition_cols = [c for c in partition_cols if c in deduped.columns]
                    if existing_partition_cols:
                        writer = writer.partitionBy(*existing_partition_cols)

                writer.saveAsTable(target_table)

                logger.info(
                    "Batch %d for %s: wrote %d rows to %s",
                    batch_id, source_name, deduped.count(), target_table,
                )

            query = (
                raw_stream
                .writeStream
                .foreachBatch(process_micro_batch)
                .option("checkpointLocation", checkpoint)
                .trigger(availableNow=True)
                .queryName(f"bronze_ingest_{source_name}")
                .start()
            )

            query.awaitTermination()
            logger.info("Auto Loader ingestion completed for %s", source_name)

        except Exception as exc:
            logger.error("Failed to ingest %s: %s", source_name, exc, exc_info=True)
            raise

    def ingest_batch(
        self,
        source_path: str,
        source_name: str,
        source_format: str,
        source_schema: StructType,
        primary_key: str,
        partition_cols: list,
        timestamp_col: Optional[str] = None,
    ) -> None:
        """
        Batch ingestion fallback for sources that do not support streaming.
        """
        target_table = self._target_table(source_name)
        logger.info("Starting batch ingestion: %s -> %s", source_path, target_table)

        try:
            reader = self.spark.read.format(source_format).schema(source_schema)

            if source_format == "csv":
                reader = reader.option("header", "true")
            if source_format == "json":
                reader = reader.option("multiLine", "true")

            raw_df = reader.load(source_path)

            enriched = add_audit_columns(raw_df, source_name)

            if timestamp_col and timestamp_col in enriched.columns:
                enriched = add_partition_columns(enriched, timestamp_col)

            deduped = deduplicate(enriched, primary_key)

            writer = (
                deduped.write
                .format("delta")
                .mode("append")
                .option("mergeSchema", "true")
            )

            if partition_cols:
                existing_cols = [c for c in partition_cols if c in deduped.columns]
                if existing_cols:
                    writer = writer.partitionBy(*existing_cols)

            writer.saveAsTable(target_table)
            logger.info("Batch ingestion complete: %d rows -> %s", deduped.count(), target_table)

        except Exception as exc:
            logger.error("Batch ingestion failed for %s: %s", source_name, exc, exc_info=True)
            raise

    def quarantine_bad_records(self, source_name: str) -> None:
        """Move rescued (unparseable) records to a quarantine table."""
        target_table = self._target_table(source_name)
        quarantine_table = f"{self.catalog}.{self.schema}.{source_name}_quarantine"

        bad_records = (
            self.spark.read.table(target_table)
            .filter(F.col("_rescued_data").isNotNull())
        )

        count = bad_records.count()
        if count > 0:
            logger.warning("Found %d bad records in %s. Moving to quarantine.", count, target_table)
            bad_records.write.format("delta").mode("append").saveAsTable(quarantine_table)

            # Remove bad records from main table
            self.spark.sql(f"""
                DELETE FROM {target_table}
                WHERE _rescued_data IS NOT NULL
            """)
            logger.info("Quarantined %d records to %s", count, quarantine_table)
        else:
            logger.info("No bad records found in %s", target_table)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Run Bronze ingestion for all configured sources."""
    spark = SparkSession.builder.appName("BronzeIngestion").getOrCreate()
    spark.sql(f"USE CATALOG {CATALOG}")

    # Load pipeline config if provided
    config_path = "delta-lake/pipeline_config.yml"
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--config" and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]

    pipeline_config = load_pipeline_config(config_path)
    source_base_path = pipeline_config.get("sources", {}).get("base_path", "/mnt/raw-data")

    ingestion = BronzeIngestion(
        spark=spark,
        catalog=CATALOG,
        schema=BRONZE_SCHEMA,
        checkpoint_base=CHECKPOINT_BASE,
    )

    # Ingest each source
    for source_name, config in SOURCE_CONFIGS.items():
        source_path = f"{source_base_path}/{source_name}"
        timestamp_col = "event_timestamp" if source_name == "events" else (
            "transaction_timestamp" if source_name == "transactions" else None
        )

        logger.info("="*60)
        logger.info("Ingesting source: %s", source_name)
        logger.info("="*60)

        try:
            ingestion.ingest_with_autoloader(
                source_path=source_path,
                source_name=source_name,
                source_format=config["format"],
                source_schema=config["schema"],
                primary_key=config["primary_key"],
                partition_cols=config["partition_cols"],
                timestamp_col=timestamp_col,
            )

            # Quarantine bad records
            ingestion.quarantine_bad_records(source_name)

        except Exception as exc:
            logger.error("Ingestion failed for %s: %s. Trying batch fallback.", source_name, exc)
            try:
                ingestion.ingest_batch(
                    source_path=source_path,
                    source_name=source_name,
                    source_format=config["format"],
                    source_schema=config["schema"],
                    primary_key=config["primary_key"],
                    partition_cols=config["partition_cols"],
                    timestamp_col=timestamp_col,
                )
            except Exception as batch_exc:
                logger.error("Batch fallback also failed for %s: %s", source_name, batch_exc)

    logger.info("Bronze ingestion pipeline complete.")


if __name__ == "__main__":
    main()
