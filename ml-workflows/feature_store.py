"""
Databricks Feature Store Management

Manages feature table creation, feature lookups, point-in-time lookups,
freshness monitoring, and online store publishing.

Usage:
    python ml-workflows/feature_store.py --action create
    python ml-workflows/feature_store.py --action monitor
    python ml-workflows/feature_store.py --action publish-online
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
from databricks.feature_engineering import (
    FeatureEngineeringClient,
    FeatureFunction,
    FeatureLookup,
)
import logging
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("feature_store")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATALOG = "analytics_platform"
GOLD_SCHEMA = "gold"
FEATURE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.feature_table"
USER_FEATURE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.user_features"
TXN_FEATURE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.transaction_features"


# ---------------------------------------------------------------------------
# Feature Store Manager
# ---------------------------------------------------------------------------

class FeatureStoreManager:
    """Manages Databricks Feature Store operations."""

    def __init__(self, spark: SparkSession, catalog: str = CATALOG):
        self.spark = spark
        self.catalog = catalog
        self.fe_client = FeatureEngineeringClient()

    # ----- Feature Table Creation -----

    def create_user_feature_table(self, df: DataFrame) -> None:
        """Create a user-level feature table with daily granularity."""
        table_name = USER_FEATURE_TABLE

        self.fe_client.create_table(
            name=table_name,
            primary_keys=["user_id", "event_date"],
            timestamp_keys=["event_date"],
            df=df,
            description=(
                "User-level behavioural and transactional features aggregated daily. "
                "Includes rolling window aggregations (7d, 30d, 90d), time-based features, "
                "and encoded categorical features."
            ),
            tags={
                "team": "data-science",
                "project": "analytics-platform",
                "domain": "user-behaviour",
                "refresh_frequency": "daily",
            },
        )

        logger.info("Feature table created: %s (%d rows)", table_name, df.count())

    def create_transaction_feature_table(self, df: DataFrame) -> None:
        """Create a transaction-level feature table."""
        table_name = TXN_FEATURE_TABLE

        self.fe_client.create_table(
            name=table_name,
            primary_keys=["user_id"],
            df=df,
            description=(
                "User-level transaction aggregate features. Includes total spend, "
                "average order value, purchase frequency, and refund rate."
            ),
            tags={
                "team": "data-science",
                "project": "analytics-platform",
                "domain": "transactions",
                "refresh_frequency": "daily",
            },
        )

        logger.info("Feature table created: %s (%d rows)", table_name, df.count())

    def update_feature_table(self, table_name: str, df: DataFrame, mode: str = "merge") -> None:
        """Update an existing feature table with new data."""
        self.fe_client.write_table(
            name=table_name,
            df=df,
            mode=mode,  # "merge" or "overwrite"
        )
        logger.info("Feature table updated: %s (mode=%s, rows=%d)", table_name, mode, df.count())

    # ----- Feature Lookups -----

    def create_feature_lookups(
        self,
        entity_df: DataFrame,
        feature_tables: Optional[List[str]] = None,
    ) -> DataFrame:
        """
        Perform feature lookups by joining entity keys with feature tables.
        Returns a DataFrame enriched with features.
        """
        if feature_tables is None:
            feature_tables = [USER_FEATURE_TABLE, TXN_FEATURE_TABLE]

        lookups = []
        for table in feature_tables:
            # Determine lookup keys based on table
            if "user_features" in table:
                lookup = FeatureLookup(
                    table_name=table,
                    lookup_key=["user_id", "event_date"],
                )
            else:
                lookup = FeatureLookup(
                    table_name=table,
                    lookup_key=["user_id"],
                )
            lookups.append(lookup)

        training_set = self.fe_client.create_training_set(
            df=entity_df,
            feature_lookups=lookups,
            label="label" if "label" in entity_df.columns else None,
            exclude_columns=["user_id", "event_date"],
        )

        result_df = training_set.load_df()
        logger.info(
            "Feature lookups complete: %d rows, %d columns (from %d tables)",
            result_df.count(), len(result_df.columns), len(feature_tables),
        )

        return result_df

    # ----- Point-in-Time Lookups -----

    def point_in_time_lookup(
        self,
        entity_df: DataFrame,
        feature_table: str,
        entity_key: str = "user_id",
        timestamp_key: str = "event_date",
    ) -> DataFrame:
        """
        Perform a point-in-time correct feature lookup.
        For each entity row, returns features as of the entity's timestamp.
        This prevents data leakage by ensuring no future data is used.
        """
        lookup = FeatureLookup(
            table_name=feature_table,
            lookup_key=[entity_key],
            timestamp_lookup_key=[timestamp_key],
        )

        training_set = self.fe_client.create_training_set(
            df=entity_df,
            feature_lookups=[lookup],
            label="label" if "label" in entity_df.columns else None,
        )

        result_df = training_set.load_df()
        logger.info(
            "Point-in-time lookup complete: %d rows from %s",
            result_df.count(), feature_table,
        )

        return result_df

    # ----- Feature Freshness Monitoring -----

    def monitor_freshness(
        self,
        table_name: str,
        timestamp_col: str = "event_date",
        max_stale_hours: int = 48,
    ) -> Dict:
        """
        Monitor feature table freshness by checking the latest timestamp.
        Returns freshness metrics and alerts if stale.
        """
        df = self.spark.read.table(table_name)

        latest_row = (
            df
            .agg(
                F.max(timestamp_col).alias("latest_timestamp"),
                F.min(timestamp_col).alias("earliest_timestamp"),
                F.count("*").alias("total_rows"),
                F.countDistinct("user_id").alias("unique_users"),
            )
            .collect()[0]
        )

        latest_ts = latest_row["latest_timestamp"]
        now = datetime.now()
        staleness_hours = (now - datetime.combine(latest_ts, datetime.min.time())).total_seconds() / 3600 if latest_ts else float("inf")

        is_fresh = staleness_hours <= max_stale_hours

        freshness_report = {
            "table": table_name,
            "latest_timestamp": str(latest_ts),
            "earliest_timestamp": str(latest_row["earliest_timestamp"]),
            "total_rows": latest_row["total_rows"],
            "unique_users": latest_row["unique_users"],
            "staleness_hours": round(staleness_hours, 1),
            "max_stale_hours": max_stale_hours,
            "is_fresh": is_fresh,
            "checked_at": now.isoformat(),
        }

        if not is_fresh:
            logger.warning(
                "STALE FEATURE TABLE: %s is %.1f hours old (threshold: %d hours)",
                table_name, staleness_hours, max_stale_hours,
            )
        else:
            logger.info(
                "Feature table %s is fresh (%.1f hours old, threshold: %d)",
                table_name, staleness_hours, max_stale_hours,
            )

        return freshness_report

    def monitor_all_tables(self) -> List[Dict]:
        """Monitor freshness of all feature tables."""
        tables = [
            (USER_FEATURE_TABLE, "event_date", 48),
            (TXN_FEATURE_TABLE, None, 48),
        ]

        reports = []
        for table_name, ts_col, max_hours in tables:
            try:
                if ts_col:
                    report = self.monitor_freshness(table_name, ts_col, max_hours)
                else:
                    # For tables without a timestamp column, check row count trends
                    report = {
                        "table": table_name,
                        "total_rows": self.spark.read.table(table_name).count(),
                        "checked_at": datetime.now().isoformat(),
                    }
                reports.append(report)
            except Exception as exc:
                logger.error("Failed to monitor %s: %s", table_name, exc)
                reports.append({"table": table_name, "error": str(exc)})

        return reports

    # ----- Online Store Publishing -----

    def publish_to_online_store(
        self,
        table_name: str,
        online_store_spec: Optional[Dict] = None,
    ) -> None:
        """
        Publish a feature table to an online store for real-time serving.
        Uses Databricks Online Tables (serverless).
        """
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.catalog import (
            OnlineTableSpec,
            OnlineTableSpecTriggeredSchedulingPolicy,
        )

        w = WorkspaceClient()

        spec = OnlineTableSpec(
            source_table_full_name=table_name,
            primary_key_columns=["user_id"],
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
            perform_full_copy=False,
        )

        online_table_name = f"{table_name}_online"

        try:
            w.online_tables.create(
                name=online_table_name,
                spec=spec,
            )
            logger.info(
                "Online table created: %s (source: %s)",
                online_table_name, table_name,
            )
        except Exception as exc:
            if "ALREADY_EXISTS" in str(exc):
                logger.info("Online table %s already exists. Triggering refresh.", online_table_name)
                # Pipeline refresh will happen automatically
            else:
                logger.error("Failed to create online table: %s", exc)
                raise

    def publish_all_online(self) -> None:
        """Publish all feature tables to online stores."""
        tables = [USER_FEATURE_TABLE, TXN_FEATURE_TABLE]
        for table in tables:
            try:
                self.publish_to_online_store(table)
            except Exception as exc:
                logger.error("Failed to publish %s to online store: %s", table, exc)

    # ----- Feature Table Metadata -----

    def get_feature_table_info(self, table_name: str) -> Dict:
        """Get metadata about a feature table."""
        try:
            # Read table properties
            df = self.spark.read.table(table_name)
            schema_info = [
                {"name": f.name, "type": str(f.dataType), "nullable": f.nullable}
                for f in df.schema.fields
            ]

            info = {
                "table_name": table_name,
                "columns": schema_info,
                "column_count": len(df.columns),
                "row_count": df.count(),
            }

            logger.info("Feature table info: %s — %d cols, %d rows",
                        table_name, info["column_count"], info["row_count"])
            return info

        except Exception as exc:
            logger.error("Failed to get info for %s: %s", table_name, exc)
            return {"table_name": table_name, "error": str(exc)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Feature Store management")
    parser.add_argument(
        "--action",
        choices=["create", "update", "lookup", "monitor", "publish-online", "info"],
        required=True,
    )
    parser.add_argument("--table", default=None, help="Feature table name")
    return parser.parse_args()


def main():
    args = parse_args()
    spark = SparkSession.builder.appName("FeatureStore").getOrCreate()
    spark.sql(f"USE CATALOG {CATALOG}")

    manager = FeatureStoreManager(spark)

    if args.action == "create":
        # Build feature DataFrames from Silver tables
        from delta_lake.silver.transform_clean import SilverTransformations
        logger.info("Creating feature tables from Silver data...")

        user_features = spark.read.table(f"{CATALOG}.{GOLD_SCHEMA}.feature_table")
        manager.create_user_feature_table(user_features)

        txn_features = (
            spark.read.table(f"{CATALOG}.silver.transactions")
            .groupBy("user_id")
            .agg(
                F.sum("amount").alias("total_spend"),
                F.count("*").alias("total_transactions"),
                F.avg("amount").alias("avg_transaction_value"),
            )
        )
        manager.create_transaction_feature_table(txn_features)

    elif args.action == "monitor":
        reports = manager.monitor_all_tables()
        for report in reports:
            print(json.dumps(report, indent=2, default=str))

    elif args.action == "publish-online":
        manager.publish_all_online()

    elif args.action == "info":
        table = args.table or USER_FEATURE_TABLE
        info = manager.get_feature_table_info(table)
        print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
