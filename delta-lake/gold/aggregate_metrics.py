"""
Gold Layer — Business Metric Aggregations

Compute business KPIs from Silver tables: daily active users, revenue metrics,
conversion rates, retention cohorts, and customer lifetime value. Produce
materialized aggregations at daily, weekly, and monthly granularity.

Usage:
    spark-submit delta-lake/gold/aggregate_metrics.py
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
import logging
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gold.aggregate")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CATALOG = "analytics_platform"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"


# ---------------------------------------------------------------------------
# KPI Calculation Functions
# ---------------------------------------------------------------------------

def compute_daily_active_users(events_df: DataFrame) -> DataFrame:
    """
    Compute daily active users (DAU) and related engagement metrics.
    An active user is one who triggered at least one event on a given day.
    """
    dau = (
        events_df
        .withColumn("event_date", F.to_date("event_timestamp"))
        .groupBy("event_date")
        .agg(
            F.countDistinct("user_id").alias("daily_active_users"),
            F.count("*").alias("total_events"),
            F.countDistinct("session_id").alias("total_sessions"),
            F.countDistinct("event_type").alias("unique_event_types"),
        )
        .withColumn(
            "events_per_user",
            F.round(F.col("total_events") / F.col("daily_active_users"), 2),
        )
        .withColumn(
            "sessions_per_user",
            F.round(F.col("total_sessions") / F.col("daily_active_users"), 2),
        )
        .orderBy("event_date")
    )

    # Add 7-day and 30-day rolling averages
    window_7d = Window.orderBy("event_date").rowsBetween(-6, 0)
    window_30d = Window.orderBy("event_date").rowsBetween(-29, 0)

    dau = (
        dau
        .withColumn("dau_7d_avg", F.round(F.avg("daily_active_users").over(window_7d), 2))
        .withColumn("dau_30d_avg", F.round(F.avg("daily_active_users").over(window_30d), 2))
    )

    # WAU and MAU approximations using rolling distinct (pre-aggregated here as running sum)
    logger.info("DAU metrics computed.")
    return dau


def compute_revenue_metrics(transactions_df: DataFrame) -> dict:
    """
    Compute revenue KPIs at daily, weekly, and monthly grain.
    Returns a dict of DataFrames keyed by granularity.
    """
    txn_with_dates = (
        transactions_df
        .filter(F.col("transaction_status") != "refunded")
        .withColumn("txn_date", F.to_date("transaction_timestamp"))
        .withColumn("txn_week", F.date_trunc("week", F.col("transaction_timestamp")))
        .withColumn("txn_month", F.date_trunc("month", F.col("transaction_timestamp")))
    )

    # --- Daily Revenue ---
    daily_revenue = (
        txn_with_dates
        .groupBy("txn_date")
        .agg(
            F.sum("amount").alias("gross_revenue"),
            F.count("*").alias("transaction_count"),
            F.countDistinct("user_id").alias("paying_users"),
            F.avg("amount").alias("avg_order_value"),
            F.expr("percentile_approx(amount, 0.5)").alias("median_order_value"),
            F.max("amount").alias("max_order_value"),
        )
        .withColumn(
            "revenue_per_user",
            F.round(F.col("gross_revenue") / F.col("paying_users"), 2),
        )
        .orderBy("txn_date")
    )

    # Running total
    running_window = Window.orderBy("txn_date").rowsBetween(Window.unboundedPreceding, 0)
    daily_revenue = daily_revenue.withColumn(
        "cumulative_revenue", F.sum("gross_revenue").over(running_window)
    )

    # 7-day moving average
    window_7d = Window.orderBy("txn_date").rowsBetween(-6, 0)
    daily_revenue = daily_revenue.withColumn(
        "revenue_7d_avg", F.round(F.avg("gross_revenue").over(window_7d), 2)
    )

    # --- Weekly Revenue ---
    weekly_revenue = (
        txn_with_dates
        .groupBy("txn_week")
        .agg(
            F.sum("amount").alias("gross_revenue"),
            F.count("*").alias("transaction_count"),
            F.countDistinct("user_id").alias("paying_users"),
            F.avg("amount").alias("avg_order_value"),
        )
        .orderBy("txn_week")
    )

    # Week-over-week growth
    wow_window = Window.orderBy("txn_week")
    weekly_revenue = (
        weekly_revenue
        .withColumn("prev_week_revenue", F.lag("gross_revenue").over(wow_window))
        .withColumn(
            "wow_growth_pct",
            F.round(
                (F.col("gross_revenue") - F.col("prev_week_revenue"))
                / F.col("prev_week_revenue") * 100,
                2,
            )
        )
        .drop("prev_week_revenue")
    )

    # --- Monthly Revenue ---
    monthly_revenue = (
        txn_with_dates
        .groupBy("txn_month")
        .agg(
            F.sum("amount").alias("gross_revenue"),
            F.count("*").alias("transaction_count"),
            F.countDistinct("user_id").alias("paying_users"),
            F.avg("amount").alias("avg_order_value"),
        )
        .orderBy("txn_month")
    )

    mom_window = Window.orderBy("txn_month")
    monthly_revenue = (
        monthly_revenue
        .withColumn("prev_month_revenue", F.lag("gross_revenue").over(mom_window))
        .withColumn(
            "mom_growth_pct",
            F.round(
                (F.col("gross_revenue") - F.col("prev_month_revenue"))
                / F.col("prev_month_revenue") * 100,
                2,
            )
        )
        .drop("prev_month_revenue")
    )

    logger.info("Revenue metrics computed at daily, weekly, monthly grain.")
    return {
        "daily": daily_revenue,
        "weekly": weekly_revenue,
        "monthly": monthly_revenue,
    }


def compute_conversion_rates(
    events_df: DataFrame,
    transactions_df: DataFrame,
) -> DataFrame:
    """
    Compute daily conversion rates: proportion of active users who made a purchase.
    """
    daily_active = (
        events_df
        .withColumn("event_date", F.to_date("event_timestamp"))
        .groupBy("event_date")
        .agg(F.countDistinct("user_id").alias("active_users"))
    )

    daily_purchasers = (
        transactions_df
        .filter(F.col("transaction_status") != "refunded")
        .withColumn("txn_date", F.to_date("transaction_timestamp"))
        .groupBy(F.col("txn_date").alias("event_date"))
        .agg(F.countDistinct("user_id").alias("purchasing_users"))
    )

    conversion = (
        daily_active
        .join(daily_purchasers, on="event_date", how="left")
        .fillna(0, subset=["purchasing_users"])
        .withColumn(
            "conversion_rate",
            F.round(F.col("purchasing_users") / F.col("active_users") * 100, 4),
        )
        .orderBy("event_date")
    )

    # 7d moving average of conversion rate
    window_7d = Window.orderBy("event_date").rowsBetween(-6, 0)
    conversion = conversion.withColumn(
        "conversion_rate_7d_avg", F.round(F.avg("conversion_rate").over(window_7d), 4)
    )

    logger.info("Conversion rates computed.")
    return conversion


def compute_retention_cohorts(
    events_df: DataFrame,
    users_df: DataFrame,
) -> DataFrame:
    """
    Compute monthly retention cohorts based on user signup month.
    """
    # Cohort assignment: month of first activity (signup_date or first event)
    user_first_event = (
        events_df
        .groupBy("user_id")
        .agg(F.min(F.to_date("event_timestamp")).alias("first_event_date"))
    )

    cohort_base = (
        users_df
        .join(user_first_event, on="user_id", how="left")
        .withColumn(
            "cohort_month",
            F.date_trunc("month", F.coalesce(F.col("signup_date"), F.col("first_event_date"))),
        )
    )

    # Monthly activity per user
    monthly_activity = (
        events_df
        .withColumn("activity_month", F.date_trunc("month", F.col("event_timestamp")))
        .select("user_id", "activity_month")
        .distinct()
    )

    # Join cohort with activity
    retention_base = (
        cohort_base
        .select("user_id", "cohort_month")
        .join(monthly_activity, on="user_id", how="inner")
        .withColumn(
            "months_since_cohort",
            F.months_between(F.col("activity_month"), F.col("cohort_month")).cast("int"),
        )
        .filter(F.col("months_since_cohort") >= 0)
    )

    # Cohort sizes
    cohort_sizes = (
        retention_base
        .filter(F.col("months_since_cohort") == 0)
        .groupBy("cohort_month")
        .agg(F.countDistinct("user_id").alias("cohort_size"))
    )

    # Retained users per cohort per month
    retention = (
        retention_base
        .groupBy("cohort_month", "months_since_cohort")
        .agg(F.countDistinct("user_id").alias("retained_users"))
        .join(cohort_sizes, on="cohort_month", how="left")
        .withColumn(
            "retention_rate",
            F.round(F.col("retained_users") / F.col("cohort_size") * 100, 2),
        )
        .orderBy("cohort_month", "months_since_cohort")
    )

    logger.info("Retention cohorts computed.")
    return retention


def compute_customer_lifetime_value(
    transactions_df: DataFrame,
    users_df: DataFrame,
) -> DataFrame:
    """
    Compute Customer Lifetime Value (CLV) per user using historical transaction data.
    CLV = avg_order_value * purchase_frequency * customer_lifespan
    """
    user_txn = (
        transactions_df
        .filter(F.col("transaction_status") != "refunded")
        .groupBy("user_id")
        .agg(
            F.sum("amount").alias("total_revenue"),
            F.count("*").alias("total_orders"),
            F.avg("amount").alias("avg_order_value"),
            F.min("transaction_timestamp").alias("first_purchase"),
            F.max("transaction_timestamp").alias("last_purchase"),
        )
        .withColumn(
            "customer_lifespan_days",
            F.datediff(F.col("last_purchase"), F.col("first_purchase")),
        )
        .withColumn(
            "customer_lifespan_months",
            F.round(F.col("customer_lifespan_days") / 30.0, 1),
        )
        .withColumn(
            "purchase_frequency_monthly",
            F.when(
                F.col("customer_lifespan_months") > 0,
                F.round(F.col("total_orders") / F.col("customer_lifespan_months"), 2),
            ).otherwise(F.col("total_orders")),
        )
        .withColumn(
            "estimated_clv",
            F.round(
                F.col("avg_order_value")
                * F.col("purchase_frequency_monthly")
                * F.greatest(F.col("customer_lifespan_months"), F.lit(1)),
                2,
            ),
        )
    )

    # Join with user segments for segmented CLV
    clv = (
        user_txn
        .join(
            users_df.select("user_id", "user_segment", "signup_date"),
            on="user_id",
            how="left",
        )
        .withColumn(
            "clv_segment",
            F.when(F.col("estimated_clv") >= 1000, "high_value")
            .when(F.col("estimated_clv") >= 200, "medium_value")
            .otherwise("low_value"),
        )
    )

    logger.info("Customer Lifetime Value computed.")
    return clv


# ---------------------------------------------------------------------------
# Write Gold tables
# ---------------------------------------------------------------------------

def write_gold_table(df: DataFrame, table_name: str, partition_cols: Optional[list] = None) -> None:
    """Write a DataFrame to a Gold Delta table with optimizations."""
    full_table = f"{CATALOG}.{GOLD_SCHEMA}.{table_name}"

    writer = (
        df
        .withColumn("_gold_updated_at", F.current_timestamp())
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .option("delta.autoOptimize.optimizeWrite", "true")
        .option("delta.autoOptimize.autoCompact", "true")
    )

    if partition_cols:
        writer = writer.partitionBy(*partition_cols)

    writer.saveAsTable(full_table)
    logger.info("Gold table written: %s", full_table)


def optimize_gold_tables(spark: SparkSession) -> None:
    """Run OPTIMIZE on Gold tables."""
    gold_tables = [
        (f"{CATALOG}.{GOLD_SCHEMA}.daily_active_users", ["event_date"]),
        (f"{CATALOG}.{GOLD_SCHEMA}.daily_revenue", ["txn_date"]),
        (f"{CATALOG}.{GOLD_SCHEMA}.conversion_rates", ["event_date"]),
        (f"{CATALOG}.{GOLD_SCHEMA}.retention_cohorts", ["cohort_month"]),
        (f"{CATALOG}.{GOLD_SCHEMA}.customer_ltv", ["user_id"]),
    ]

    for table, zorder_cols in gold_tables:
        try:
            zorder_clause = ", ".join(zorder_cols)
            spark.sql(f"OPTIMIZE {table} ZORDER BY ({zorder_clause})")
            logger.info("Optimized %s (ZORDER: %s)", table, zorder_clause)
        except Exception as exc:
            logger.error("Failed to optimize %s: %s", table, exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Build all Gold-layer aggregation tables."""
    spark = SparkSession.builder.appName("GoldAggregation").getOrCreate()
    spark.sql(f"USE CATALOG {CATALOG}")

    # Load Silver tables
    events_df = spark.read.table(f"{CATALOG}.{SILVER_SCHEMA}.events")
    transactions_df = spark.read.table(f"{CATALOG}.{SILVER_SCHEMA}.transactions")
    users_df = spark.read.table(f"{CATALOG}.{SILVER_SCHEMA}.users").filter(F.col("_is_current") == True)

    logger.info("Silver tables loaded. Events=%d, Transactions=%d, Users=%d",
                events_df.count(), transactions_df.count(), users_df.count())

    # --- DAU ---
    logger.info("=" * 60)
    logger.info("Computing Daily Active Users")
    logger.info("=" * 60)
    dau_df = compute_daily_active_users(events_df)
    write_gold_table(dau_df, "daily_active_users")

    # --- Revenue ---
    logger.info("=" * 60)
    logger.info("Computing Revenue Metrics")
    logger.info("=" * 60)
    revenue = compute_revenue_metrics(transactions_df)
    write_gold_table(revenue["daily"], "daily_revenue")
    write_gold_table(revenue["weekly"], "weekly_revenue")
    write_gold_table(revenue["monthly"], "monthly_revenue")

    # --- Conversion ---
    logger.info("=" * 60)
    logger.info("Computing Conversion Rates")
    logger.info("=" * 60)
    conversion_df = compute_conversion_rates(events_df, transactions_df)
    write_gold_table(conversion_df, "conversion_rates")

    # --- Retention ---
    logger.info("=" * 60)
    logger.info("Computing Retention Cohorts")
    logger.info("=" * 60)
    retention_df = compute_retention_cohorts(events_df, users_df)
    write_gold_table(retention_df, "retention_cohorts")

    # --- CLV ---
    logger.info("=" * 60)
    logger.info("Computing Customer Lifetime Value")
    logger.info("=" * 60)
    clv_df = compute_customer_lifetime_value(transactions_df, users_df)
    write_gold_table(clv_df, "customer_ltv")

    # --- Optimize ---
    logger.info("=" * 60)
    logger.info("Optimizing Gold tables")
    logger.info("=" * 60)
    optimize_gold_tables(spark)

    logger.info("Gold aggregation pipeline complete.")


if __name__ == "__main__":
    main()
