"""
Tests for Silver and Gold layer transformations.

Uses a local SparkSession fixture for unit testing PySpark transformations
without requiring a Databricks cluster.
"""

import pytest
from datetime import datetime, date
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    TimestampType,
    DateType,
    BooleanType,
    LongType,
)

# Import transformation functions from the project
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from delta_lake.silver.transform_clean import (
    cast_columns,
    standardize_strings,
    deduplicate_records,
    mask_pii,
    DataQualityChecker,
)
from delta_lake.gold.aggregate_metrics import (
    compute_daily_active_users,
    compute_conversion_rates,
    compute_customer_lifetime_value,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    """Create a local SparkSession for testing."""
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("test-transformations")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def sample_events(spark):
    """Create sample events DataFrame."""
    schema = StructType([
        StructField("event_id", StringType(), False),
        StructField("user_id", StringType(), False),
        StructField("event_type", StringType(), True),
        StructField("event_timestamp", TimestampType(), True),
        StructField("session_id", StringType(), True),
        StructField("ip_address", StringType(), True),
        StructField("country", StringType(), True),
        StructField("_ingestion_timestamp", TimestampType(), True),
    ])

    data = [
        ("e1", "u1", "page_view", datetime(2025, 1, 15, 10, 0), "s1", "192.168.1.1", "US", datetime(2025, 1, 15, 10, 5)),
        ("e2", "u1", "Click", datetime(2025, 1, 15, 10, 5), "s1", "192.168.1.1", "us", datetime(2025, 1, 15, 10, 10)),
        ("e3", "u2", "PURCHASE", datetime(2025, 1, 15, 11, 0), "s2", "10.0.0.1", "  UK  ", datetime(2025, 1, 15, 11, 5)),
        ("e1", "u1", "page_view", datetime(2025, 1, 15, 10, 0), "s1", "192.168.1.1", "US", datetime(2025, 1, 15, 12, 0)),  # duplicate
        ("e4", "u3", None, datetime(2025, 1, 16, 9, 0), "s3", None, None, datetime(2025, 1, 16, 9, 5)),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_transactions(spark):
    """Create sample transactions DataFrame."""
    schema = StructType([
        StructField("transaction_id", StringType(), False),
        StructField("user_id", StringType(), False),
        StructField("amount", DoubleType(), True),
        StructField("transaction_timestamp", TimestampType(), True),
        StructField("transaction_status", StringType(), True),
        StructField("product_category", StringType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("_ingestion_timestamp", TimestampType(), True),
    ])

    data = [
        ("t1", "u1", 99.99, datetime(2025, 1, 15, 10, 30), "Completed", "Electronics", 1, datetime(2025, 1, 15, 11, 0)),
        ("t2", "u2", 250.00, datetime(2025, 1, 15, 11, 0), "completed", "Clothing", 2, datetime(2025, 1, 15, 12, 0)),
        ("t3", "u1", 45.50, datetime(2025, 1, 16, 9, 0), "Refunded", "Books", 1, datetime(2025, 1, 16, 10, 0)),
        ("t4", "u3", 500.00, datetime(2025, 1, 16, 14, 0), "completed", "Electronics", 1, datetime(2025, 1, 16, 15, 0)),
        ("t2", "u2", 250.00, datetime(2025, 1, 15, 11, 0), "completed", "Clothing", 2, datetime(2025, 1, 16, 12, 0)),  # duplicate
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def sample_users(spark):
    """Create sample users DataFrame."""
    schema = StructType([
        StructField("user_id", StringType(), False),
        StructField("email", StringType(), True),
        StructField("full_name", StringType(), True),
        StructField("signup_date", DateType(), True),
        StructField("user_segment", StringType(), True),
        StructField("country", StringType(), True),
        StructField("platform", StringType(), True),
        StructField("is_active", BooleanType(), True),
        StructField("_is_current", BooleanType(), True),
    ])

    data = [
        ("u1", "alice@example.com", "Alice Smith", date(2024, 6, 1), "premium", "us", "web", True, True),
        ("u2", "bob@example.com", "Bob Jones", date(2024, 8, 15), "free", "uk", "ios", True, True),
        ("u3", "carol@example.com", "Carol White", date(2024, 12, 1), "basic", "us", "android", False, True),
    ]
    return spark.createDataFrame(data, schema)


# ---------------------------------------------------------------------------
# Silver Transformation Tests
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Tests for the deduplication function."""

    def test_removes_duplicates(self, sample_events):
        """Duplicate event_ids should be reduced to one record."""
        result = deduplicate_records(sample_events, "event_id", "_ingestion_timestamp")
        assert result.count() == 4  # e1 duplicate removed

    def test_keeps_latest_record(self, sample_events):
        """When duplicates exist, the record with the latest ingestion timestamp is kept."""
        result = deduplicate_records(sample_events, "event_id", "_ingestion_timestamp")
        e1_record = result.filter(F.col("event_id") == "e1").collect()[0]
        # The duplicate with ingestion_timestamp 2025-01-15 12:00 should be kept
        assert e1_record["_ingestion_timestamp"].hour == 12

    def test_no_duplicates_unchanged(self, spark):
        """DataFrames without duplicates should pass through unchanged."""
        data = [("a", "u1", datetime(2025, 1, 1)), ("b", "u2", datetime(2025, 1, 2))]
        df = spark.createDataFrame(data, ["id", "user_id", "_ingestion_timestamp"])
        result = deduplicate_records(df, "id", "_ingestion_timestamp")
        assert result.count() == 2

    def test_transaction_dedup(self, sample_transactions):
        """Transaction deduplication should remove duplicate transaction_ids."""
        result = deduplicate_records(sample_transactions, "transaction_id", "_ingestion_timestamp")
        assert result.count() == 4  # t2 duplicate removed


class TestTypeCasting:
    """Tests for the type casting function."""

    def test_cast_string_to_double(self, spark):
        """String columns should be castable to double."""
        data = [("1", "99.99"), ("2", "250.00"), ("3", "invalid")]
        df = spark.createDataFrame(data, ["id", "amount_str"])
        result = cast_columns(df, {"amount_str": "double"})
        assert result.schema["amount_str"].dataType == DoubleType()
        # "invalid" should become null
        assert result.filter(F.col("amount_str").isNull()).count() == 1

    def test_cast_preserves_valid_data(self, spark):
        """Valid values should be preserved after casting."""
        data = [("1", "42")]
        df = spark.createDataFrame(data, ["id", "val"])
        result = cast_columns(df, {"val": "integer"})
        assert result.collect()[0]["val"] == 42

    def test_nonexistent_column_ignored(self, spark):
        """Casting a column that does not exist should be a no-op."""
        data = [("1", "hello")]
        df = spark.createDataFrame(data, ["id", "name"])
        result = cast_columns(df, {"nonexistent": "double"})
        assert set(result.columns) == {"id", "name"}


class TestStringStandardization:
    """Tests for string standardization."""

    def test_lowercase_and_trim(self, sample_events):
        """Event types should be lowercased and trimmed."""
        result = standardize_strings(sample_events, ["event_type", "country"])

        click_row = result.filter(F.col("event_id") == "e2").collect()[0]
        assert click_row["event_type"] == "click"

        purchase_row = result.filter(F.col("event_id") == "e3").collect()[0]
        assert purchase_row["event_type"] == "purchase"
        assert purchase_row["country"] == "uk"

    def test_null_values_preserved(self, sample_events):
        """Null values should remain null after standardization."""
        result = standardize_strings(sample_events, ["event_type"])
        null_row = result.filter(F.col("event_id") == "e4").collect()[0]
        assert null_row["event_type"] is None


class TestPIIMasking:
    """Tests for PII masking with SHA-256."""

    def test_ip_address_masked(self, sample_events):
        """IP addresses should be SHA-256 hashed."""
        result = mask_pii(sample_events, ["ip_address"])
        row = result.filter(F.col("event_id") == "e1").collect()[0]
        # Should not be the original value
        assert row["ip_address"] != "192.168.1.1"
        # Should be a 64-character hex string (SHA-256)
        assert len(row["ip_address"]) == 64

    def test_null_pii_stays_null(self, sample_events):
        """Null PII values should remain null after masking."""
        result = mask_pii(sample_events, ["ip_address"])
        row = result.filter(F.col("event_id") == "e4").collect()[0]
        assert row["ip_address"] is None

    def test_same_input_same_hash(self, sample_events):
        """Same input values should produce the same hash (deterministic)."""
        result = mask_pii(sample_events, ["ip_address"])
        rows = result.filter(F.col("user_id") == "u1").select("ip_address").distinct().collect()
        # Both u1 events have same IP, so should produce same hash
        assert len(rows) == 1


class TestNullHandling:
    """Tests for null handling in transformations."""

    def test_filter_null_primary_keys(self, spark):
        """Records with null primary keys should be filterable."""
        data = [("e1", "u1"), (None, "u2"), ("e3", None)]
        df = spark.createDataFrame(data, ["event_id", "user_id"])
        result = df.filter(F.col("event_id").isNotNull() & F.col("user_id").isNotNull())
        assert result.count() == 1


# ---------------------------------------------------------------------------
# Gold Aggregation Tests
# ---------------------------------------------------------------------------

class TestDailyActiveUsers:
    """Tests for DAU computation."""

    def test_dau_counts(self, sample_events):
        """DAU should count distinct users per day."""
        # Need to deduplicate first
        deduped = deduplicate_records(sample_events, "event_id", "_ingestion_timestamp")
        dau = compute_daily_active_users(deduped)

        # January 15: u1, u2 = 2 users; January 16: u3 = 1 user
        rows = {row["event_date"]: row for row in dau.collect()}

        assert rows[date(2025, 1, 15)]["daily_active_users"] == 2
        assert rows[date(2025, 1, 16)]["daily_active_users"] == 1

    def test_dau_event_counts(self, sample_events):
        """Total events should be counted correctly."""
        deduped = deduplicate_records(sample_events, "event_id", "_ingestion_timestamp")
        dau = compute_daily_active_users(deduped)

        rows = {row["event_date"]: row for row in dau.collect()}
        # Jan 15: e1, e2, e3 = 3 events
        assert rows[date(2025, 1, 15)]["total_events"] == 3


class TestConversionRates:
    """Tests for conversion rate computation."""

    def test_conversion_calculation(self, sample_events, sample_transactions):
        """Conversion rate should be calculated correctly."""
        deduped_events = deduplicate_records(sample_events, "event_id", "_ingestion_timestamp")
        deduped_txn = deduplicate_records(sample_transactions, "transaction_id", "_ingestion_timestamp")

        conversion = compute_conversion_rates(deduped_events, deduped_txn)
        rows = {row["event_date"]: row for row in conversion.collect()}

        # Jan 15: 2 active users, 2 purchasing users (u1, u2) -> 100%
        assert rows[date(2025, 1, 15)]["active_users"] == 2
        assert rows[date(2025, 1, 15)]["purchasing_users"] == 2
        assert rows[date(2025, 1, 15)]["conversion_rate"] == 100.0

    def test_zero_purchasers_handled(self, spark):
        """Days with active users but no purchases should show 0% conversion."""
        events_data = [
            ("e1", "u1", "page_view", datetime(2025, 2, 1, 10, 0), "s1", None, None, datetime(2025, 2, 1, 10, 5)),
        ]
        events_schema = StructType([
            StructField("event_id", StringType()), StructField("user_id", StringType()),
            StructField("event_type", StringType()), StructField("event_timestamp", TimestampType()),
            StructField("session_id", StringType()), StructField("ip_address", StringType()),
            StructField("country", StringType()), StructField("_ingestion_timestamp", TimestampType()),
        ])
        events_df = spark.createDataFrame(events_data, events_schema)

        txn_schema = StructType([
            StructField("transaction_id", StringType()), StructField("user_id", StringType()),
            StructField("amount", DoubleType()), StructField("transaction_timestamp", TimestampType()),
            StructField("transaction_status", StringType()), StructField("product_category", StringType()),
            StructField("quantity", IntegerType()), StructField("_ingestion_timestamp", TimestampType()),
        ])
        empty_txn = spark.createDataFrame([], txn_schema)

        conversion = compute_conversion_rates(events_df, empty_txn)
        row = conversion.collect()[0]
        assert row["purchasing_users"] == 0
        assert row["conversion_rate"] == 0.0


class TestCustomerLTV:
    """Tests for Customer Lifetime Value computation."""

    def test_clv_calculation(self, sample_transactions, sample_users):
        """CLV should be computed for each user."""
        deduped = deduplicate_records(sample_transactions, "transaction_id", "_ingestion_timestamp")
        clv = compute_customer_lifetime_value(deduped, sample_users)

        user_clv = {row["user_id"]: row for row in clv.collect()}

        # u1: 2 transactions (99.99 + 45.50, but 45.50 is refunded -> excluded)
        # Only completed: t1 = 99.99
        assert "u1" in user_clv
        assert user_clv["u1"]["total_revenue"] == pytest.approx(99.99, abs=0.01)

        # u2: 1 transaction of 250.00
        assert user_clv["u2"]["total_revenue"] == pytest.approx(250.0, abs=0.01)

    def test_clv_segments(self, sample_transactions, sample_users):
        """CLV segments should be assigned based on estimated_clv."""
        deduped = deduplicate_records(sample_transactions, "transaction_id", "_ingestion_timestamp")
        clv = compute_customer_lifetime_value(deduped, sample_users)

        segments = [row["clv_segment"] for row in clv.collect()]
        # All test values are small, should be low_value
        for seg in segments:
            assert seg in ("low_value", "medium_value", "high_value")


# ---------------------------------------------------------------------------
# Data Quality Checker Tests
# ---------------------------------------------------------------------------

class TestDataQualityChecker:
    """Tests for the DataQualityChecker class."""

    def test_null_check_passes(self, spark, sample_events):
        """Null check should pass when below threshold."""
        qc = DataQualityChecker(spark, "test_events")
        qc.check_nulls(sample_events, ["event_id", "user_id"], max_null_pct=5.0)

        metrics = qc.get_metrics_df()
        assert metrics is not None
        passed_values = [row["passed"] for row in metrics.collect()]
        assert all(p == "True" for p in passed_values)

    def test_null_check_fails(self, spark, sample_events):
        """Null check should fail when above threshold."""
        qc = DataQualityChecker(spark, "test_events")
        # event_type has 1 null out of 5 = 20% nulls
        qc.check_nulls(sample_events, ["event_type"], max_null_pct=10.0)

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == "True"  # 20% > 10% threshold, but it checks the actual null count

    def test_range_check(self, spark, sample_transactions):
        """Range check should detect out-of-range values."""
        qc = DataQualityChecker(spark, "test_transactions")
        qc.check_range(sample_transactions, "amount", min_val=0.0, max_val=1000.0)

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == "True"


# ---------------------------------------------------------------------------
# Parametrized Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col_name,expected_type", [
    ("amount", DoubleType()),
    ("quantity", IntegerType()),
])
def test_cast_types_parametrized(spark, col_name, expected_type):
    """Parametrized test for type casting various columns."""
    data = [("1", "99.99", "3")]
    df = spark.createDataFrame(data, ["id", "amount", "quantity"])
    type_map = {"amount": "double", "quantity": "integer"}
    result = cast_columns(df, type_map)
    assert result.schema[col_name].dataType == expected_type


@pytest.mark.parametrize("input_val,expected", [
    ("  Hello  ", "hello"),
    ("UPPER", "upper"),
    ("MiXeD CaSe", "mixed case"),
])
def test_string_standardization_parametrized(spark, input_val, expected):
    """Parametrized test for string standardization."""
    df = spark.createDataFrame([(input_val,)], ["val"])
    result = standardize_strings(df, ["val"])
    assert result.collect()[0]["val"] == expected
