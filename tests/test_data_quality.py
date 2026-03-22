"""
Tests for data quality check functions.

Validates schema checks, null checks, range validations, and referential
integrity checks using local SparkSession fixtures.
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
    LongType,
)

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from delta_lake.silver.transform_clean import DataQualityChecker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    """Create a local SparkSession for testing."""
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("test-data-quality")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def events_df(spark):
    """Sample events DataFrame for quality tests."""
    schema = StructType([
        StructField("event_id", StringType(), False),
        StructField("user_id", StringType(), False),
        StructField("event_type", StringType(), True),
        StructField("event_timestamp", TimestampType(), True),
        StructField("amount", DoubleType(), True),
    ])
    data = [
        ("e1", "u1", "click", datetime(2025, 1, 15, 10, 0), 10.0),
        ("e2", "u2", "purchase", datetime(2025, 1, 15, 11, 0), 250.0),
        ("e3", "u3", None, datetime(2025, 1, 15, 12, 0), None),
        ("e4", "u4", "page_view", None, 0.5),
        ("e5", "u5", "click", datetime(2025, 1, 16, 9, 0), 99999.0),
    ]
    return spark.createDataFrame(data, schema)


@pytest.fixture
def users_df(spark):
    """Sample users DataFrame."""
    data = [("u1",), ("u2",), ("u3",), ("u10",)]
    return spark.createDataFrame(data, ["user_id"])


@pytest.fixture
def transactions_df(spark):
    """Sample transactions DataFrame with foreign keys."""
    data = [
        ("t1", "u1", 100.0),
        ("t2", "u2", 200.0),
        ("t3", "u99", 50.0),  # orphan — u99 not in users
    ]
    return spark.createDataFrame(data, ["transaction_id", "user_id", "amount"])


# ---------------------------------------------------------------------------
# Schema Validation Tests
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """Tests for schema validation checks."""

    def test_schema_check_passes_when_all_columns_present(self, spark, events_df):
        """Schema check should pass when all expected columns exist."""
        qc = DataQualityChecker(spark, "events")
        qc.check_schema(events_df, ["event_id", "user_id", "event_type"])

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == "True"

    def test_schema_check_fails_on_missing_column(self, spark, events_df):
        """Schema check should fail when an expected column is missing."""
        qc = DataQualityChecker(spark, "events")
        qc.check_schema(events_df, ["event_id", "nonexistent_column"])

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == "False"

    def test_schema_check_empty_expected(self, spark, events_df):
        """Schema check with empty expected columns should pass."""
        qc = DataQualityChecker(spark, "events")
        qc.check_schema(events_df, [])

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == "True"

    def test_schema_check_records_missing_columns(self, spark, events_df):
        """Schema check should record which columns are missing."""
        qc = DataQualityChecker(spark, "events")
        qc.check_schema(events_df, ["event_id", "col_a", "col_b"])

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert "col_a" in row["missing_columns"]
        assert "col_b" in row["missing_columns"]


# ---------------------------------------------------------------------------
# Null Validation Tests
# ---------------------------------------------------------------------------

class TestNullChecks:
    """Tests for null validation."""

    def test_null_check_passes_for_complete_column(self, spark, events_df):
        """Column with no nulls should pass null check."""
        qc = DataQualityChecker(spark, "events")
        qc.check_nulls(events_df, ["event_id"], max_null_pct=0.0)

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == "True"

    def test_null_check_fails_for_column_above_threshold(self, spark, events_df):
        """Column with nulls above threshold should fail."""
        # event_type has 1 null out of 5 = 20%
        qc = DataQualityChecker(spark, "events")
        qc.check_nulls(events_df, ["event_type"], max_null_pct=10.0)

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        # 20% > 10% threshold -> should fail
        assert row["passed"] == "False"

    def test_null_check_passes_at_threshold(self, spark, events_df):
        """Column with nulls exactly at threshold should pass."""
        qc = DataQualityChecker(spark, "events")
        qc.check_nulls(events_df, ["event_type"], max_null_pct=20.0)

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == "True"

    def test_null_check_multiple_columns(self, spark, events_df):
        """Multiple columns should each be checked independently."""
        qc = DataQualityChecker(spark, "events")
        qc.check_nulls(events_df, ["event_type", "event_timestamp", "amount"], max_null_pct=10.0)

        metrics = qc.get_metrics_df()
        assert metrics.count() == 3

    def test_null_check_reports_counts(self, spark, events_df):
        """Null check should report correct null counts."""
        qc = DataQualityChecker(spark, "events")
        qc.check_nulls(events_df, ["event_type"], max_null_pct=50.0)

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["null_count"] == "1"

    @pytest.mark.parametrize("threshold,expected_pass", [
        (0.0, False),
        (10.0, False),
        (19.99, False),
        (20.0, True),
        (50.0, True),
        (100.0, True),
    ])
    def test_null_check_various_thresholds(self, spark, events_df, threshold, expected_pass):
        """Parametrized test for different null thresholds."""
        qc = DataQualityChecker(spark, "events")
        qc.check_nulls(events_df, ["event_type"], max_null_pct=threshold)

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == str(expected_pass)


# ---------------------------------------------------------------------------
# Range Validation Tests
# ---------------------------------------------------------------------------

class TestRangeChecks:
    """Tests for range validation."""

    def test_range_check_passes_when_all_in_range(self, spark, events_df):
        """Range check should pass when all values are within bounds."""
        qc = DataQualityChecker(spark, "events")
        qc.check_range(events_df, "amount", min_val=0.0, max_val=100000.0)

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == "True"

    def test_range_check_fails_on_outlier(self, spark, events_df):
        """Range check should fail when values exceed the range."""
        qc = DataQualityChecker(spark, "events")
        # e5 has amount=99999.0 which exceeds max=1000
        qc.check_range(events_df, "amount", min_val=0.0, max_val=1000.0)

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == "False"
        assert row["out_of_range_count"] == "1"

    def test_range_check_with_negatives(self, spark):
        """Range check should catch negative values when min is zero."""
        data = [(1, -5.0), (2, 10.0), (3, -1.0)]
        df = spark.createDataFrame(data, ["id", "value"])

        qc = DataQualityChecker(spark, "test")
        qc.check_range(df, "value", min_val=0.0, max_val=100.0)

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == "False"
        assert row["out_of_range_count"] == "2"

    @pytest.mark.parametrize("min_val,max_val,expected_oor", [
        (0.0, 100000.0, 0),   # All in range
        (0.0, 1000.0, 1),     # 99999.0 out of range
        (1.0, 100.0, 2),      # 0.5 and 99999.0 out of range
        (100.0, 200.0, 3),    # 10.0, 0.5, 99999.0 out of range
    ])
    def test_range_check_parametrized(self, spark, events_df, min_val, max_val, expected_oor):
        """Parametrized range check test."""
        qc = DataQualityChecker(spark, "events")
        qc.check_range(events_df, "amount", min_val=min_val, max_val=max_val)

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["out_of_range_count"] == str(expected_oor)


# ---------------------------------------------------------------------------
# Referential Integrity Tests
# ---------------------------------------------------------------------------

class TestReferentialIntegrity:
    """Tests for referential integrity checks."""

    def test_integrity_check_detects_orphans(self, spark, transactions_df, users_df):
        """Should detect orphan records (FK not in reference table)."""
        qc = DataQualityChecker(spark, "transactions")
        qc.check_referential_integrity(transactions_df, "user_id", users_df, "user_id")

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == "False"
        assert row["orphan_count"] == "1"  # u99 is orphan

    def test_integrity_check_passes_when_all_match(self, spark, users_df):
        """Should pass when all FKs exist in reference table."""
        data = [("t1", "u1", 100.0), ("t2", "u2", 200.0)]
        txn_df = spark.createDataFrame(data, ["transaction_id", "user_id", "amount"])

        qc = DataQualityChecker(spark, "transactions")
        qc.check_referential_integrity(txn_df, "user_id", users_df, "user_id")

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == "True"

    def test_integrity_check_empty_source(self, spark, users_df):
        """Empty source DataFrame should have zero orphans."""
        empty_df = spark.createDataFrame([], "transaction_id STRING, user_id STRING, amount DOUBLE")

        qc = DataQualityChecker(spark, "transactions")
        qc.check_referential_integrity(empty_df, "user_id", users_df, "user_id")

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert row["passed"] == "True"
        assert row["orphan_count"] == "0"


# ---------------------------------------------------------------------------
# Metrics Collection Tests
# ---------------------------------------------------------------------------

class TestMetricsCollection:
    """Tests for quality metrics collection and reporting."""

    def test_metrics_accumulate(self, spark, events_df):
        """Multiple checks should accumulate metrics."""
        qc = DataQualityChecker(spark, "events")
        qc.check_schema(events_df, ["event_id"])
        qc.check_nulls(events_df, ["event_id"], max_null_pct=5.0)
        qc.check_range(events_df, "amount", min_val=0.0, max_val=100000.0)

        metrics = qc.get_metrics_df()
        assert metrics.count() == 3

    def test_metrics_include_timestamp(self, spark, events_df):
        """Each metric should include a timestamp."""
        qc = DataQualityChecker(spark, "events")
        qc.check_nulls(events_df, ["event_id"], max_null_pct=5.0)

        metrics = qc.get_metrics_df()
        row = metrics.collect()[0]
        assert "timestamp" in row.asDict()
        assert row["timestamp"] is not None

    def test_no_metrics_returns_none(self, spark):
        """Getting metrics with no checks run should return None."""
        qc = DataQualityChecker(spark, "test")
        assert qc.get_metrics_df() is None
