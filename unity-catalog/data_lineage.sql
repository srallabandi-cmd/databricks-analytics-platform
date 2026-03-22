-- =============================================================================
-- Data Lineage Tracking Queries
-- =============================================================================
-- Queries for tracking data lineage, table dependencies, and column-level
-- lineage within Unity Catalog.
-- =============================================================================

USE CATALOG analytics_platform;

-- ---------------------------------------------------------------------------
-- 1. Table-Level Lineage — Which tables read from / write to each other
-- ---------------------------------------------------------------------------

-- List all tables across all schemas with metadata
SELECT
    table_catalog,
    table_schema,
    table_name,
    table_type,
    data_source_format,
    comment,
    created,
    created_by,
    last_altered,
    last_altered_by
FROM
    information_schema.tables
WHERE
    table_catalog = 'analytics_platform'
ORDER BY
    table_schema,
    table_name;

-- ---------------------------------------------------------------------------
-- 2. Column-Level Lineage — Detailed column metadata
-- ---------------------------------------------------------------------------

-- All columns with types and comments
SELECT
    table_schema,
    table_name,
    column_name,
    ordinal_position,
    data_type,
    is_nullable,
    comment
FROM
    information_schema.columns
WHERE
    table_catalog = 'analytics_platform'
ORDER BY
    table_schema,
    table_name,
    ordinal_position;

-- ---------------------------------------------------------------------------
-- 3. Identify PII Columns (columns that are SHA-256 hashed in Silver)
-- ---------------------------------------------------------------------------

SELECT
    table_schema,
    table_name,
    column_name,
    comment
FROM
    information_schema.columns
WHERE
    table_catalog = 'analytics_platform'
    AND (
        comment LIKE '%hash%'
        OR comment LIKE '%PII%'
        OR comment LIKE '%SHA%'
        OR column_name IN ('email', 'full_name', 'ip_address')
    )
ORDER BY
    table_schema,
    table_name;

-- ---------------------------------------------------------------------------
-- 4. Table Dependencies — Medallion Architecture Flow
-- ---------------------------------------------------------------------------

-- Bronze -> Silver dependencies (manual mapping since Unity Catalog
-- lineage is captured automatically via query history in Databricks)
SELECT
    'bronze' AS source_layer,
    source_table,
    'silver' AS target_layer,
    target_table,
    transformation
FROM VALUES
    ('events_raw',       'events',           'Dedup, type cast, PII mask, null filter'),
    ('transactions_raw', 'transactions',     'Dedup, type cast, range validation, standardize'),
    ('users_raw',        'users',            'Dedup, type cast, PII mask, SCD Type 2 merge')
AS lineage(source_table, target_table, transformation);

-- Silver -> Gold dependencies
SELECT
    'silver' AS source_layer,
    source_table,
    'gold' AS target_layer,
    target_table,
    transformation
FROM VALUES
    ('events',       'daily_active_users',  'Group by date, count distinct users, 7d/30d avg'),
    ('transactions', 'daily_revenue',       'Group by date, sum/avg/count, running totals'),
    ('events',       'conversion_rates',    'Join events DAU with transactions purchasers'),
    ('transactions', 'conversion_rates',    'Join transactions purchasers with events DAU'),
    ('events',       'retention_cohorts',   'Cohort by signup month, monthly retention'),
    ('users',        'retention_cohorts',   'User signup date for cohort assignment'),
    ('transactions', 'customer_ltv',        'User-level aggregation, CLV formula'),
    ('users',        'customer_ltv',        'User segment for CLV segmentation'),
    ('events',       'feature_table',       'Time features, rolling aggregations'),
    ('transactions', 'feature_table',       'Transaction aggregation features'),
    ('users',        'feature_table',       'User demographic features')
AS lineage(source_table, target_table, transformation);

-- ---------------------------------------------------------------------------
-- 5. Table Freshness Check
-- ---------------------------------------------------------------------------

-- Check when each table was last updated (via Delta history)
-- Note: Run these individually per table or in a notebook loop

-- Events freshness
SELECT
    'silver.events' AS table_name,
    MAX(event_timestamp) AS latest_record,
    COUNT(*) AS total_rows
FROM silver.events;

-- Transactions freshness
SELECT
    'silver.transactions' AS table_name,
    MAX(transaction_timestamp) AS latest_record,
    COUNT(*) AS total_rows
FROM silver.transactions;

-- Gold tables freshness
SELECT
    'gold.daily_active_users' AS table_name,
    MAX(event_date) AS latest_date,
    COUNT(*) AS total_rows
FROM gold.daily_active_users;

SELECT
    'gold.daily_revenue' AS table_name,
    MAX(txn_date) AS latest_date,
    COUNT(*) AS total_rows
FROM gold.daily_revenue;

-- ---------------------------------------------------------------------------
-- 6. Delta Table History — Audit Trail
-- ---------------------------------------------------------------------------

-- Recent operations on Bronze events
DESCRIBE HISTORY bronze.events_raw LIMIT 20;

-- Recent operations on Silver events
DESCRIBE HISTORY silver.events LIMIT 20;

-- Recent operations on Gold feature table
DESCRIBE HISTORY gold.feature_table LIMIT 20;

-- ---------------------------------------------------------------------------
-- 7. Schema Evolution Tracking
-- ---------------------------------------------------------------------------

-- Compare current schema against expected schema
-- Useful for detecting schema drift from source systems

SELECT
    c.table_schema,
    c.table_name,
    c.column_name,
    c.data_type,
    c.ordinal_position,
    CASE
        WHEN c.column_name LIKE '_%' THEN 'metadata'
        ELSE 'business'
    END AS column_category
FROM
    information_schema.columns c
WHERE
    c.table_catalog = 'analytics_platform'
    AND c.table_schema = 'bronze'
ORDER BY
    c.table_name,
    c.ordinal_position;

-- ---------------------------------------------------------------------------
-- 8. Table Size and Storage Metrics
-- ---------------------------------------------------------------------------

-- Table row counts across all schemas
SELECT
    'bronze.events_raw' AS table_name, COUNT(*) AS row_count FROM bronze.events_raw
UNION ALL
SELECT
    'bronze.transactions_raw', COUNT(*) FROM bronze.transactions_raw
UNION ALL
SELECT
    'bronze.users_raw', COUNT(*) FROM bronze.users_raw
UNION ALL
SELECT
    'silver.events', COUNT(*) FROM silver.events
UNION ALL
SELECT
    'silver.transactions', COUNT(*) FROM silver.transactions
UNION ALL
SELECT
    'silver.users', COUNT(*) FROM silver.users
UNION ALL
SELECT
    'gold.daily_active_users', COUNT(*) FROM gold.daily_active_users
UNION ALL
SELECT
    'gold.daily_revenue', COUNT(*) FROM gold.daily_revenue
UNION ALL
SELECT
    'gold.feature_table', COUNT(*) FROM gold.feature_table;

-- ---------------------------------------------------------------------------
-- 9. Query-Based Lineage (from system tables — requires Premium workspace)
-- ---------------------------------------------------------------------------

-- Uncomment if system.access.table_lineage is available:
--
-- SELECT
--     source_table_full_name,
--     target_table_full_name,
--     event_time,
--     entity_type
-- FROM system.access.table_lineage
-- WHERE
--     source_table_full_name LIKE 'analytics_platform.%'
--     OR target_table_full_name LIKE 'analytics_platform.%'
-- ORDER BY event_time DESC
-- LIMIT 100;

-- Column-level lineage (system tables):
--
-- SELECT
--     source_table_full_name,
--     source_column_name,
--     target_table_full_name,
--     target_column_name,
--     event_time
-- FROM system.access.column_lineage
-- WHERE
--     source_table_full_name LIKE 'analytics_platform.%'
-- ORDER BY event_time DESC
-- LIMIT 100;
