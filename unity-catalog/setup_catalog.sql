-- =============================================================================
-- Unity Catalog Setup — Catalog, Schemas, and Tables
-- =============================================================================
-- Run this script against a Databricks SQL warehouse to create the full
-- catalog structure for the analytics platform.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Create Catalog
-- ---------------------------------------------------------------------------

CREATE CATALOG IF NOT EXISTS analytics_platform
COMMENT 'Analytics platform catalog — houses Bronze, Silver, Gold, and ML schemas for the medallion architecture.';

ALTER CATALOG analytics_platform SET TAGS ('environment' = 'production', 'team' = 'data-engineering');

USE CATALOG analytics_platform;

-- ---------------------------------------------------------------------------
-- 2. Create Schemas
-- ---------------------------------------------------------------------------

CREATE SCHEMA IF NOT EXISTS bronze
COMMENT 'Raw ingestion layer. Data is stored as-is from source systems with audit metadata.';

CREATE SCHEMA IF NOT EXISTS silver
COMMENT 'Cleaned and conformed layer. Data quality checks applied, PII masked, SCD Type 2 for dimensions.';

CREATE SCHEMA IF NOT EXISTS gold
COMMENT 'Business aggregation layer. Pre-computed KPIs, feature tables, and analytics-ready datasets.';

CREATE SCHEMA IF NOT EXISTS ml
COMMENT 'Machine learning schema. Model artifacts, prediction tables, and experiment metadata.';

ALTER SCHEMA bronze SET TAGS ('layer' = 'bronze', 'data_classification' = 'raw');
ALTER SCHEMA silver SET TAGS ('layer' = 'silver', 'data_classification' = 'cleaned');
ALTER SCHEMA gold   SET TAGS ('layer' = 'gold', 'data_classification' = 'aggregated');
ALTER SCHEMA ml     SET TAGS ('layer' = 'ml', 'data_classification' = 'ml_artifacts');

-- ---------------------------------------------------------------------------
-- 3. Bronze Tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS bronze.events_raw (
    event_id            STRING      NOT NULL    COMMENT 'Unique event identifier',
    user_id             STRING      NOT NULL    COMMENT 'User who generated the event',
    event_type          STRING                  COMMENT 'Type of event (page_view, click, purchase, etc.)',
    event_timestamp     STRING                  COMMENT 'Raw timestamp from the source system',
    session_id          STRING                  COMMENT 'Browser/app session identifier',
    page_url            STRING                  COMMENT 'URL of the page where the event occurred',
    referrer            STRING                  COMMENT 'Referring URL',
    device_type         STRING                  COMMENT 'Device type (desktop, mobile, tablet)',
    os                  STRING                  COMMENT 'Operating system',
    browser             STRING                  COMMENT 'Browser name',
    ip_address          STRING                  COMMENT 'Client IP address (PII — masked in Silver)',
    country             STRING                  COMMENT 'Geo-located country',
    city                STRING                  COMMENT 'Geo-located city',
    properties          STRING                  COMMENT 'JSON blob of additional event properties',
    _rescued_data       STRING                  COMMENT 'Auto Loader rescued data for unparseable records',
    _ingestion_timestamp TIMESTAMP              COMMENT 'Timestamp when the record was ingested',
    _source_file        STRING                  COMMENT 'Source file path from cloud storage',
    _batch_id           STRING                  COMMENT 'Ingestion batch identifier',
    _source_name        STRING                  COMMENT 'Source system name',
    _ingestion_date     DATE                    COMMENT 'Ingestion date partition key',
    year                INT                     COMMENT 'Year partition',
    month               INT                     COMMENT 'Month partition',
    day                 INT                     COMMENT 'Day partition'
)
USING DELTA
PARTITIONED BY (year, month, day)
COMMENT 'Raw event stream ingested via Auto Loader. No transformations applied.'
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true',
    'quality' = 'bronze'
);

CREATE TABLE IF NOT EXISTS bronze.transactions_raw (
    transaction_id          STRING      NOT NULL    COMMENT 'Unique transaction identifier',
    user_id                 STRING      NOT NULL    COMMENT 'User who made the transaction',
    amount                  DOUBLE                  COMMENT 'Transaction amount in source currency',
    currency                STRING                  COMMENT 'ISO 4217 currency code',
    transaction_timestamp   STRING                  COMMENT 'Raw timestamp from payment processor',
    transaction_status      STRING                  COMMENT 'Status: completed, pending, refunded, failed',
    payment_method          STRING                  COMMENT 'Payment method used',
    product_id              STRING                  COMMENT 'Product identifier',
    product_category        STRING                  COMMENT 'Product category',
    quantity                INT                     COMMENT 'Quantity purchased',
    _rescued_data           STRING                  COMMENT 'Auto Loader rescued data',
    _ingestion_timestamp    TIMESTAMP               COMMENT 'Ingestion timestamp',
    _source_file            STRING                  COMMENT 'Source file path',
    _batch_id               STRING                  COMMENT 'Batch identifier',
    _source_name            STRING                  COMMENT 'Source system',
    _ingestion_date         DATE                    COMMENT 'Ingestion date',
    year                    INT                     COMMENT 'Year partition',
    month                   INT                     COMMENT 'Month partition'
)
USING DELTA
PARTITIONED BY (year, month)
COMMENT 'Raw transaction data from payment processor.'
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true', 'quality' = 'bronze');

CREATE TABLE IF NOT EXISTS bronze.users_raw (
    user_id         STRING      NOT NULL    COMMENT 'Unique user identifier',
    email           STRING                  COMMENT 'User email (PII — hashed in Silver)',
    full_name       STRING                  COMMENT 'User full name (PII — hashed in Silver)',
    signup_date     STRING                  COMMENT 'Date user signed up',
    user_segment    STRING                  COMMENT 'Marketing segment: free, basic, premium, enterprise',
    country         STRING                  COMMENT 'Country of registration',
    platform        STRING                  COMMENT 'Registration platform: web, ios, android',
    age_group       STRING                  COMMENT 'Age group bucket',
    is_active       STRING                  COMMENT 'Whether the user is currently active',
    _rescued_data       STRING              COMMENT 'Auto Loader rescued data',
    _ingestion_timestamp TIMESTAMP          COMMENT 'Ingestion timestamp',
    _source_file    STRING                  COMMENT 'Source file path',
    _batch_id       STRING                  COMMENT 'Batch identifier',
    _source_name    STRING                  COMMENT 'Source system',
    _ingestion_date DATE                    COMMENT 'Ingestion date'
)
USING DELTA
COMMENT 'Raw user dimension data.'
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true', 'quality' = 'bronze');

-- ---------------------------------------------------------------------------
-- 4. Silver Tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS silver.events (
    event_id            STRING      NOT NULL    COMMENT 'Unique event identifier',
    user_id             STRING      NOT NULL    COMMENT 'User identifier',
    event_type          STRING                  COMMENT 'Standardized event type (lowercase)',
    event_timestamp     TIMESTAMP               COMMENT 'Parsed event timestamp',
    session_id          STRING                  COMMENT 'Session identifier',
    page_url            STRING                  COMMENT 'Page URL',
    referrer            STRING                  COMMENT 'Referrer URL',
    device_type         STRING                  COMMENT 'Standardized device type',
    os                  STRING                  COMMENT 'Standardized OS',
    browser             STRING                  COMMENT 'Standardized browser',
    ip_address          STRING                  COMMENT 'SHA-256 hashed IP address',
    country             STRING                  COMMENT 'Standardized country',
    city                STRING                  COMMENT 'City',
    properties          STRING                  COMMENT 'Event properties JSON',
    _ingestion_timestamp TIMESTAMP              COMMENT 'Original ingestion timestamp',
    _source_file        STRING                  COMMENT 'Source file',
    _processed_at       TIMESTAMP               COMMENT 'Silver processing timestamp',
    _silver_version     INT                     COMMENT 'Silver transformation version'
)
USING DELTA
COMMENT 'Cleaned and deduplicated events. IP addresses are SHA-256 hashed.'
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true',
    'quality' = 'silver'
);

CREATE TABLE IF NOT EXISTS silver.transactions (
    transaction_id          STRING      NOT NULL,
    user_id                 STRING      NOT NULL,
    amount                  DOUBLE,
    currency                STRING,
    transaction_timestamp   TIMESTAMP,
    transaction_status      STRING,
    payment_method          STRING,
    product_id              STRING,
    product_category        STRING,
    quantity                INT,
    _ingestion_timestamp    TIMESTAMP,
    _source_file            STRING,
    _processed_at           TIMESTAMP,
    _silver_version         INT
)
USING DELTA
COMMENT 'Cleaned transactions. Invalid amounts removed, types cast, deduplicated.'
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true', 'quality' = 'silver');

CREATE TABLE IF NOT EXISTS silver.users (
    user_id         STRING      NOT NULL    COMMENT 'User identifier',
    email           STRING                  COMMENT 'SHA-256 hashed email',
    full_name       STRING                  COMMENT 'SHA-256 hashed name',
    signup_date     DATE                    COMMENT 'Signup date',
    user_segment    STRING                  COMMENT 'Standardized segment',
    country         STRING                  COMMENT 'Standardized country',
    platform        STRING                  COMMENT 'Standardized platform',
    age_group       STRING                  COMMENT 'Age group',
    is_active       BOOLEAN                 COMMENT 'Active flag',
    _ingestion_timestamp TIMESTAMP,
    _source_file    STRING,
    _processed_at   TIMESTAMP,
    _silver_version INT,
    _effective_from TIMESTAMP               COMMENT 'SCD2 effective start',
    _effective_to   TIMESTAMP               COMMENT 'SCD2 effective end (NULL = current)',
    _is_current     BOOLEAN                 COMMENT 'SCD2 current record flag',
    _updated_at     TIMESTAMP               COMMENT 'Last update timestamp'
)
USING DELTA
COMMENT 'User dimension with SCD Type 2 history. PII columns are hashed.'
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true', 'quality' = 'silver');

CREATE TABLE IF NOT EXISTS silver.data_quality_metrics (
    table_name      STRING      COMMENT 'Table that was validated',
    check_type      STRING      COMMENT 'Type of quality check',
    column_name     STRING      COMMENT 'Column checked',
    metric_value    STRING      COMMENT 'Serialized metric value',
    passed          STRING      COMMENT 'Whether the check passed',
    timestamp       STRING      COMMENT 'When the check was run'
)
USING DELTA
COMMENT 'Data quality check results from Silver transformations.'
TBLPROPERTIES ('quality' = 'silver');

-- ---------------------------------------------------------------------------
-- 5. Gold Tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS gold.daily_active_users (
    event_date          DATE        NOT NULL,
    daily_active_users  BIGINT,
    total_events        BIGINT,
    total_sessions      BIGINT,
    unique_event_types  BIGINT,
    events_per_user     DOUBLE,
    sessions_per_user   DOUBLE,
    dau_7d_avg          DOUBLE,
    dau_30d_avg         DOUBLE,
    _gold_updated_at    TIMESTAMP
)
USING DELTA
COMMENT 'Daily active user counts with 7d and 30d rolling averages.'
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true', 'quality' = 'gold');

CREATE TABLE IF NOT EXISTS gold.daily_revenue (
    txn_date            DATE        NOT NULL,
    gross_revenue       DOUBLE,
    transaction_count   BIGINT,
    paying_users        BIGINT,
    avg_order_value     DOUBLE,
    median_order_value  DOUBLE,
    max_order_value     DOUBLE,
    revenue_per_user    DOUBLE,
    cumulative_revenue  DOUBLE,
    revenue_7d_avg      DOUBLE,
    _gold_updated_at    TIMESTAMP
)
USING DELTA
COMMENT 'Daily revenue KPIs with running totals and 7d moving averages.'
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true', 'quality' = 'gold');

CREATE TABLE IF NOT EXISTS gold.conversion_rates (
    event_date              DATE    NOT NULL,
    active_users            BIGINT,
    purchasing_users        BIGINT,
    conversion_rate         DOUBLE,
    conversion_rate_7d_avg  DOUBLE,
    _gold_updated_at        TIMESTAMP
)
USING DELTA
COMMENT 'Daily conversion rates: active users who made a purchase.'
TBLPROPERTIES ('quality' = 'gold');

CREATE TABLE IF NOT EXISTS gold.retention_cohorts (
    cohort_month            DATE,
    months_since_cohort     INT,
    retained_users          BIGINT,
    cohort_size             BIGINT,
    retention_rate          DOUBLE,
    _gold_updated_at        TIMESTAMP
)
USING DELTA
COMMENT 'Monthly retention cohort analysis.'
TBLPROPERTIES ('quality' = 'gold');

CREATE TABLE IF NOT EXISTS gold.customer_ltv (
    user_id                     STRING      NOT NULL,
    total_revenue               DOUBLE,
    total_orders                BIGINT,
    avg_order_value             DOUBLE,
    first_purchase              TIMESTAMP,
    last_purchase               TIMESTAMP,
    customer_lifespan_days      INT,
    customer_lifespan_months    DOUBLE,
    purchase_frequency_monthly  DOUBLE,
    estimated_clv               DOUBLE,
    user_segment                STRING,
    signup_date                 DATE,
    clv_segment                 STRING,
    _gold_updated_at            TIMESTAMP
)
USING DELTA
COMMENT 'Customer Lifetime Value estimates per user.'
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true', 'quality' = 'gold');

CREATE TABLE IF NOT EXISTS gold.feature_table (
    user_id                     STRING      NOT NULL,
    event_date                  DATE        NOT NULL,
    daily_event_count           BIGINT,
    daily_unique_event_types    BIGINT,
    event_count_7d              BIGINT,
    event_count_30d             BIGINT,
    event_count_90d             BIGINT,
    purchase_count_7d           BIGINT,
    purchase_count_30d          BIGINT,
    purchase_count_90d          BIGINT,
    avg_daily_events_7d         DOUBLE,
    avg_daily_events_30d        DOUBLE,
    avg_daily_events_90d        DOUBLE,
    total_transactions          BIGINT,
    total_spend                 DOUBLE,
    avg_transaction_value       DOUBLE,
    max_transaction_value       DOUBLE,
    stddev_transaction_value    DOUBLE,
    unique_categories_purchased BIGINT,
    customer_tenure_days        INT,
    refund_count                BIGINT,
    refund_rate                 DOUBLE,
    account_age_days            INT,
    hour_of_day                 INT,
    day_of_week                 INT,
    is_weekend                  INT,
    is_business_hours           INT,
    days_since_last_event       INT,
    user_segment_index          DOUBLE,
    country_index               DOUBLE,
    platform_index              DOUBLE
)
USING DELTA
COMMENT 'ML feature table — user-level features for model training and inference.'
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true',
    'quality' = 'gold'
);

CREATE TABLE IF NOT EXISTS gold.predictions (
    user_id                 STRING      NOT NULL,
    event_date              DATE        NOT NULL,
    prediction              DOUBLE,
    probability             DOUBLE,
    prediction_label        STRING,
    prediction_timestamp    TIMESTAMP,
    model_name              STRING,
    model_version           STRING
)
USING DELTA
COMMENT 'Model predictions — batch inference output.'
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true', 'quality' = 'gold');

CREATE TABLE IF NOT EXISTS gold.training_labels (
    user_id     STRING      NOT NULL    COMMENT 'User identifier',
    event_date  DATE        NOT NULL    COMMENT 'Label date',
    label       DOUBLE      NOT NULL    COMMENT 'Binary label: 1.0 = purchased within 7 days, 0.0 = did not'
)
USING DELTA
COMMENT 'Training labels for the purchase prediction model.'
TBLPROPERTIES ('quality' = 'gold');

-- ---------------------------------------------------------------------------
-- 6. ML Schema Tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ml.drift_metrics (
    feature             STRING,
    psi                 DOUBLE      COMMENT 'Population Stability Index',
    drift_status        STRING      COMMENT 'no_drift, moderate_drift, significant_drift',
    reference_mean      DOUBLE,
    current_mean        DOUBLE,
    analysis_timestamp  TIMESTAMP,
    model_version       STRING
)
USING DELTA
COMMENT 'Feature drift detection metrics (PSI).'
TBLPROPERTIES ('quality' = 'ml');

-- ---------------------------------------------------------------------------
-- Done
-- ---------------------------------------------------------------------------

SELECT 'Unity Catalog setup complete.' AS status;
