-- =============================================================================
-- Unity Catalog Access Policies
-- =============================================================================
-- Fine-grained access control with RBAC, row-level security, column masking,
-- and dynamic views.
-- =============================================================================

USE CATALOG analytics_platform;

-- ---------------------------------------------------------------------------
-- 1. Create Groups
-- ---------------------------------------------------------------------------

-- Data Engineering — full access to Bronze and Silver
CREATE GROUP IF NOT EXISTS data_engineering;

-- Data Science — read access to Silver/Gold, write to ML schema
CREATE GROUP IF NOT EXISTS data_science;

-- Data Analysts — read-only access to Gold tables
CREATE GROUP IF NOT EXISTS data_analysts;

-- ML Engineering — manage ML schema and model serving
CREATE GROUP IF NOT EXISTS ml_engineering;

-- Auditors — read-only, no access to PII
CREATE GROUP IF NOT EXISTS auditors;

-- ---------------------------------------------------------------------------
-- 2. Catalog-Level Grants
-- ---------------------------------------------------------------------------

-- Data Engineering: full catalog usage
GRANT USE CATALOG ON CATALOG analytics_platform TO data_engineering;
GRANT USE CATALOG ON CATALOG analytics_platform TO data_science;
GRANT USE CATALOG ON CATALOG analytics_platform TO data_analysts;
GRANT USE CATALOG ON CATALOG analytics_platform TO ml_engineering;
GRANT USE CATALOG ON CATALOG analytics_platform TO auditors;

-- ---------------------------------------------------------------------------
-- 3. Schema-Level Grants
-- ---------------------------------------------------------------------------

-- Data Engineering — full access to Bronze and Silver
GRANT USE SCHEMA ON SCHEMA bronze TO data_engineering;
GRANT CREATE TABLE ON SCHEMA bronze TO data_engineering;
GRANT MODIFY ON SCHEMA bronze TO data_engineering;
GRANT SELECT ON SCHEMA bronze TO data_engineering;

GRANT USE SCHEMA ON SCHEMA silver TO data_engineering;
GRANT CREATE TABLE ON SCHEMA silver TO data_engineering;
GRANT MODIFY ON SCHEMA silver TO data_engineering;
GRANT SELECT ON SCHEMA silver TO data_engineering;

GRANT USE SCHEMA ON SCHEMA gold TO data_engineering;
GRANT CREATE TABLE ON SCHEMA gold TO data_engineering;
GRANT MODIFY ON SCHEMA gold TO data_engineering;
GRANT SELECT ON SCHEMA gold TO data_engineering;

-- Data Science — read Silver/Gold, write ML
GRANT USE SCHEMA ON SCHEMA silver TO data_science;
GRANT SELECT ON SCHEMA silver TO data_science;

GRANT USE SCHEMA ON SCHEMA gold TO data_science;
GRANT SELECT ON SCHEMA gold TO data_science;

GRANT USE SCHEMA ON SCHEMA ml TO data_science;
GRANT CREATE TABLE ON SCHEMA ml TO data_science;
GRANT MODIFY ON SCHEMA ml TO data_science;
GRANT SELECT ON SCHEMA ml TO data_science;

-- Data Analysts — read-only Gold
GRANT USE SCHEMA ON SCHEMA gold TO data_analysts;
GRANT SELECT ON SCHEMA gold TO data_analysts;

-- ML Engineering — read Gold, full ML
GRANT USE SCHEMA ON SCHEMA gold TO ml_engineering;
GRANT SELECT ON SCHEMA gold TO ml_engineering;

GRANT USE SCHEMA ON SCHEMA ml TO ml_engineering;
GRANT CREATE TABLE ON SCHEMA ml TO ml_engineering;
GRANT MODIFY ON SCHEMA ml TO ml_engineering;
GRANT SELECT ON SCHEMA ml TO ml_engineering;

-- Auditors — read-only on Silver and Gold (will use masked views)
GRANT USE SCHEMA ON SCHEMA silver TO auditors;
GRANT USE SCHEMA ON SCHEMA gold TO auditors;

-- ---------------------------------------------------------------------------
-- 4. Table-Level Grants
-- ---------------------------------------------------------------------------

-- Revoke direct access to PII-containing tables for auditors
-- They will use masked views instead
REVOKE SELECT ON TABLE silver.users FROM auditors;

-- Grant auditors access to non-PII Silver tables
GRANT SELECT ON TABLE silver.events TO auditors;
GRANT SELECT ON TABLE silver.transactions TO auditors;

-- Grant all Gold tables to analysts
GRANT SELECT ON TABLE gold.daily_active_users TO data_analysts;
GRANT SELECT ON TABLE gold.daily_revenue TO data_analysts;
GRANT SELECT ON TABLE gold.conversion_rates TO data_analysts;
GRANT SELECT ON TABLE gold.retention_cohorts TO data_analysts;
GRANT SELECT ON TABLE gold.customer_ltv TO data_analysts;

-- ---------------------------------------------------------------------------
-- 5. Row-Level Security
-- ---------------------------------------------------------------------------

-- Row filter function: restrict data analysts to their assigned country
CREATE OR REPLACE FUNCTION silver.country_row_filter(country_value STRING)
RETURNS BOOLEAN
COMMENT 'Row-level security: users only see data for their assigned country'
RETURN
    IS_ACCOUNT_GROUP_MEMBER('data_engineering')
    OR IS_ACCOUNT_GROUP_MEMBER('data_science')
    OR (
        IS_ACCOUNT_GROUP_MEMBER('data_analysts')
        AND country_value = CURRENT_USER_ATTRIBUTE('country')
    );

-- Apply row filter to events table
ALTER TABLE silver.events SET ROW FILTER silver.country_row_filter ON (country);

-- Apply row filter to transactions (via user join, or directly if country exists)
-- For transactions, we apply a simpler date-based filter for analysts
CREATE OR REPLACE FUNCTION silver.analyst_date_filter(txn_timestamp TIMESTAMP)
RETURNS BOOLEAN
COMMENT 'Row-level security: analysts can only see last 365 days of data'
RETURN
    IS_ACCOUNT_GROUP_MEMBER('data_engineering')
    OR IS_ACCOUNT_GROUP_MEMBER('data_science')
    OR (
        IS_ACCOUNT_GROUP_MEMBER('data_analysts')
        AND txn_timestamp >= DATE_SUB(CURRENT_DATE(), 365)
    );

ALTER TABLE silver.transactions SET ROW FILTER silver.analyst_date_filter ON (transaction_timestamp);

-- ---------------------------------------------------------------------------
-- 6. Column Masking Functions
-- ---------------------------------------------------------------------------

-- Mask email: show domain only for non-engineering roles
CREATE OR REPLACE FUNCTION silver.mask_email(email_value STRING)
RETURNS STRING
COMMENT 'Column mask: shows full email for data engineering, domain only for others'
RETURN
    CASE
        WHEN IS_ACCOUNT_GROUP_MEMBER('data_engineering') THEN email_value
        WHEN email_value IS NULL THEN NULL
        ELSE CONCAT('***@', SUBSTRING_INDEX(email_value, '@', -1))
    END;

-- Mask name: show initials only for non-engineering roles
CREATE OR REPLACE FUNCTION silver.mask_name(name_value STRING)
RETURNS STRING
COMMENT 'Column mask: shows full name for data engineering, initials for others'
RETURN
    CASE
        WHEN IS_ACCOUNT_GROUP_MEMBER('data_engineering') THEN name_value
        WHEN name_value IS NULL THEN NULL
        ELSE CONCAT(
            UPPER(LEFT(name_value, 1)),
            '***',
            UPPER(LEFT(SUBSTRING_INDEX(name_value, ' ', -1), 1)),
            '***'
        )
    END;

-- Mask IP address: show /24 subnet only
CREATE OR REPLACE FUNCTION silver.mask_ip(ip_value STRING)
RETURNS STRING
COMMENT 'Column mask: shows subnet /24 for non-engineering roles'
RETURN
    CASE
        WHEN IS_ACCOUNT_GROUP_MEMBER('data_engineering') THEN ip_value
        WHEN ip_value IS NULL THEN NULL
        ELSE CONCAT(
            SUBSTRING_INDEX(ip_value, '.', 3),
            '.xxx'
        )
    END;

-- Mask monetary amounts: show rounded for analysts
CREATE OR REPLACE FUNCTION gold.mask_revenue(amount_value DOUBLE)
RETURNS DOUBLE
COMMENT 'Column mask: shows exact amounts for engineering/science, rounded for analysts'
RETURN
    CASE
        WHEN IS_ACCOUNT_GROUP_MEMBER('data_engineering')
             OR IS_ACCOUNT_GROUP_MEMBER('data_science') THEN amount_value
        ELSE ROUND(amount_value, -2)  -- Round to nearest hundred
    END;

-- Apply column masks
ALTER TABLE silver.users ALTER COLUMN email SET MASK silver.mask_email;
ALTER TABLE silver.users ALTER COLUMN full_name SET MASK silver.mask_name;
ALTER TABLE silver.events ALTER COLUMN ip_address SET MASK silver.mask_ip;

-- ---------------------------------------------------------------------------
-- 7. Dynamic Views for Auditors
-- ---------------------------------------------------------------------------

-- Auditor view for users: no PII, only aggregated info
CREATE OR REPLACE VIEW silver.v_users_audit AS
SELECT
    user_id,
    -- Hashed PII (already hashed in Silver, but double-masked for auditors)
    'REDACTED' AS email,
    'REDACTED' AS full_name,
    signup_date,
    user_segment,
    country,
    platform,
    age_group,
    is_active,
    _is_current,
    _effective_from,
    _effective_to
FROM silver.users
WHERE _is_current = true;

GRANT SELECT ON VIEW silver.v_users_audit TO auditors;

-- Auditor view for transactions: no exact amounts
CREATE OR REPLACE VIEW gold.v_revenue_audit AS
SELECT
    txn_date,
    transaction_count,
    paying_users,
    ROUND(avg_order_value, -1) AS avg_order_value_approx,
    ROUND(gross_revenue, -2) AS gross_revenue_approx,
    _gold_updated_at
FROM gold.daily_revenue;

GRANT SELECT ON VIEW gold.v_revenue_audit TO auditors;

-- Analyst view with pre-filtered CLV data
CREATE OR REPLACE VIEW gold.v_customer_segments AS
SELECT
    user_segment,
    clv_segment,
    COUNT(*) AS user_count,
    ROUND(AVG(estimated_clv), 2) AS avg_clv,
    ROUND(AVG(total_orders), 1) AS avg_orders,
    ROUND(AVG(avg_order_value), 2) AS avg_order_value,
    ROUND(AVG(customer_lifespan_months), 1) AS avg_lifespan_months
FROM gold.customer_ltv
GROUP BY user_segment, clv_segment;

GRANT SELECT ON VIEW gold.v_customer_segments TO data_analysts;
GRANT SELECT ON VIEW gold.v_customer_segments TO auditors;

-- ---------------------------------------------------------------------------
-- 8. Verify Grants
-- ---------------------------------------------------------------------------

-- Show grants on catalog
SHOW GRANTS ON CATALOG analytics_platform;

-- Show grants on schemas
SHOW GRANTS ON SCHEMA bronze;
SHOW GRANTS ON SCHEMA silver;
SHOW GRANTS ON SCHEMA gold;
SHOW GRANTS ON SCHEMA ml;

-- Show grants on specific tables
SHOW GRANTS ON TABLE silver.users;
SHOW GRANTS ON TABLE gold.daily_revenue;

SELECT 'Access policies configured successfully.' AS status;
