-- ===================================================================
-- Equities Data Import Script
-- ===================================================================
-- Documentation: See docs/column_mapping_reference.md for column aliases
-- Usage: psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql
\echo 'Starting equities data import...'

-- ===================================================================
-- SESSION-LEVEL TUNING FOR BULK IMPORT
-- ===================================================================
-- These settings optimize PostgreSQL for bulk data import operations.
-- They will be reset when the session ends.
-- WARNING: synchronous_commit = OFF should only be used for imports, not production!

SET work_mem = '256MB'; -- Increase memory for sorting/hashing operations
SET maintenance_work_mem = '512MB'; -- Increase memory for maintenance operations
SET synchronous_commit = OFF; -- Defer WAL writes (faster, but less durable during import)
SET checkpoint_completion_target = 0.9; -- Spread checkpoint I/O over longer period

\echo 'Session tuning applied for bulk import optimization.'

DO
$$
    BEGIN
        RAISE NOTICE 'Import started at %', NOW();
    END
$$;

-- Show current table status
SELECT 'Current equities table row count:' AS status, COUNT(*) AS row_count
FROM equities;

-- ===================================================================
-- HELPER FUNCTIONS
-- ===================================================================

-- ===================================================================
-- HELPER FUNCTION: Month Abbreviation to Number
-- ===================================================================
-- Converts a 3-letter month abbreviation to its numeric value (1-12)
CREATE OR REPLACE FUNCTION month_abbrev_to_number(month_abbrev TEXT)
    RETURNS INTEGER AS
$$
BEGIN
    RETURN CASE UPPER(LEFT(TRIM(COALESCE(month_abbrev, '')), 3))
               WHEN 'JAN' THEN 1
               WHEN 'FEB' THEN 2
               WHEN 'MAR' THEN 3
               WHEN 'APR' THEN 4
               WHEN 'MAY' THEN 5
               WHEN 'JUN' THEN 6
               WHEN 'JUL' THEN 7
               WHEN 'AUG' THEN 8
               WHEN 'SEP' THEN 9
               WHEN 'OCT' THEN 10
               WHEN 'NOV' THEN 11
               WHEN 'DEC' THEN 12
        END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Get Expected Reporting Lag Days
-- ===================================================================
-- Returns the typical number of days between period end and earnings release
CREATE OR REPLACE FUNCTION get_expected_reporting_lag_days(earnings_report_frequency TEXT)
    RETURNS INTEGER AS
$$
BEGIN
    RETURN CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'QUARTERLY')))
               WHEN 'QUARTERLY' THEN 45
               WHEN 'SEMI-ANNUALLY' THEN 60
               WHEN 'SEMI-ANNUAL' THEN 60
               WHEN 'ANNUALLY' THEN 90
               WHEN 'ANNUAL' THEN 90
               ELSE 45
        END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Converts TEXT to NUMERIC, treating common non-numeric patterns as NULL
CREATE OR REPLACE FUNCTION text_to_numeric_safe(input_text TEXT)
    RETURNS NUMERIC AS
$$
SELECT CASE
           WHEN input_text IS NULL
               OR TRIM(input_text) IN ('', '-', '--', 'N/A', 'NA', 'NULL', 'NONE', 'n/a', 'na', 'null', 'none')
               THEN NULL
           WHEN TRIM(input_text) ~ '^-?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
               THEN TRIM(input_text)::NUMERIC
           END AS result
$$ LANGUAGE SQL IMMUTABLE
                PARALLEL SAFE;

-- Converts TEXT to DATE safely, returns NULL for invalid input
CREATE OR REPLACE FUNCTION text_to_date_safe(input_text TEXT, date_format TEXT DEFAULT 'YYYY-MM-DD')
    RETURNS DATE AS
$$
BEGIN
    IF input_text IS NULL OR TRIM(input_text) IN ('', '-', '--', 'N/A', 'NA', 'NULL', 'NONE') THEN
        RETURN NULL;
    END IF;
    RETURN TO_DATE(TRIM(input_text), date_format);
EXCEPTION
    WHEN OTHERS THEN
        RETURN NULL;
END;
$$ LANGUAGE plpgsql IMMUTABLE
                    STRICT;

-- ===================================================================
-- HELPER FUNCTION: Parse FY End to Date
-- ===================================================================
CREATE OR REPLACE FUNCTION parse_fiscal_year_end_date(fy_end_text TEXT)
    RETURNS DATE AS
$$
DECLARE
    month_name TEXT;
    year_text  TEXT;
    month_num  INTEGER;
    year_value INTEGER;
BEGIN
    IF fy_end_text IS NULL OR TRIM(fy_end_text) = '' THEN
        RETURN NULL;
    END IF;

    fy_end_text := TRIM(fy_end_text);
    month_name := SPLIT_PART(fy_end_text, ' ', 1);
    year_text := SPLIT_PART(fy_end_text, ' ', 2);

    -- Validate year format and range
    IF year_text !~ '^\d{4}$' THEN
        RETURN NULL;
    END IF;

    year_value := year_text::INTEGER;
    IF year_value < 1900 OR year_value > 2100 THEN
        RETURN NULL;
    END IF;

    month_num := month_abbrev_to_number(month_name);
    IF month_num IS NULL THEN
        RETURN NULL;
    END IF;

    RETURN (MAKE_DATE(year_value, month_num, 1) + INTERVAL '1 month - 1 day')::DATE;
END;
$$ LANGUAGE plpgsql IMMUTABLE
                    STRICT;

-- ===================================================================
-- HELPER FUNCTION: Convert Frequency to Interval Months
-- ===================================================================
CREATE OR REPLACE FUNCTION frequency_to_months(
    earnings_report_frequency TEXT,
    fy_end_date               DATE DEFAULT NULL,
    next_fy_end_date          DATE DEFAULT NULL
)
    RETURNS INTEGER AS
$$
DECLARE
    fy_range_months INTEGER;
BEGIN
    -- Calculate the fiscal year range in months (should always be 12)
    IF fy_end_date IS NOT NULL AND next_fy_end_date IS NOT NULL THEN
        fy_range_months := ((EXTRACT(YEAR FROM next_fy_end_date) - EXTRACT(YEAR FROM fy_end_date)) * 12
            + (EXTRACT(MONTH FROM next_fy_end_date) - EXTRACT(MONTH FROM fy_end_date)))::INTEGER;
    ELSE
        -- Default to standard 12-month fiscal year
        fy_range_months := 12;
    END IF;

    -- Derive reporting interval as a divisor of the fiscal year range
    RETURN CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'QUARTERLY')))
        -- Quarterly: FY range / 4 reporting periods
               WHEN 'QUARTERLY' THEN fy_range_months / 4
        -- Semi-Annual: FY range / 2 reporting periods
               WHEN 'SEMI-ANNUALLY' THEN fy_range_months / 2
               WHEN 'SEMI-ANNUAL' THEN fy_range_months / 2
        -- Annual: Full FY range (1 reporting period)
               WHEN 'ANNUALLY' THEN fy_range_months
               WHEN 'ANNUAL' THEN fy_range_months
        -- Default to quarterly (FY range / 4)
               ELSE fy_range_months / 4
        END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Convert Interval Months to Frequency Text
-- ===================================================================
CREATE OR REPLACE FUNCTION months_to_frequency(interval_months INTEGER)
    RETURNS TEXT AS
$$
BEGIN
    RETURN CASE
               WHEN interval_months <= 3 THEN 'Quarterly'
               WHEN interval_months <= 6 THEN 'Semi-Annually'
               ELSE 'Annually'
        END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Derive Earnings Report Frequency
-- ===================================================================
CREATE OR REPLACE FUNCTION derive_earnings_report_frequency(
    income_statement_report_date DATE,
    fy_end_date                  DATE
)
    RETURNS TEXT AS
$$
DECLARE
    months_diff INTEGER;
BEGIN
    IF income_statement_report_date IS NULL OR fy_end_date IS NULL THEN
        RETURN 'Quarterly';
    END IF;

    months_diff := ABS(
            (EXTRACT(YEAR FROM income_statement_report_date) - EXTRACT(YEAR FROM fy_end_date)) * 12
                + (EXTRACT(MONTH FROM income_statement_report_date) - EXTRACT(MONTH FROM fy_end_date))
                   )::INTEGER;

    -- Normalize to 1-12 range
    months_diff := COALESCE(NULLIF(months_diff % 12, 0), 12);

    -- Determine frequency: check if months align with semi-annual or quarterly
    RETURN CASE
               WHEN months_diff IN (6, 12) THEN 'Semi-Annually'
               ELSE 'Quarterly'
        END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- Unified Fiscal Date Calculator
-- Derives all calculations based on FY End Date reporting ranges
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_fiscal_info(
    reference_date                DATE,
    fy_end_date                   DATE,
    input_earnings_frequency      TEXT DEFAULT NULL,
    OUT fiscal_month              INTEGER,
    OUT fiscal_quarter            INTEGER,
    OUT fiscal_year               INTEGER,
    OUT next_quarter              INTEGER,
    OUT next_quarter_year         INTEGER,
    OUT reporting_interval        INTEGER,
    OUT earnings_report_frequency TEXT,
    OUT next_earnings_report_type TEXT
) AS
$$
DECLARE
    next_fy_end_date    DATE;
    fy_range_months     INTEGER;
    months_since_fy_end INTEGER;
    interval_months     INTEGER;
    periods_per_year    INTEGER;
    current_period      INTEGER;
    next_period         INTEGER;
BEGIN
    IF reference_date IS NULL OR fy_end_date IS NULL THEN
        RETURN;
    END IF;

    -- Calculate Next FY End Date (defines the reporting range)
    next_fy_end_date := (fy_end_date + INTERVAL '1 year')::DATE;

    -- Calculate fiscal year range in months (the base for all interval calculations)
    fy_range_months := ((EXTRACT(YEAR FROM next_fy_end_date) - EXTRACT(YEAR FROM fy_end_date)) * 12
        + (EXTRACT(MONTH FROM next_fy_end_date) - EXTRACT(MONTH FROM fy_end_date)))::INTEGER;

    -- Determine earnings frequency
    earnings_report_frequency := COALESCE(NULLIF(TRIM(input_earnings_frequency), ''),
                                          derive_earnings_report_frequency(reference_date, fy_end_date));

    -- Derive interval months based on FY range
    interval_months := CASE UPPER(TRIM(earnings_report_frequency))
                           WHEN 'QUARTERLY' THEN fy_range_months / 4
                           WHEN 'SEMI-ANNUALLY' THEN fy_range_months / 2
                           WHEN 'SEMI-ANNUAL' THEN fy_range_months / 2
                           WHEN 'ANNUALLY' THEN fy_range_months
                           WHEN 'ANNUAL' THEN fy_range_months
                           ELSE fy_range_months / 4
        END;

    reporting_interval := interval_months;

    -- Calculate periods per fiscal year based on the FY range
    periods_per_year := fy_range_months / interval_months;

    -- Calculate months since fiscal year end
    months_since_fy_end := ((EXTRACT(YEAR FROM reference_date) - EXTRACT(YEAR FROM fy_end_date)) * 12
        + (EXTRACT(MONTH FROM reference_date) - EXTRACT(MONTH FROM fy_end_date)))::INTEGER;

    -- Fiscal month (1-12) derived from position within FY range
    fiscal_month := ((months_since_fy_end - 1) % fy_range_months) + 1;
    IF fiscal_month <= 0 THEN
        fiscal_month := fiscal_month + fy_range_months;
    END IF;

    -- Fiscal quarter derived from fiscal month relative to FY range
    -- Each quarter represents (fy_range_months / 4) months
    fiscal_quarter := CEIL(fiscal_month / (fy_range_months / 4.0))::INTEGER;

    -- Ensure fiscal_quarter stays within 1-4 range
    IF fiscal_quarter > 4 THEN
        fiscal_quarter := 4;
    END IF;

    -- Calculate current reporting period within the fiscal year
    current_period := CEIL(fiscal_month / interval_months::NUMERIC)::INTEGER;
    IF current_period > periods_per_year THEN
        current_period := periods_per_year;
    END IF;

    -- Calculate next reporting period
    next_period := current_period + 1;
    IF next_period > periods_per_year THEN
        next_period := 1;
    END IF;

    -- Convert next_period back to quarter for output
    -- Next quarter is derived from which reporting period we're moving to
    next_quarter := CASE
                        WHEN periods_per_year = 4 THEN next_period -- Quarterly
                        WHEN periods_per_year = 2 THEN next_period * 2 -- Semi-annual (Q2 or Q4)
                        WHEN periods_per_year = 1 THEN 4 -- Annual (always Q4/full year)
                        ELSE ((fiscal_quarter + (interval_months / (fy_range_months / 4)) - 1) % 4) + 1
        END;

    -- Fiscal year calculations based on FY range
    fiscal_year := EXTRACT(YEAR FROM fy_end_date)::INTEGER + 1 + ((months_since_fy_end - 1) / fy_range_months);

    -- Next quarter year
    next_quarter_year := CASE
                             WHEN next_period = 1 AND current_period = periods_per_year THEN fiscal_year + 1
                             ELSE fiscal_year
        END;

    -- Report type derived from reporting periods and FY range
    next_earnings_report_type := CASE
        -- Full year if annual reporting OR if next period completes the FY
                                     WHEN interval_months = fy_range_months THEN 'Full Year'
                                     WHEN next_period = periods_per_year AND periods_per_year > 1 THEN 'Full Year'
        -- Half year for semi-annual mid-year report
                                     WHEN interval_months = fy_range_months / 2 AND next_period = 1 THEN 'Half Year'
                                     ELSE 'Interim'
        END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next Income Statement Report Date
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_next_income_statement_report_date(
    income_statement_report_date DATE,
    earnings_report_frequency    TEXT
)
    RETURNS DATE AS
$$
BEGIN
    IF income_statement_report_date IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN (income_statement_report_date +
            (frequency_to_months(earnings_report_frequency) || ' months')::INTERVAL)::DATE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next Fiscal Year End Date
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_next_fy_end_date(fy_end_date DATE)
    RETURNS DATE AS
$$
BEGIN
    IF fy_end_date IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN (fy_end_date + INTERVAL '1 year')::DATE;
END;
$$ LANGUAGE plpgsql IMMUTABLE
                    STRICT;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next Fiscal Quarter
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_next_fiscal_quarter(
    next_earnings_date           DATE,
    income_statement_report_date DATE,
    fy_end_date                  DATE,
    earnings_report_frequency    TEXT DEFAULT 'Quarterly'
)
    RETURNS INTEGER AS
$$
DECLARE
    next_fy_end_date      DATE;
    reference_date        DATE;
    fy_range_months       INTEGER;
    interval_months       INTEGER;
    months_into_fy        INTEGER;
    next_period_end_month INTEGER;
    fiscal_quarter        INTEGER;
BEGIN
    -- Return NULL if essential dates are missing
    IF fy_end_date IS NULL THEN
        RETURN NULL;
    END IF;

    -- Determine the reference date: prefer Next Earnings, fallback to Income Statement + interval
    IF next_earnings_date IS NOT NULL THEN
        reference_date := next_earnings_date;
    ELSIF income_statement_report_date IS NOT NULL THEN
        -- Estimate next report date by adding the reporting interval
        interval_months := CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'QUARTERLY')))
                               WHEN 'QUARTERLY' THEN 3
                               WHEN 'SEMI-ANNUALLY' THEN 6
                               WHEN 'SEMI-ANNUAL' THEN 6
                               WHEN 'ANNUALLY' THEN 12
                               WHEN 'ANNUAL' THEN 12
                               ELSE 3
            END;
        reference_date := (income_statement_report_date + (interval_months || ' months')::INTERVAL)::DATE;
    ELSE
        RETURN NULL;
    END IF;

    -- Calculate fiscal year boundaries
    -- Determine which fiscal year the reference_date falls into
    next_fy_end_date := fy_end_date;
    WHILE next_fy_end_date < reference_date
        LOOP
            next_fy_end_date := (next_fy_end_date + INTERVAL '1 year')::DATE;
        END LOOP;

    -- The current FY end for this period is one year before next_fy_end_date
    -- unless reference_date is exactly on or before the original fy_end_date
    IF next_fy_end_date = fy_end_date THEN
        -- Reference date is before/on the first FY end, use it directly
        NULL; -- next_fy_end_date is already correct
    END IF;

    -- Fiscal year range is always 12 months
    fy_range_months := 12;

    -- Calculate months from the START of the fiscal year to the reference date
    -- FY starts the day after the previous FY end
    months_into_fy := (
        (EXTRACT(YEAR FROM reference_date) - EXTRACT(YEAR FROM (next_fy_end_date - INTERVAL '1 year'))) * 12
            + (EXTRACT(MONTH FROM reference_date) - EXTRACT(MONTH FROM (next_fy_end_date - INTERVAL '1 year')))
        )::INTEGER;

    -- Normalize to 1-12 range (months within the fiscal year)
    months_into_fy := ((months_into_fy - 1) % 12) + 1;
    IF months_into_fy <= 0 THEN
        months_into_fy := months_into_fy + 12;
    END IF;

    -- Derive fiscal quarter from the fiscal month
    -- Q1: months 1-3, Q2: months 4-6, Q3: months 7-9, Q4: months 10-12
    fiscal_quarter := CEIL(months_into_fy / 3.0)::INTEGER;

    -- Ensure quarter is within valid range
    IF fiscal_quarter < 1 THEN
        fiscal_quarter := 1;
    ELSIF fiscal_quarter > 4 THEN
        fiscal_quarter := 4;
    END IF;

    RETURN fiscal_quarter;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Reporting Lag
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_reporting_lag(
    next_earnings                DATE,
    income_statement_report_date DATE,
    earnings_report_frequency    TEXT DEFAULT 'Quarterly'
)
    RETURNS INTEGER AS
$$
BEGIN
    IF next_earnings IS NULL OR income_statement_report_date IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN next_earnings - income_statement_report_date;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Expected Report Date
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_expected_report_date(
    period_end_date           DATE,
    earnings_report_frequency TEXT
)
    RETURNS DATE AS
$$
BEGIN
    IF period_end_date IS NULL THEN
        RETURN NULL;
    END IF;
    RETURN period_end_date + (get_expected_reporting_lag_days(earnings_report_frequency) || ' days')::INTERVAL;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Validate Fiscal Dates
-- ===================================================================
CREATE OR REPLACE FUNCTION validate_fiscal_dates(
    fy_end_date    DATE,
    report_date    DATE,
    reference_date DATE DEFAULT CURRENT_DATE
)
    RETURNS TABLE
            (
                issue    TEXT,
                severity TEXT
            )
AS
$$
BEGIN
    IF fy_end_date > reference_date THEN
        RETURN QUERY SELECT 'FY End Date is in the future'::TEXT as fy_end_future, 'WARNING'::TEXT as fy_end_warning;
    END IF;

    IF report_date IS NOT NULL AND report_date < fy_end_date - INTERVAL '1 year' THEN
        RETURN QUERY SELECT 'Report date predates fiscal year'::TEXT as report_date_predates,
                            'ERROR'::TEXT                            as report_date_error;
    END IF;

    IF report_date > reference_date + INTERVAL '1 day' THEN
        RETURN QUERY SELECT 'Report date is in the future'::TEXT as report_date_future,
                            'WARNING'::TEXT                      as report_date_warning;
    END IF;

    IF fy_end_date != (DATE_TRUNC('month', fy_end_date) + INTERVAL '1 month - 1 day')::DATE THEN
        RETURN QUERY SELECT 'FY End is not last day of month'::TEXT as fy_end_ldm, 'INFO'::TEXT as fy_end_info;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- Importing US Region Data...
-- ===================================================================
\echo 'Importing regional data (US, EU, APAC, ROTW)...'

-- ===================================================================
-- STAGING TABLE CREATION
-- Columns organized by logical category for maintainability
-- ===================================================================
DROP TABLE IF EXISTS screening_staging;
CREATE TEMP TABLE screening_staging
(
    "Ticker"                                           TEXT, -- alias: ticker
    "ISIN"                                             TEXT, -- alias: isin
    "Name"                                             TEXT, -- alias: name
    "Description"                                      TEXT, -- alias: description
    "Region"                                           TEXT, -- alias: region
    "Country"                                          TEXT, -- alias: country
    "Trading Country"                                  TEXT, -- alias: trading_country
    "Exchange"                                         TEXT, -- alias: exchange
    "Unit"                                             TEXT, -- alias: unit
    "Sector"                                           TEXT, -- alias: sector
    "Industry"                                         TEXT, -- alias: industry
    "Style Class"                                      TEXT, -- alias: style_class
    "Size Class"                                       TEXT, -- alias: size_class
    "Last Updated"                                     TEXT, -- alias: last_updated
    "Income Statement Report Date"                     TEXT, -- alias: income_statement_report_date
    "FY End"                                           TEXT, -- alias: fy_end
    "Next Earnings"                                    TEXT, -- alias: next_earnings
    "Next Earnings (When)"                             TEXT, -- alias: next_earnings_when
    "Next Earnings (Status)"                           TEXT, -- alias: next_earnings_status
    "Dividend Record (Currency)"                       TEXT, -- alias: dividend_record_currency
    "Dividend Record (Amount)"                         TEXT, -- alias: dividend_record_amount
    "Dividend Record (Frequency)"                      TEXT, -- alias: dividend_record_frequency
    "Dividend Streak"                                  TEXT, -- alias: dividend_streak
    "Dividend Record (Announce Date)"                  TEXT, -- alias: dividend_record_announce_date
    "Dividend Record (Payable Date)"                   TEXT, -- alias: dividend_record_payable_date
    "Dividend Record (Record Date)"                    TEXT, -- alias: dividend_record_record_date
    "Dividend Record (Ex Date)"                        TEXT, -- alias: dividend_record_ex_date
    "Market Cap"                                       TEXT, -- alias: market_cap
    "Enterprise Value"                                 TEXT, -- alias: enterprise_value
    "Last Price"                                       TEXT, -- alias: last_price
    "Price Target (YTD Ago)"                           TEXT, -- alias: price_target_ytd_ago
    "Total Return (YTD)"                               TEXT, -- alias: total_return_ytd
    "Price Target"                                     TEXT, -- alias: price_target
    "Price Target - Low"                               TEXT, -- alias: price_target_low
    "Price Target - Median"                            TEXT, -- alias: price_target_median
    "Price Target - High"                              TEXT, -- alias: price_target_high
    "Price Target - #"                                 TEXT, -- alias: price_target_count
    "P/E (NTM)"                                        TEXT, -- alias: p_e_ntm
    "P/E (LTM)"                                        TEXT, -- alias: p_e_ltm
    "Altman Z-Score (FY)"                              TEXT, -- alias: altman_z_score_fy
    "Altman Z-Score (FQ)"                              TEXT, -- alias: altman_z_score_fq
    "Altman Z-Score (LTM)"                             TEXT, -- alias: altman_z_score_ltm
    "Beta (1Y)"                                        TEXT, -- alias: beta_1y
    "Beta (2Y)"                                        TEXT, -- alias: beta_2y
    "Beta (5Y)"                                        TEXT, -- alias: beta_5y
    "Analyst Rating"                                   TEXT, -- alias: analyst_rating
    "# Strong Sell Ratings"                            TEXT, -- alias: num_strong_sell_ratings
    "# Strong Buys Ratings"                            TEXT, -- alias: num_strong_buys_ratings
    "# Hold Ratings"                                   TEXT, -- alias: num_hold_ratings
    "# Buys Ratings"                                   TEXT, -- alias: num_buys_ratings
    "# Sell Ratings"                                   TEXT, -- alias: num_sell_ratings
    "# No Opinion Ratings"                             TEXT, -- alias: num_no_opinion_ratings
    "Total Revenues/CAGR (5Y FY)"                      TEXT, -- alias: total_revenues_cagr_5y_fy
    "Total Revenues (FQ)"                              TEXT, -- alias: total_revenues_fq
    "Total Revenues (-1FY)"                            TEXT, -- alias: total_revenues_1fy
    "Total Revenues (FY)"                              TEXT, -- alias: total_revenues_fy
    "Total Revenues (LTM)"                             TEXT, -- alias: total_revenues_ltm
    "Total Operating Expenses (LTM)"                   TEXT, -- alias: total_operating_expenses_ltm
    "P/TBV (LTM)"                                      TEXT, -- alias: p_tbv_ltm
    "TBV (FY)"                                         TEXT, -- alias: tbv_fy
    "TBV (LTM)"                                        TEXT, -- alias: tbv_ltm
    "Market Cap (Country R)"                           TEXT, -- alias: market_cap_country_r
    "Tot. Return %/CAGR (3Y)"                          TEXT, -- alias: tot_return_pct_cagr_3y
    "Tot. Return %/CAGR (10Y)"                         TEXT, -- alias: tot_return_pct_cagr_10y
    "Total Return (5Y)"                                TEXT, -- alias: total_return_5y
    "Total Return (10Y)"                               TEXT, -- alias: total_return_10y
    "Net Income/Adj. (-1FY)"                           TEXT, -- alias: net_income_adj_1fy
    "CFF (LTM)"                                        TEXT, -- alias: cff_ltm
    "CFI (LTM)"                                        TEXT, -- alias: cfi_ltm
    "FCF (LTM)"                                        TEXT, -- alias: fcf_ltm
    "CFO (LTM)"                                        TEXT, -- alias: cfo_ltm
    "EBITDA (FQ)"                                      TEXT, -- alias: ebitda_fq
    "EBITDA (LTM)"                                     TEXT, -- alias: ebitda_ltm
    "EBITDA (FY)"                                      TEXT, -- alias: ebitda_fy
    "EBITDA (-1FY)"                                    TEXT, -- alias: ebitda_1fy
    "EBITDA/Adj. (LTM)"                                TEXT, -- alias: ebitda_adj_ltm
    "EBITDA/Adj. (FY)"                                 TEXT, -- alias: ebitda_adj_fy
    "EBITDA/Adj. (-1FY)"                               TEXT, -- alias: ebitda_adj_1fy
    "EBIT (FQ)"                                        TEXT, -- alias: ebit_fq
    "EBIT (LTM)"                                       TEXT, -- alias: ebit_ltm
    "EBIT (FY)"                                        TEXT, -- alias: ebit_fy
    "EBIT (-1FY)"                                      TEXT, -- alias: ebit_1fy
    "EBIT/Adj. (-1FY)"                                 TEXT, -- alias: ebit_adj_1fy
    "EBIT/Adj. (FY)"                                   TEXT, -- alias: ebit_adj_fy
    "EBIT/Adj. (LTM)"                                  TEXT, -- alias: ebit_adj_ltm
    "EBIT - Est Med (FY1E)"                            TEXT, -- alias: ebit_est_med_fy1e
    "EBIT - Est Med (NTM)"                             TEXT, -- alias: ebit_est_med_ntm
    "Return On Equity % (LTM)"                         TEXT, -- alias: return_on_equity_pct_ltm
    "Return On Equity % (FY)"                          TEXT, -- alias: return_on_equity_pct_fy
    "Net Income - (IS) (FY)"                           TEXT, -- alias: net_income_is_fy
    "Net Income - (IS) (LTM)"                          TEXT, -- alias: net_income_is_ltm
    "Normalized Net Income (FY)"                       TEXT, -- alias: normalized_net_income_fy
    "Normalized Net Income (LTM)"                      TEXT, -- alias: normalized_net_income_ltm
    "Net Income/Adj. (FY)"                             TEXT, -- alias: net_income_adj_fy
    "Net Income/Adj. (LTM)"                            TEXT, -- alias: net_income_adj_ltm
    "Net Income Margin % (FY)"                         TEXT, -- alias: net_income_margin_pct_fy
    "Net Income Margin % (LTM)"                        TEXT, -- alias: net_income_margin_pct_ltm
    "Volatility (1M)"                                  TEXT, -- alias: volatility_1m
    "Volatility (3M)"                                  TEXT, -- alias: volatility_3m
    "Volatility (6M)"                                  TEXT, -- alias: volatility_6m
    "Volatility (1Y)"                                  TEXT, -- alias: volatility_1y
    "Volume (Shrs)"                                    TEXT, -- alias: volume_shrs
    "Dividend Per Share (LTM)"                         TEXT, -- alias: dividend_per_share_ltm
    "Div Yield (Ind)"                                  TEXT, -- alias: div_yield_ind
    "Div Yield (LTM)"                                  TEXT, -- alias: div_yield_ltm
    "Total Debt (FY)"                                  TEXT, -- alias: total_debt_fy
    "Total Equity (FY)"                                TEXT, -- alias: total_equity_fy
    "Total Equity (LTM)"                               TEXT, -- alias: total_equity_ltm
    "Total Debt (LTM)"                                 TEXT, -- alias: total_debt_ltm
    "Total Assets (LTM)"                               TEXT, -- alias: total_assets_ltm
    "Total Assets (FY)"                                TEXT, -- alias: total_assets_fy
    "Current Ratio (FY)"                               TEXT, -- alias: current_ratio_fy
    "Current Ratio (LTM)"                              TEXT, -- alias: current_ratio_ltm
    "Gross Profit Margin % (FY)"                       TEXT, -- alias: gross_profit_margin_pct_fy
    "Gross Profit Margin % (LTM)"                      TEXT, -- alias: gross_profit_margin_pct_ltm
    "Asset Turnover (FY)"                              TEXT, -- alias: asset_turnover_fy
    "Asset Turnover (LTM)"                             TEXT, -- alias: asset_turnover_ltm
    "Gross Profit (LTM)"                               TEXT, -- alias: gross_profit_ltm
    "Gross Profit (FY)"                                TEXT, -- alias: gross_profit_fy
    "EPS Norm - Est Avg (NTM)"                         TEXT, -- alias: eps_norm_est_avg_ntm
    "EPS/Adj. (-1FY)"                                  TEXT, -- alias: eps_adj_1fy
    "EPS/Adj. (FY)"                                    TEXT, -- alias: eps_adj_fy
    "EPS/Adj. (LTM)"                                   TEXT, -- alias: eps_adj_ltm
    "EPS Norm - Est Avg (FY1E)"                        TEXT, -- alias: eps_norm_est_avg_fy1e
    "Gain (Loss) On Sale Of Assets (LTM)"              TEXT, -- alias: gain_loss_on_sale_of_assets_ltm
    "Cost Of Revenues (LTM)"                           TEXT, -- alias: cost_of_revenues_ltm
    "Cash Acquisitions (LTM)"                          TEXT, -- alias: cash_acquisitions_ltm
    "Cash Acquisitions (FY)"                           TEXT, -- alias: cash_acquisitions_fy
    "Cash Acquisitions (-1FY)"                         TEXT, -- alias: cash_acquisitions_1fy
    "Inventory (LTM)"                                  TEXT, -- alias: inventory_ltm
    "Goodwill (FQ)"                                    TEXT, -- alias: goodwill_fq
    "Goodwill (LTM)"                                   TEXT, -- alias: goodwill_ltm
    "Goodwill (FY)"                                    TEXT, -- alias: goodwill_fy
    "Goodwill (-1FY)"                                  TEXT, -- alias: goodwill_1fy
    "Impairment of Goodwill (FQ)"                      TEXT, -- alias: impairment_of_goodwill_fq
    "Impairment of Goodwill (LTM)"                     TEXT, -- alias: impairment_of_goodwill_ltm
    "Impairment of Goodwill (-1FY)"                    TEXT, -- alias: impairment_of_goodwill_1fy
    "Impairment of Goodwill (FY)"                      TEXT, -- alias: impairment_of_goodwill_fy
    "Operating Income (LTM)"                           TEXT, -- alias: operating_income_ltm
    "Asset Writedown (LTM)"                            TEXT, -- alias: asset_writedown_ltm
    "Asset Writedown (FY)"                             TEXT, -- alias: asset_writedown_fy
    "Asset Writedown (-1FY)"                           TEXT, -- alias: asset_writedown_1fy
    "Operating Income (FY)"                            TEXT, -- alias: operating_income_fy
    "Capital Expenditure (LTM)"                        TEXT, -- alias: capital_expenditure_ltm
    "Capital Expenditure (-1FY)"                       TEXT, -- alias: capital_expenditure_1fy
    "Capital Expenditure (FY)"                         TEXT, -- alias: capital_expenditure_fy
    "Retained Earnings (LTM)"                          TEXT, -- alias: retained_earnings_ltm
    "Total Current Assets (LTM)"                       TEXT, -- alias: total_current_assets_ltm
    "Total Current Liabilities (LTM)"                  TEXT, -- alias: total_current_liabilities_ltm
    "R&D Expenses (LTM)"                               TEXT, -- alias: randd_expenses_ltm
    "Restructuring Charges (LTM)"                      TEXT, -- alias: restructuring_charges_ltm
    "Restructuring Charges (FQ)"                       TEXT, -- alias: restructuring_charges_fq
    "Restructuring Charges (-1FY)"                     TEXT, -- alias: restructuring_charges_1fy
    "Restructuring Charges (FY)"                       TEXT, -- alias: restructuring_charges_fy
    "Interest Expense/Total (LTM)"                     TEXT, -- alias: interest_expense_total_ltm
    "Merger & Restructuring Charges (LTM)"             TEXT, -- alias: merger_and_restructuring_charges_ltm
    "Working Capital (LTM)"                            TEXT, -- alias: working_capital_ltm
    "Other Unusual Items/Total (LTM)"                  TEXT, -- alias: other_unusual_items_total_ltm
    "Interest Income On Investments (LTM)"             TEXT, -- alias: interest_income_on_investments_ltm
    "Buyback Yield (LTM)"                              TEXT, -- alias: buyback_yield_ltm
    "Return on Assets (ROA) % (LTM)"                   TEXT, -- alias: return_on_assets_roa_pct_ltm
    "Return on Assets (ROA) % (FY)"                    TEXT, -- alias: return_on_assets_roa_pct_fy
    "Net Income - (IS) (-1FY)"                         TEXT, -- alias: net_income_is_1fy
    "Normalized Net Income (-1FY)"                     TEXT, -- alias: normalized_net_income_1fy
    "CFF (FY)"                                         TEXT, -- alias: cff_fy
    "CFF (-1FY)"                                       TEXT, -- alias: cff_1fy
    "CFI (FY)"                                         TEXT, -- alias: cfi_fy
    "CFI (-1FY)"                                       TEXT, -- alias: cfi_1fy
    "CFO (FY)"                                         TEXT, -- alias: cfo_fy
    "CFO (-1FY)"                                       TEXT, -- alias: cfo_1fy
    "Div Yield (-1FYInd)"                              TEXT, -- alias: div_yield_1fyind
    "FCF (FY)"                                         TEXT, -- alias: fcf_fy
    "FCF (-1FY)"                                       TEXT, -- alias: fcf_1fy
    "Capital Expenditure (FQ)"                         TEXT, -- alias: capital_expenditure_fq
    "Capital Expenditure (5YAVGFQ)"                    TEXT, -- alias: capital_expenditure_5yavgfq
    "CFF (FQ)"                                         TEXT, -- alias: cff_fq
    "CFI (FQ)"                                         TEXT, -- alias: cfi_fq
    "CFO (FQ)"                                         TEXT, -- alias: cfo_fq
    "FCF (FQ)"                                         TEXT, -- alias: fcf_fq
    "Total Revenues (5YAVGFQ)"                         TEXT, -- alias: total_revenues_5yavgfq
    "EBITDA (5YAVGFQ)"                                 TEXT, -- alias: ebitda_5yavgfq
    "EBIT (5YAVGFQ)"                                   TEXT, -- alias: ebit_5yavgfq
    "FCF (5YAVGFQ)"                                    TEXT, -- alias: fcf_5yavgfq
    "Cash Acquisitions (FQ)"                           TEXT, -- alias: cash_acquisitions_fq
    "Cash Acquisitions (5YAVGFQ)"                      TEXT, -- alias: cash_acquisitions_5yavgfq
    "Asset Writedown (FQ)"                             TEXT, -- alias: asset_writedown_fq
    "Asset Writedown (5YAVGFQ)"                        TEXT, -- alias: asset_writedown_5yavgfq
    "Impairment of Goodwill (5YAVGFQ)"                 TEXT, -- alias: impairment_of_goodwill_5yavgfq
    "Operating Income (FQ)"                            TEXT, -- alias: operating_income_fq
    "Operating Income (5YAVGFQ)"                       TEXT, -- alias: operating_income_5yavgfq
    "P/B (LTM)"                                        TEXT, -- alias: p_b_ltm
    "P/B (-1FY)"                                       TEXT, -- alias: p_b_1fy
    "P/B (5YAVG)"                                      TEXT, -- alias: p_b_5yavg
    "Cash And Equivalents (LTM)"                       TEXT, -- alias: cash_and_equivalents_ltm
    "Cash And Equivalents (FQ)"                        TEXT, -- alias: cash_and_equivalents_fq
    "Cash And Equivalents (FY)"                        TEXT, -- alias: cash_and_equivalents_fy
    "Cash And Equivalents (5YAVGFQ)"                   TEXT, -- alias: cash_and_equivalents_5yavgfq
    "Inventory (FQ)"                                   TEXT, -- alias: inventory_fq
    "Inventory (FY)"                                   TEXT, -- alias: inventory_fy
    "Goodwill (5YAVGFQ)"                               TEXT, -- alias: goodwill_5yavgfq
    "Inventory (5YAVGFQ)"                              TEXT, -- alias: inventory_5yavgfq
    "Retained Earnings (FQ)"                           TEXT, -- alias: retained_earnings_fq
    "Retained Earnings (FY)"                           TEXT, -- alias: retained_earnings_fy
    "Retained Earnings (5YAVGFQ)"                      TEXT, -- alias: retained_earnings_5yavgfq
    "Working Capital (FQ)"                             TEXT, -- alias: working_capital_fq
    "Working Capital (FY)"                             TEXT, -- alias: working_capital_fy
    "Working Capital (5YAVGFY)"                        TEXT, -- alias: working_capital_5yavgfy
    "Div Yield (TTM)"                                  TEXT, -- alias: div_yield_ttm
    "Div Yield (NTM)"                                  TEXT, -- alias: div_yield_ntm
    "Div Yield (5YAVGLTM)"                             TEXT, -- alias: div_yield_5yavgltm
    "Gross Intangible Assets (LTM)"                    TEXT, -- alias: gross_intangible_assets_ltm
    "Gross Intangible Assets (FY)"                     TEXT, -- alias: gross_intangible_assets_fy
    "Gross Intangible Assets (5YAVGFQ)"                TEXT, -- alias: gross_intangible_assets_5yavgfq
    "Restructuring Charges (5YAVGFQ)"                  TEXT, -- alias: restructuring_charges_5yavgfq
    "Merger & Restructuring Charges (FQ)"              TEXT, -- alias: merger_and_restructuring_charges_fq
    "Merger & Restructuring Charges (FY)"              TEXT, -- alias: merger_and_restructuring_charges_fy
    "Merger & Restructuring Charges (5YAVGFQ)"         TEXT, -- alias: merger_and_restructuring_charges_5yavgfq
    "Normalized Net Income (FQ)"                       TEXT, -- alias: normalized_net_income_fq
    "Normalized Net Income (5YAVGFQ)"                  TEXT, -- alias: normalized_net_income_5yavgfq
    "Net Income/Adj. (FQ)"                             TEXT, -- alias: net_income_adj_fq
    "Net Income/Adj. (5YAVGFQ)"                        TEXT, -- alias: net_income_adj_5yavgfq
    "Net Income - (IS) (FQ)"                           TEXT, -- alias: net_income_is_fq
    "Net Income - (IS) (5YAVGFQ)"                      TEXT, -- alias: net_income_is_5yavgfq
    "Net Income - (IS) (5YAVGLTM)"                     TEXT, -- alias: net_income_is_5yavgltm
    "Normalized Net Income (5YAVGLTM)"                 TEXT, -- alias: normalized_net_income_5yavgltm
    "EBITDA (5YAVGLTM)"                                TEXT, -- alias: ebitda_5yavgltm
    "EBIT (5YAVGLTM)"                                  TEXT, -- alias: ebit_5yavgltm
    "Total Revenues (5YAVGLTM)"                        TEXT, -- alias: total_revenues_5yavgltm
    "Revenues - Est YoY % (FY1E)"                      TEXT, -- alias: revenues_est_yoy_pct_fy1e
    "Price Chg. % (1M)"                                TEXT, -- alias: price_chg_pct_1m
    "Price Chg. % (3M)"                                TEXT, -- alias: price_chg_pct_3m
    "1-Day %"                                          TEXT, -- alias: one_day_pct
    "Price (5D Ago)"                                   TEXT, -- alias: price_5d_ago
    "Price (1W Ago)"                                   TEXT, -- alias: price_1w_ago
    "Price (1M Ago)"                                   TEXT, -- alias: price_1m_ago
    "Price (3M Ago)"                                   TEXT, -- alias: price_3m_ago
    "Price (6M Ago)"                                   TEXT, -- alias: price_6m_ago
    "Price (1Y Ago)"                                   TEXT, -- alias: price_1y_ago
    "Price (3Y Ago)"                                   TEXT, -- alias: price_3y_ago
    "Price (5Y Ago)"                                   TEXT, -- alias: price_5y_ago
    "Price (QTD Ago)"                                  TEXT, -- alias: price_qtd_ago
    "Rel. Volume"                                      TEXT, -- alias: rel_volume
    "Shrs Out"                                         TEXT, -- alias: shares_outstanding
    "Shrs Out (-1FY)"                                  TEXT, -- alias: shrs_out_1fy
    "Common Dividends Paid (LTM)"                      TEXT, -- alias: common_dividends_paid_ltm
    "Common Dividends Paid (FY)"                       TEXT, -- alias: common_dividends_paid_fy
    "Selling General & Admin Expenses/Total (FQ)"      TEXT, -- alias: selling_general_and_admin_expenses_total_fq
    "Selling General & Admin Expenses/Total (FY)"      TEXT, -- alias: selling_general_and_admin_expenses_total_fy
    "Selling General & Admin Expenses/Total (-1FY)"    TEXT, -- alias: selling_general_and_admin_expenses_total_1fy
    "Selling General & Admin Expenses/Total (5YAVGFQ)" TEXT, -- alias: selling_general_and_admin_expenses_total_5yavgfq
    "Accounts Receivable/Total (FY)"                   TEXT, -- alias: accounts_receivable_total_fy
    "Accounts Receivable/Total (-1FY)"                 TEXT, -- alias: accounts_receivable_total_1fy
    "Accounts Receivable/Total (5YAVGFQ)"              TEXT, -- alias: accounts_receivable_total_5yavgfq
    "Marketing Expenses (FQ)"                          TEXT, -- alias: marketing_expenses_fq
    "Marketing Expenses (FY)"                          TEXT, -- alias: marketing_expenses_fy
    "Marketing Expenses (-1FY)"                        TEXT, -- alias: marketing_expenses_1fy
    "Marketing Expenses (5YAVGLTM)"                    TEXT, -- alias: marketing_expenses_5yavgltm
    "Revenues - Est Avg (NTM)"                         TEXT, -- alias: revenues_est_avg_ntm
    "Revenues - Est Avg (FY1E)"                        TEXT, -- alias: revenues_est_avg_fy1e
    "Revenues - Est Med (NTM)"                         TEXT, -- alias: revenues_est_med_ntm
    "Revenues - Est Med (FY1E)"                        TEXT, -- alias: revenues_est_med_fy1e
    "EV/Sales (EST FY1)"                               TEXT, -- alias: ev_sales_est_fy1
    "EV/Sales (LTM)"                                   TEXT, -- alias: ev_sales_ltm
    "EV/Sales (NTM)"                                   TEXT, -- alias: ev_sales_ntm
    "EV/Sales (-1FYLTM)"                               TEXT, -- alias: ev_sales_1fyltm
    "EV/Sales (-2FYLTM)"                               TEXT, -- alias: ev_sales_2fyltm
    "EV/Sales (-3FYLTM)"                               TEXT, -- alias: ev_sales_3fyltm
    "EV/Sales (3YAVGLTM)"                              TEXT, -- alias: ev_sales_3yavgltm
    "EV/Sales (-1FQLTM)"                               TEXT, -- alias: ev_sales_1fqltm
    "EV/Sales (-2FQLTM)"                               TEXT, -- alias: ev_sales_2fqltm
    "EV/Sales (-3FQLTM)"                               TEXT, -- alias: ev_sales_3fqltm
    "EV/Sales (-4FQLTM)"                               TEXT, -- alias: ev_sales_4fqltm
    "52W High/Adj"                                     TEXT, -- alias: 52w_high_adj
    "52W Low/Adj"                                      TEXT, -- alias: 52w_low_adj
    "EMA (20D)"                                        TEXT, -- alias: ema_20d
    "EMA (50D)"                                        TEXT, -- alias: ema_50d
    "EMA (100D)"                                       TEXT, -- alias: ema_100d
    "EMA (250D)"                                       TEXT, -- alias: ema_250d
    "EV/EBITDA (LTM)"                                  TEXT, -- alias: ev_ebitda_ltm
    "EV/EBITDA (NTM)"                                  TEXT, -- alias: ev_ebitda_ntm
    "EV/EBITDA (-1FYLTM)"                              TEXT, -- alias: ev_ebitda_1fyltm
    "EV/EBITDA (-1FQLTM)"                              TEXT, -- alias: ev_ebitda_1fqltm
    "EV/EBITDA (3YAVGLTM)"                             TEXT, -- alias: ev_ebitda_3yavgltm
    "EV/EBITDA (EST FY1)"                              TEXT, -- alias: ev_ebitda_est_fy1
    "P/E (EST FY1)"                                    TEXT, -- alias: p_e_est_fy1
    "P/E (-1FYLTM)"                                    TEXT, -- alias: p_e_1fyltm
    "P/E (-2FYLTM)"                                    TEXT, -- alias: p_e_2fyltm
    "P/E (-3FYLTM)"                                    TEXT, -- alias: p_e_3fyltm
    "P/E (3YAVGLTM)"                                   TEXT, -- alias: p_e_3yavgltm
    "P/E (-1FQLTM)"                                    TEXT, -- alias: p_e_1fqltm
    "P/E (-2FQLTM)"                                    TEXT, -- alias: p_e_2fqltm
    "P/E (-3FQLTM)"                                    TEXT, -- alias: p_e_3fqltm
    "P/E (5YAVGLTM)"                                   TEXT, -- alias: p_e_5yavgltm
    "P/E (-0FQQoQLTM)"                                 TEXT, -- alias: p_e_0fqqoqltm
    "P/E (-0FYYoYLTM)"                                 TEXT, -- alias: p_e_0fyyoyltm
    "P/E (-1FYYoYLTM)"                                 TEXT, -- alias: p_e_1fyyoyltm
    "P/E (-0FQYoYLTM)"                                 TEXT, -- alias: p_e_0fqyoyltm
    "Full Time Employees (FQ)"                         TEXT, -- alias: full_time_employees_fq
    "Full Time Employees (FY)"                         TEXT, -- alias: full_time_employees_fy
    "Full Time Employees (-1FY)"                       TEXT, -- alias: full_time_employees_1fy
    "Full Time Employees (-2FY)"                       TEXT, -- alias: full_time_employees_2fy
    "Full Time Employees (-3FY)"                       TEXT, -- alias: full_time_employees_3fy
    "Avg Employees (5YAVGFY)"                          TEXT, -- alias: avg_employees_5yavgfy
    "Net EPS - Basic (LTM)"                            TEXT, -- alias: net_eps_basic_ltm
    "Net EPS - Basic (FQ)"                             TEXT, -- alias: net_eps_basic_fq
    "Net EPS - Basic (FY)"                             TEXT, -- alias: net_eps_basic_fy
    "Net EPS - Basic (-1FQFQ)"                         TEXT, -- alias: net_eps_basic_1fqfq
    "Net EPS - Basic (-2FQFQ)"                         TEXT, -- alias: net_eps_basic_2fqfq
    "Net EPS - Basic (-3FQFQ)"                         TEXT, -- alias: net_eps_basic_3fqfq
    "Net EPS - Basic (-4FQFQ)"                         TEXT, -- alias: net_eps_basic_4fqfq
    "Net EPS - Basic (-1FY)"                           TEXT, -- alias: net_eps_basic_1fy
    "Net EPS - Basic (-2FY)"                           TEXT, -- alias: net_eps_basic_2fy
    "Net EPS - Basic (-3FY)"                           TEXT, -- alias: net_eps_basic_3fy
    "Net EPS - Basic (-4FY)"                           TEXT, -- alias: net_eps_basic_4fy
    "Net EPS - Basic (-5FY)"                           TEXT, -- alias: net_eps_basic_5fy
    "EPS Est Avg Rev % (FY1E - 1W)"                    TEXT, -- alias: eps_est_avg_rev_pct_fy1e_1w
    "EPS Est Avg Rev % (FY1E - 1M)"                    TEXT, -- alias: eps_est_avg_rev_pct_fy1e_1m
    "EPS Est Avg Rev % (FY1E - 3M)"                    TEXT, -- alias: eps_est_avg_rev_pct_fy1e_3m
    "EPS Est Avg Rev % (FY1E - 6M)"                    TEXT, -- alias: eps_est_avg_rev_pct_fy1e_6m
    "EPS Est Avg Rev % (FY1E - 1Y)"                    TEXT, -- alias: eps_est_avg_rev_pct_fy1e_1y
    "Div Yield (-2FYInd)"                              TEXT, -- alias: div_yield_2fyind
    "Div Yield (-3FYInd)"                              TEXT, -- alias: div_yield_3fyind
    "Div Yield (-4FYInd)"                              TEXT, -- alias: div_yield_4fyind
    "Div Yield (-5FYInd)"                              TEXT, -- alias: div_yield_5fyind
    "EBITDA - Est Avg (NTM)"                           TEXT, -- alias: ebitda_est_avg_ntm
    "EBITDA - Est Avg (FY1E)"                          TEXT, -- alias: ebitda_est_avg_fy1e
    "EPS GAAP - Est Avg (NTM)"                         TEXT, -- alias: eps_gaap_est_avg_ntm
    "EPS GAAP - Est Avg (FY1E)"                        TEXT, -- alias: eps_gaap_est_avg_fy1e
    "EPS GAAP Est Avg Rev % (FY1E - 1M)"               TEXT, -- alias: eps_gaap_est_avg_rev_pct_fy1e_1m
    "EPS GAAP Est Avg Rev % (FY1E - 3M)"               TEXT, -- alias: eps_gaap_est_avg_rev_pct_fy1e_3m
    "EPS GAAP Est Avg Rev % (FY1E - 6M)"               TEXT, -- alias: eps_gaap_est_avg_rev_pct_fy1e_6m
    "EPS GAAP Est Avg Rev % (FY1E - 1Y)"               TEXT, -- alias: eps_gaap_est_avg_rev_pct_fy1e_1y
    "EPS Norm - Est # (FY1E)"                          TEXT, -- alias: eps_norm_est_num_fy1e
    "CFO (-1FQFQ)"                                     TEXT, -- alias: cfo_1fqfq
    "CFO (-2FQFQ)"                                     TEXT, -- alias: cfo_2fqfq
    "CFO (-3FQFQ)"                                     TEXT, -- alias: cfo_3fqfq
    "CFO (-4FQFQ)"                                     TEXT, -- alias: cfo_4fqfq
    "CFI (-1FQFQ)"                                     TEXT, -- alias: cfi_1fqfq
    "CFI (-2FQFQ)"                                     TEXT, -- alias: cfi_2fqfq
    "CFI (-3FQFQ)"                                     TEXT, -- alias: cfi_3fqfq
    "CFI (-4FQFQ)"                                     TEXT, -- alias: cfi_4fqfq
    "CFI (-2FY)"                                       TEXT, -- alias: cfi_2fy
    "CFI (-3FY)"                                       TEXT, -- alias: cfi_3fy
    "CFI (-4FY)"                                       TEXT, -- alias: cfi_4fy
    "FCF (-1FQFQ)"                                     TEXT, -- alias: fcf_1fqfq
    "FCF (-2FQFQ)"                                     TEXT, -- alias: fcf_2fqfq
    "FCF (-3FQFQ)"                                     TEXT, -- alias: fcf_3fqfq
    "FCF (-4FQFQ)"                                     TEXT, -- alias: fcf_4fqfq
    "CFF (-2FY)"                                       TEXT, -- alias: cff_2fy
    "CFF (-3FY)"                                       TEXT, -- alias: cff_3fy
    "CFF (-4FY)"                                       TEXT, -- alias: cff_4fy
    "CFF (-1FQFQ)"                                     TEXT, -- alias: cff_1fqfq
    "CFF (-2FQFQ)"                                     TEXT, -- alias: cff_2fqfq
    "CFF (-3FQFQ)"                                     TEXT, -- alias: cff_3fqfq
    "CFF (-4FQFQ)"                                     TEXT, -- alias: cff_4fqfq
    "CFO (-2FY)"                                       TEXT, -- alias: cfo_2fy
    "CFO (-3FY)"                                       TEXT, -- alias: cfo_3fy
    "CFO (-4FY)"                                       TEXT, -- alias: cfo_4fy
    "Cash Acquisitions (-1FQFQ)"                       TEXT, -- alias: cash_acquisitions_1fqfq
    "Cash Acquisitions (-2FQFQ)"                       TEXT, -- alias: cash_acquisitions_2fqfq
    "Cash Acquisitions (-3FQFQ)"                       TEXT, -- alias: cash_acquisitions_3fqfq
    "Cash Acquisitions (-4FQFQ)"                       TEXT, -- alias: cash_acquisitions_4fqfq
    "FCF (-2FY)"                                       TEXT, -- alias: fcf_2fy
    "FCF (-3FY)"                                       TEXT, -- alias: fcf_3fy
    "FCF (-4FY)"                                       TEXT, -- alias: fcf_4fy
    "Price Target (1W Ago)"                            TEXT, -- alias: price_target_1w_ago
    "Price Target (1M Ago)"                            TEXT, -- alias: price_target_1m_ago
    "Price Target (3M Ago)"                            TEXT, -- alias: price_target_3m_ago
    "Price Target (6M Ago)"                            TEXT, -- alias: price_target_6m_ago
    "Price Target (MTD Ago)"                           TEXT, -- alias: price_target_mtd_ago
    "Price Target (QTD Ago)"                           TEXT, -- alias: price_target_qtd_ago
    "Price Target (1Y Ago)"                            TEXT, -- alias: price_target_1y_ago
    "Price Target - # (3M Ago)"                        TEXT, -- alias: price_target_num_3m_ago
    "Price Target - # (6M Ago)"                        TEXT, -- alias: price_target_num_6m_ago
    "Price Target - # (YTD Ago)"                       TEXT, -- alias: price_target_num_ytd_ago
    "Price Target - # (1Y Ago)"                        TEXT, -- alias: price_target_num_1y_ago
    "Price Target - # (1W Ago)"                        TEXT, -- alias: price_target_num_1w_ago
    "Price Target - # (1M Ago)"                        TEXT, -- alias: price_target_num_1m_ago
    "Price Target - # (MTD Ago)"                       TEXT, -- alias: price_target_num_mtd_ago
    "Price Target - # (QTD Ago)"                       TEXT, -- alias: price_target_num_qtd_ago
    "Price Target - High (1W Ago)"                     TEXT, -- alias: price_target_high_1w_ago
    "Price Target - High (1M Ago)"                     TEXT, -- alias: price_target_high_1m_ago
    "Price Target - High (6M Ago)"                     TEXT, -- alias: price_target_high_6m_ago
    "Price Target - High (MTD Ago)"                    TEXT, -- alias: price_target_high_mtd_ago
    "Price Target - High (3M Ago)"                     TEXT, -- alias: price_target_high_3m_ago
    "Price Target - High (QTD Ago)"                    TEXT, -- alias: price_target_high_qtd_ago
    "Price Target - High (1Y Ago)"                     TEXT, -- alias: price_target_high_1y_ago
    "Price Target - High (YTD Ago)"                    TEXT, -- alias: price_target_high_ytd_ago
    "Price Target - Low (1W Ago)"                      TEXT, -- alias: price_target_low_1w_ago
    "Price Target - Low (1M Ago)"                      TEXT, -- alias: price_target_low_1m_ago
    "Price Target - Low (3M Ago)"                      TEXT, -- alias: price_target_low_3m_ago
    "Price Target - Low (6M Ago)"                      TEXT, -- alias: price_target_low_6m_ago
    "Price Target - Low (MTD Ago)"                     TEXT, -- alias: price_target_low_mtd_ago
    "Price Target - Low (QTD Ago)"                     TEXT, -- alias: price_target_low_qtd_ago
    "Price Target - Low (YTD Ago)"                     TEXT, -- alias: price_target_low_ytd_ago
    "Price Target - Low (1Y Ago)"                      TEXT, -- alias: price_target_low_1y_ago
    "Price Target - Median (1W Ago)"                   TEXT, -- alias: price_target_median_1w_ago
    "Price Target - Median (1M Ago)"                   TEXT, -- alias: price_target_median_1m_ago
    "Price Target - Median (3M Ago)"                   TEXT, -- alias: price_target_median_3m_ago
    "Price Target - Median (6M Ago)"                   TEXT, -- alias: price_target_median_6m_ago
    "Price Target - Median (MTD Ago)"                  TEXT, -- alias: price_target_median_mtd_ago
    "Price Target - Median (QTD Ago)"                  TEXT, -- alias: price_target_median_qtd_ago
    "Price Target - Median (YTD Ago)"                  TEXT, -- alias: price_target_median_ytd_ago
    "Price Target - Median (1Y Ago)"                   TEXT, -- alias: price_target_median_1y_ago
    "Impairment of Goodwill (-1FQFQ)"                  TEXT, -- alias: impairment_of_goodwill_1fqfq
    "Impairment of Goodwill (-2FQFQ)"                  TEXT, -- alias: impairment_of_goodwill_2fqfq
    "Impairment of Goodwill (-3FQFQ)"                  TEXT, -- alias: impairment_of_goodwill_3fqfq
    "Impairment of Goodwill (-4FQFQ)"                  TEXT, -- alias: impairment_of_goodwill_4fqfq
    "Impairment of Goodwill (-2FY)"                    TEXT, -- alias: impairment_of_goodwill_2fy
    "Impairment of Goodwill (-3FY)"                    TEXT, -- alias: impairment_of_goodwill_3fy
    "Impairment of Goodwill (-4FY)"                    TEXT, -- alias: impairment_of_goodwill_4fy
    "Asset Writedown (-1FQFQ)"                         TEXT, -- alias: asset_writedown_1fqfq
    "Asset Writedown (-2FQFQ)"                         TEXT, -- alias: asset_writedown_2fqfq
    "Asset Writedown (-3FQFQ)"                         TEXT, -- alias: asset_writedown_3fqfq
    "Asset Writedown (-4FQFQ)"                         TEXT, -- alias: asset_writedown_4fqfq
    "Asset Writedown (-2FY)"                           TEXT, -- alias: asset_writedown_2fy
    "Asset Writedown (-3FY)"                           TEXT, -- alias: asset_writedown_3fy
    "Asset Writedown (-4FY)"                           TEXT, -- alias: asset_writedown_4fy
    "Asset Writedown (-5FY)"                           TEXT, -- alias: asset_writedown_5fy
    "Gain (Loss) On Sale Of Assets (FQ)"               TEXT, -- alias: gain_loss_on_sale_of_assets_fq
    "Gain (Loss) On Sale Of Assets (FY)"               TEXT, -- alias: gain_loss_on_sale_of_assets_fy
    "Gain (Loss) On Sale Of Assets (-1FQFQ)"           TEXT, -- alias: gain_loss_on_sale_of_assets_1fqfq
    "Gain (Loss) On Sale Of Assets (-2FQFQ)"           TEXT, -- alias: gain_loss_on_sale_of_assets_2fqfq
    "Gain (Loss) On Sale Of Assets (-3FQFQ)"           TEXT, -- alias: gain_loss_on_sale_of_assets_3fqfq
    "Gain (Loss) On Sale Of Assets (-4FQFQ)"           TEXT, -- alias: gain_loss_on_sale_of_assets_4fqfq
    "Gain (Loss) On Sale Of Assets (-1FY)"             TEXT, -- alias: gain_loss_on_sale_of_assets_1fy
    "Gain (Loss) On Sale Of Assets (-2FY)"             TEXT, -- alias: gain_loss_on_sale_of_assets_2fy
    "Gain (Loss) On Sale Of Assets (-3FY)"             TEXT, -- alias: gain_loss_on_sale_of_assets_3fy
    "Gain (Loss) On Sale Of Assets (-4FY)"             TEXT, -- alias: gain_loss_on_sale_of_assets_4fy
    "Restructuring Charges (-1FQFQ)"                   TEXT, -- alias: restructuring_charges_1fqfq
    "Restructuring Charges (-2FQFQ)"                   TEXT, -- alias: restructuring_charges_2fqfq
    "Restructuring Charges (-3FQFQ)"                   TEXT, -- alias: restructuring_charges_3fqfq
    "Restructuring Charges (-4FQFQ)"                   TEXT, -- alias: restructuring_charges_4fqfq
    "Restructuring Charges (-2FY)"                     TEXT, -- alias: restructuring_charges_2fy
    "Restructuring Charges (-3FY)"                     TEXT, -- alias: restructuring_charges_3fy
    "Restructuring Charges (-4FY)"                     TEXT, -- alias: restructuring_charges_4fy
    "Net Income - (IS) (-1FQFQ)"                       TEXT, -- alias: net_income_is_1fqfq
    "Net Income - (IS) (-2FQFQ)"                       TEXT, -- alias: net_income_is_2fqfq
    "Net Income - (IS) (-3FQFQ)"                       TEXT, -- alias: net_income_is_3fqfq
    "Net Income - (IS) (-4FQFQ)"                       TEXT, -- alias: net_income_is_4fqfq
    "Net Income - (IS) (-2FY)"                         TEXT, -- alias: net_income_is_2fy
    "Net Income - (IS) (-3FY)"                         TEXT, -- alias: net_income_is_3fy
    "Net Income - (IS) (-4FY)"                         TEXT, -- alias: net_income_is_4fy
    "Normalized Net Income (-1FQFQ)"                   TEXT, -- alias: normalized_net_income_1fqfq
    "Normalized Net Income (-2FQFQ)"                   TEXT, -- alias: normalized_net_income_2fqfq
    "Normalized Net Income (-3FQFQ)"                   TEXT, -- alias: normalized_net_income_3fqfq
    "Normalized Net Income (-4FQFQ)"                   TEXT, -- alias: normalized_net_income_4fqfq
    "Normalized Net Income (-2FY)"                     TEXT, -- alias: normalized_net_income_2fy
    "Normalized Net Income (-3FY)"                     TEXT, -- alias: normalized_net_income_3fy
    "Normalized Net Income (-4FY)"                     TEXT, -- alias: normalized_net_income_4fy
    "Net Income/Adj. (-1FQFQ)"                         TEXT, -- alias: net_income_adj_1fqfq
    "Net Income/Adj. (-2FQFQ)"                         TEXT, -- alias: net_income_adj_2fqfq
    "Net Income/Adj. (-3FQFQ)"                         TEXT, -- alias: net_income_adj_3fqfq
    "Net Income/Adj. (-4FQFQ)"                         TEXT, -- alias: net_income_adj_4fqfq
    "Net Income/Adj. (-2FY)"                           TEXT, -- alias: net_income_adj_2fy
    "Net Income/Adj. (-3FY)"                           TEXT, -- alias: net_income_adj_3fy
    "Net Income/Adj. (-4FY)"                           TEXT, -- alias: net_income_adj_4fy
    "EBIT (-1FQFQ)"                                    TEXT, -- alias: ebit_1fqfq
    "EBIT (-2FQFQ)"                                    TEXT, -- alias: ebit_2fqfq
    "EBIT (-3FQFQ)"                                    TEXT, -- alias: ebit_3fqfq
    "EBIT (-4FQFQ)"                                    TEXT, -- alias: ebit_4fqfq
    "EBIT (-2FY)"                                      TEXT, -- alias: ebit_2fy
    "EBIT (-3FY)"                                      TEXT, -- alias: ebit_3fy
    "EBIT (-4FY)"                                      TEXT, -- alias: ebit_4fy
    "EBIT/Adj. (FQ)"                                   TEXT, -- alias: ebit_adj_fq
    "EBIT/Adj. (-1FQFQ)"                               TEXT, -- alias: ebit_adj_1fqfq
    "EBIT/Adj. (-2FQFQ)"                               TEXT, -- alias: ebit_adj_2fqfq
    "EBIT/Adj. (-3FQFQ)"                               TEXT, -- alias: ebit_adj_3fqfq
    "EBIT/Adj. (-4FQFQ)"                               TEXT, -- alias: ebit_adj_4fqfq
    "EBIT/Adj. (-2FY)"                                 TEXT, -- alias: ebit_adj_2fy
    "EBIT/Adj. (-3FY)"                                 TEXT, -- alias: ebit_adj_3fy
    "EBIT/Adj. (-4FY)"                                 TEXT, -- alias: ebit_adj_4fy
    "EBITDA (-1FQFQ)"                                  TEXT, -- alias: ebitda_1fqfq
    "EBITDA (-2FQFQ)"                                  TEXT, -- alias: ebitda_2fqfq
    "EBITDA (-3FQFQ)"                                  TEXT, -- alias: ebitda_3fqfq
    "EBITDA (-4FQFQ)"                                  TEXT, -- alias: ebitda_4fqfq
    "EBITDA (-2FY)"                                    TEXT, -- alias: ebitda_2fy
    "EBITDA (-4FY)"                                    TEXT, -- alias: ebitda_4fy
    "EBITDA (-3FY)"                                    TEXT, -- alias: ebitda_3fy
    "EBITDA/Adj. (FQ)"                                 TEXT, -- alias: ebitda_adj_fq
    "EBITDA/Adj. (-1FQFQ)"                             TEXT, -- alias: ebitda_adj_1fqfq
    "EBITDA/Adj. (-2FQFQ)"                             TEXT, -- alias: ebitda_adj_2fqfq
    "EBITDA/Adj. (-3FQFQ)"                             TEXT, -- alias: ebitda_adj_3fqfq
    "EBITDA/Adj. (-4FQFQ)"                             TEXT, -- alias: ebitda_adj_4fqfq
    "EBITDA/Adj. (-2FY)"                               TEXT, -- alias: ebitda_adj_2fy
    "EBITDA/Adj. (-3FY)"                               TEXT, -- alias: ebitda_adj_3fy
    "EBITDA/Adj. (-4FY)"                               TEXT, -- alias: ebitda_adj_4fy
    "Basic EPS - Cont (LTM)"                           TEXT, -- alias: basic_eps_cont_ltm
    "Basic EPS - Cont (FQ)"                            TEXT, -- alias: basic_eps_cont_fq
    "Basic EPS - Cont (FY)"                            TEXT, -- alias: basic_eps_cont_fy
    "Basic EPS - Cont (-1FQFQ)"                        TEXT, -- alias: basic_eps_cont_1fqfq
    "Basic EPS - Cont (-2FQFQ)"                        TEXT, -- alias: basic_eps_cont_2fqfq
    "Basic EPS - Cont (-4FQFQ)"                        TEXT, -- alias: basic_eps_cont_4fqfq
    "Basic EPS - Cont (-3FQFQ)"                        TEXT, -- alias: basic_eps_cont_3fqfq
    "Basic EPS - Cont (-1FY)"                          TEXT, -- alias: basic_eps_cont_1fy
    "Basic EPS - Cont (-2FY)"                          TEXT, -- alias: basic_eps_cont_2fy
    "Basic EPS - Cont (-3FY)"                          TEXT, -- alias: basic_eps_cont_3fy
    "Basic EPS - Cont (-4FY)"                          TEXT, -- alias: basic_eps_cont_4fy
    "EPS/Adj. (FQ)"                                    TEXT, -- alias: eps_adj_fq
    "EPS/Adj. (-1FQFQ)"                                TEXT, -- alias: eps_adj_1fqfq
    "EPS/Adj. (-3FQFQ)"                                TEXT, -- alias: eps_adj_3fqfq
    "EPS/Adj. (-4FQFQ)"                                TEXT, -- alias: eps_adj_4fqfq
    "EPS/Adj. (-2FQFQ)"                                TEXT, -- alias: eps_adj_2fqfq
    "EPS/Adj. (-2FY)"                                  TEXT, -- alias: eps_adj_2fy
    "EPS/Adj. (-3FY)"                                  TEXT, -- alias: eps_adj_3fy
    "EPS/Adj. (-4FY)"                                  TEXT, -- alias: eps_adj_4fy
    "Cash Acquisitions (-2FY)"                         TEXT, -- alias: cash_acquisitions_2fy
    "Cash Acquisitions (-3FY)"                         TEXT, -- alias: cash_acquisitions_3fy
    "Cash Acquisitions (-4FY)"                         TEXT, -- alias: cash_acquisitions_4fy
    "Capital Expenditure (-1FQFQ)"                     TEXT, -- alias: capital_expenditure_1fqfq
    "Capital Expenditure (-3FQFQ)"                     TEXT, -- alias: capital_expenditure_3fqfq
    "Capital Expenditure (-4FQFQ)"                     TEXT, -- alias: capital_expenditure_4fqfq
    "Capital Expenditure (-2FQFQ)"                     TEXT, -- alias: capital_expenditure_2fqfq
    "Capital Expenditure (-2FY)"                       TEXT, -- alias: capital_expenditure_2fy
    "Capital Expenditure (-3FY)"                       TEXT, -- alias: capital_expenditure_3fy
    "Capital Expenditure (-4FY)"                       TEXT, -- alias: capital_expenditure_4fy
    "Working Capital (-1FQ)"                           TEXT, -- alias: working_capital_1fq
    "Working Capital (-2FQ)"                           TEXT, -- alias: working_capital_2fq
    "Working Capital (-3FQ)"                           TEXT, -- alias: working_capital_3fq
    "Working Capital (-4FQ)"                           TEXT, -- alias: working_capital_4fq
    "Working Capital (-1FY)"                           TEXT, -- alias: working_capital_1fy
    "Working Capital (-2FY)"                           TEXT, -- alias: working_capital_2fy
    "Working Capital (-3FY)"                           TEXT, -- alias: working_capital_3fy
    "Working Capital (-4FY)"                           TEXT, -- alias: working_capital_4fy
    "Total Debt (FQ)"                                  TEXT, -- alias: total_debt_fq
    "Total Debt (-1FQ)"                                TEXT, -- alias: total_debt_1fq
    "Total Debt (-2FQ)"                                TEXT, -- alias: total_debt_2fq
    "Total Debt (-3FQ)"                                TEXT, -- alias: total_debt_3fq
    "Total Debt (-4FQ)"                                TEXT, -- alias: total_debt_4fq
    "Total Debt (-1FY)"                                TEXT, -- alias: total_debt_1fy
    "Total Debt (-2FY)"                                TEXT, -- alias: total_debt_2fy
    "Total Debt (-3FY)"                                TEXT, -- alias: total_debt_3fy
    "Total Debt (-4FY)"                                TEXT, -- alias: total_debt_4fy
    "Total Assets (FQ)"                                TEXT, -- alias: total_assets_fq
    "Total Assets (-1FQ)"                              TEXT, -- alias: total_assets_1fq
    "Total Assets (-2FQ)"                              TEXT, -- alias: total_assets_2fq
    "Total Assets (-3FQ)"                              TEXT, -- alias: total_assets_3fq
    "Total Assets (-4FQ)"                              TEXT, -- alias: total_assets_4fq
    "Total Assets (-1FY)"                              TEXT, -- alias: total_assets_1fy
    "Total Assets (-2FY)"                              TEXT, -- alias: total_assets_2fy
    "Total Assets (-3FY)"                              TEXT, -- alias: total_assets_3fy
    "Total Assets (-4FY)"                              TEXT, -- alias: total_assets_4fy
    "Gross Profit (FQ)"                                TEXT, -- alias: gross_profit_fq
    "Gross Profit (-1FQFQ)"                            TEXT, -- alias: gross_profit_1fqfq
    "Gross Profit (-3FQFQ)"                            TEXT, -- alias: gross_profit_3fqfq
    "Gross Profit (-4FQFQ)"                            TEXT, -- alias: gross_profit_4fqfq
    "Gross Profit (-2FQFQ)"                            TEXT, -- alias: gross_profit_2fqfq
    "Gross Profit (-1FY)"                              TEXT, -- alias: gross_profit_1fy
    "Gross Profit (-2FY)"                              TEXT, -- alias: gross_profit_2fy
    "Gross Profit (-3FY)"                              TEXT, -- alias: gross_profit_3fy
    "Gross Profit (-4FY)"                              TEXT, -- alias: gross_profit_4fy
    "Inventory (-1FQ)"                                 TEXT, -- alias: inventory_1fq
    "Inventory (-3FQ)"                                 TEXT, -- alias: inventory_3fq
    "Inventory (-4FQ)"                                 TEXT, -- alias: inventory_4fq
    "Inventory (-2FQ)"                                 TEXT, -- alias: inventory_2fq
    "Inventory (-1FY)"                                 TEXT, -- alias: inventory_1fy
    "Inventory (-2FY)"                                 TEXT, -- alias: inventory_2fy
    "Inventory (-4FY)"                                 TEXT, -- alias: inventory_4fy
    "Inventory (-3FY)"                                 TEXT, -- alias: inventory_3fy
    "Goodwill (-1FQ)"                                  TEXT, -- alias: goodwill_1fq
    "Goodwill (-4FQ)"                                  TEXT, -- alias: goodwill_4fq
    "Goodwill (-2FQ)"                                  TEXT, -- alias: goodwill_2fq
    "Goodwill (-3FQ)"                                  TEXT, -- alias: goodwill_3fq
    "Goodwill (-2FY)"                                  TEXT, -- alias: goodwill_2fy
    "Goodwill (-3FY)"                                  TEXT, -- alias: goodwill_3fy
    "Goodwill (-4FY)"                                  TEXT, -- alias: goodwill_4fy
    "Operating Income (-1FQFQ)"                        TEXT, -- alias: operating_income_1fqfq
    "Operating Income (-3FQFQ)"                        TEXT, -- alias: operating_income_3fqfq
    "Operating Income (-4FQFQ)"                        TEXT, -- alias: operating_income_4fqfq
    "Operating Income (-2FQFQ)"                        TEXT, -- alias: operating_income_2fqfq
    "Operating Income (-1FY)"                          TEXT, -- alias: operating_income_1fy
    "Operating Income (-2FY)"                          TEXT, -- alias: operating_income_2fy
    "Operating Income (-4FY)"                          TEXT, -- alias: operating_income_4fy
    "Operating Income (-3FY)"                          TEXT, -- alias: operating_income_3fy
    "Retained Earnings (-1FQ)"                         TEXT, -- alias: retained_earnings_1fq
    "Retained Earnings (-2FQ)"                         TEXT, -- alias: retained_earnings_2fq
    "Retained Earnings (-3FQ)"                         TEXT, -- alias: retained_earnings_3fq
    "Retained Earnings (-4FQ)"                         TEXT, -- alias: retained_earnings_4fq
    "Retained Earnings (-1FY)"                         TEXT, -- alias: retained_earnings_1fy
    "Retained Earnings (-2FY)"                         TEXT, -- alias: retained_earnings_2fy
    "Retained Earnings (-3FY)"                         TEXT, -- alias: retained_earnings_3fy
    "Retained Earnings (-4FY)"                         TEXT, -- alias: retained_earnings_4fy
    "R&D Expenses (FQ)"                                TEXT, -- alias: randd_expenses_fq
    "R&D Expenses (FY)"                                TEXT, -- alias: randd_expenses_fy
    "R&D Expenses (-1FQFQ)"                            TEXT, -- alias: randd_expenses_1fqfq
    "R&D Expenses (-2FQFQ)"                            TEXT, -- alias: randd_expenses_2fqfq
    "R&D Expenses (-3FQFQ)"                            TEXT, -- alias: randd_expenses_3fqfq
    "R&D Expenses (-4FQFQ)"                            TEXT, -- alias: randd_expenses_4fqfq
    "R&D Expenses (-1FY)"                              TEXT, -- alias: randd_expenses_1fy
    "R&D Expenses (-2FY)"                              TEXT, -- alias: randd_expenses_2fy
    "R&D Expenses (-4FY)"                              TEXT, -- alias: randd_expenses_4fy
    "R&D Expenses (-3FY)"                              TEXT, -- alias: randd_expenses_3fy
    "Merger & Restructuring Charges (-1FQFQ)"          TEXT, -- alias: merger_and_restructuring_charges_1fqfq
    "Merger & Restructuring Charges (-3FQFQ)"          TEXT, -- alias: merger_and_restructuring_charges_3fqfq
    "Merger & Restructuring Charges (-4FQFQ)"          TEXT, -- alias: merger_and_restructuring_charges_4fqfq
    "Merger & Restructuring Charges (-2FQFQ)"          TEXT, -- alias: merger_and_restructuring_charges_2fqfq
    "Merger & Restructuring Charges (-1FY)"            TEXT, -- alias: merger_and_restructuring_charges_1fy
    "Merger & Restructuring Charges (-3FY)"            TEXT, -- alias: merger_and_restructuring_charges_3fy
    "Merger & Restructuring Charges (-4FY)"            TEXT, -- alias: merger_and_restructuring_charges_4fy
    "Merger & Restructuring Charges (-2FY)"            TEXT, -- alias: merger_and_restructuring_charges_2fy
    "Cash And Equivalents (-1FQ)"                      TEXT, -- alias: cash_and_equivalents_1fq
    "Cash And Equivalents (-3FQ)"                      TEXT, -- alias: cash_and_equivalents_3fq
    "Cash And Equivalents (-4FQ)"                      TEXT, -- alias: cash_and_equivalents_4fq
    "Cash And Equivalents (-2FQ)"                      TEXT, -- alias: cash_and_equivalents_2fq
    "Cash And Equivalents (-1FY)"                      TEXT, -- alias: cash_and_equivalents_1fy
    "Cash And Equivalents (-2FY)"                      TEXT, -- alias: cash_and_equivalents_2fy
    "Cash And Equivalents (-3FY)"                      TEXT, -- alias: cash_and_equivalents_3fy
    "Cash And Equivalents (-4FY)"                      TEXT, -- alias: cash_and_equivalents_4fy
    "Gross Intangible Assets (FQ)"                     TEXT, -- alias: gross_intangible_assets_fq
    "Gross Intangible Assets (-1FQ)"                   TEXT, -- alias: gross_intangible_assets_1fq
    "Gross Intangible Assets (-3FQ)"                   TEXT, -- alias: gross_intangible_assets_3fq
    "Gross Intangible Assets (-4FQ)"                   TEXT, -- alias: gross_intangible_assets_4fq
    "Gross Intangible Assets (-2FQ)"                   TEXT, -- alias: gross_intangible_assets_2fq
    "Gross Intangible Assets (-1FY)"                   TEXT, -- alias: gross_intangible_assets_1fy
    "Gross Intangible Assets (-2FY)"                   TEXT, -- alias: gross_intangible_assets_2fy
    "Gross Intangible Assets (-3FY)"                   TEXT, -- alias: gross_intangible_assets_3fy
    "Gross Intangible Assets (-4FY)"                   TEXT, -- alias: gross_intangible_assets_4fy
    "Total Revenues (-1FQFQ)"                          TEXT, -- alias: total_revenues_1fqfq
    "Total Revenues (-2FQFQ)"                          TEXT, -- alias: total_revenues_2fqfq
    "Total Revenues (-3FQFQ)"                          TEXT, -- alias: total_revenues_3fqfq
    "Total Revenues (-4FQFQ)"                          TEXT, -- alias: total_revenues_4fqfq
    "Total Revenues (-2FY)"                            TEXT, -- alias: total_revenues_2fy
    "Total Revenues (-3FY)"                            TEXT, -- alias: total_revenues_3fy
    "Total Revenues (-4FY)"                            TEXT, -- alias: total_revenues_4fy
    "Interest And Investment Income (LTM)"             TEXT, -- alias: interest_and_investment_income_ltm
    "Interest And Investment Income (FQ)"              TEXT, -- alias: interest_and_investment_income_fq
    "Interest And Investment Income (FY)"              TEXT, -- alias: interest_and_investment_income_fy
    "Interest And Investment Income (-1FQFQ)"          TEXT, -- alias: interest_and_investment_income_1fqfq
    "Interest And Investment Income (-2FQFQ)"          TEXT, -- alias: interest_and_investment_income_2fqfq
    "Interest And Investment Income (-3FQFQ)"          TEXT, -- alias: interest_and_investment_income_3fqfq
    "Interest And Investment Income (-4FQFQ)"          TEXT, -- alias: interest_and_investment_income_4fqfq
    "Interest And Investment Income (-1FY)"            TEXT, -- alias: interest_and_investment_income_1fy
    "Interest And Investment Income (-2FY)"            TEXT, -- alias: interest_and_investment_income_2fy
    "Interest And Investment Income (-3FY)"            TEXT, -- alias: interest_and_investment_income_3fy
    "Interest And Investment Income (-4FY)"            TEXT, -- alias: interest_and_investment_income_4fy
    "Effective Tax Rate - (Ratio) (LTM)"               TEXT,
    "Effective Tax Rate - (Ratio) (FQ)"                TEXT,
    "Effective Tax Rate - (Ratio) (-1FQFQ)"            TEXT,
    "Effective Tax Rate - (Ratio) (-2FQFQ)"            TEXT,
    "Effective Tax Rate - (Ratio) (-4FQFQ)"            TEXT,
    "Effective Tax Rate - (Ratio) (-3FQFQ)"            TEXT,
    "Effective Tax Rate - (Ratio) (FY)"                TEXT,
    "Effective Tax Rate - (Ratio) (-1FY)"              TEXT,
    "Effective Tax Rate - (Ratio) (-2FY)"              TEXT,
    "Effective Tax Rate - (Ratio) (-3FY)"              TEXT,
    "Effective Tax Rate - (Ratio) (-4FY)"              TEXT,
    "FCF - Est Avg (FY1E)"                             TEXT, -- alias: fcf_est_avg_fy1e
    "FCF - Est Avg (FY2E)"                             TEXT, -- alias: fcf_est_avg_fy2e
    "FCF - Est Avg (FY3E)"                             TEXT, -- alias: fcf_est_avg_fy3e
    "FCF - Est Avg (FY4E)"                             TEXT, -- alias: fcf_est_avg_fy4e
    "FCF - Est Avg (FY5E)"                             TEXT,  -- alias: fcf_est_avg_fy5e
    "Total Operating Expenses (FQ)"                    TEXT, -- alias: total_operating_expenses_fq
    "Total Operating Expenses (FY)"                    TEXT, -- alias: total_operating_expenses_fy
    "Total Operating Expenses (-1FQFQ)"                TEXT, -- alias: total_operating_expenses_1fqfq
    "Total Operating Expenses (-2FQFQ)"                TEXT, -- alias: total_operating_expenses_2fqfq
    "Total Operating Expenses (-3FQFQ)"                TEXT, -- alias: total_operating_expenses_3fqfq
    "Total Operating Expenses (-4FQFQ)"                TEXT, -- alias: total_operating_expenses_4fqfq
    "Total Operating Expenses (-1FY)"                  TEXT, -- alias: total_operating_expenses_1fy
    "Total Operating Expenses (-2FY)"                  TEXT, -- alias: total_operating_expenses_2fy
    "Total Operating Expenses (-3FY)"                  TEXT, -- alias: total_operating_expenses_3fy
    "Total Operating Expenses (-4FY)"                  TEXT -- alias: total_operating_expenses_4fy
);
-- ===================================================================
-- DATA IMPORT EXECUTION
-- ===================================================================

-- US Region
\echo 'Importing US data...'
\copy screening_staging FROM 'data/screening_us.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description"));

-- EU Region
\echo 'Importing EU data...'
\copy screening_staging FROM 'data/screening_eu.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description"));

-- APAC Region
\echo 'Importing APAC data...'
\copy screening_staging FROM 'data/screening_apac.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description"));

-- ROTW Region
\echo 'Importing ROTW data...'
\copy screening_staging FROM 'data/screening_rotw.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description"));

-- ===================================================================
-- DATA VALIDATION (PRE-INSERT)
-- ===================================================================
\echo 'Validating imported data...'
SELECT 'Total rows in staging:' AS info, COUNT(*) AS count
FROM screening_staging;

TRUNCATE TABLE equities;
INSERT INTO equities ("Ticker", -- alias: ticker
                      "ISIN", -- alias: isin
                      "Name", -- alias: name
                      "Description", -- alias: description
                      "Region", -- alias: region
                      "Country", -- alias: country
                      "Trading Country", -- alias: trading_country
                      "Exchange", -- alias: exchange
                      "Unit", -- alias: unit
                      "Sector", -- alias: sector
                      "Industry", -- alias: industry
                      "Style Class", -- alias: style_class
                      "Size Class", -- alias: size_class
                      "FY End", -- alias: fy_end
                      "Next Earnings (When)", -- alias: next_earnings_when
                      "Next Earnings (Status)", -- alias: next_earnings_status
                      "Dividend Record (Currency)", -- alias: dividend_record_currency
                      "Dividend Record (Frequency)", -- alias: dividend_record_frequency
                      "Current Fiscal Quarter", -- alias: current_fiscal_quarter
                      "Next Fiscal Quarter", -- alias: next_fiscal_quarter
                      "Next Earnings (Report)", -- alias: next_earnings_report
                      "Reporting Interval", -- alias: reporting_interval
                      "Earnings Report (Frequency)", -- alias: earnings_report_frequency
                      "Last Updated", -- alias: last_updated
                      "Income Statement Report Date", -- alias: income_statement_report_date
                      "Next Earnings", -- alias: next_earnings
                      "Dividend Record (Announce Date)", -- alias: dividend_record_announce_date
                      "Dividend Record (Payable Date)", -- alias: dividend_record_payable_date
                      "Dividend Record (Record Date)", -- alias: dividend_record_record_date
                      "Dividend Record (Ex Date)", -- alias: dividend_record_ex_date
                      "Reference Date", -- alias: reference_date
                      "FY End Date", -- alias: fy_end_date
                      "Next FY End Date", -- alias: next_fy_end_date
                      "Next Income Statement Report Date", -- alias: next_income_statement_report_date
                      "Price Target", -- alias: price_target
                      "Price Target - Median", -- alias: price_target_median
                      "Dividend Record (Amount)", -- alias: dividend_record_amount
                      "Market Cap", -- alias: market_cap
                      "Enterprise Value", -- alias: enterprise_value
                      "Last Price", -- alias: last_price
                      "Price Target (YTD Ago)", -- alias: price_target_ytd_ago
                      "Price Target - Low", -- alias: price_target_low
                      "Price Target - High", -- alias: price_target_high
                      "Market Cap (Country R)", -- alias: market_cap_country_r
                      "Volume (Shrs)", -- alias: volume_shrs
                      "Dividend Per Share (LTM)", -- alias: dividend_per_share_ltm
                      "Price (5D Ago)", -- alias: price_5d_ago
                      "Price (1W Ago)", -- alias: price_1w_ago
                      "Price (1M Ago)", -- alias: price_1m_ago
                      "Price (3M Ago)", -- alias: price_3m_ago
                      "Price (6M Ago)", -- alias: price_6m_ago
                      "Price (1Y Ago)", -- alias: price_1y_ago
                      "Price (3Y Ago)", -- alias: price_3y_ago
                      "Price (5Y Ago)", -- alias: price_5y_ago
                      "Price (QTD Ago)", -- alias: price_qtd_ago
                      "Rel. Volume", -- alias: rel_volume
                      "52W High/Adj", -- alias: 52w_high_adj
                      "52W Low/Adj", -- alias: 52w_low_adj
                      "EMA (20D)", -- alias: ema_20d
                      "EMA (50D)", -- alias: ema_50d
                      "EMA (100D)", -- alias: ema_100d
                      "EMA (250D)", -- alias: ema_250d
                      "Price Target (1W Ago)", -- alias: price_target_1w_ago
                      "Price Target (1M Ago)", -- alias: price_target_1m_ago
                      "Price Target (3M Ago)", -- alias: price_target_3m_ago
                      "Price Target (6M Ago)", -- alias: price_target_6m_ago
                      "Price Target (MTD Ago)", -- alias: price_target_mtd_ago
                      "Price Target (QTD Ago)", -- alias: price_target_qtd_ago
                      "Price Target (1Y Ago)", -- alias: price_target_1y_ago
                      "Price Target - High (1W Ago)", -- alias: price_target_high_1w_ago
                      "Price Target - High (1M Ago)", -- alias: price_target_high_1m_ago
                      "Price Target - High (6M Ago)", -- alias: price_target_high_6m_ago
                      "Price Target - High (MTD Ago)", -- alias: price_target_high_mtd_ago
                      "Price Target - High (3M Ago)", -- alias: price_target_high_3m_ago
                      "Price Target - High (QTD Ago)", -- alias: price_target_high_qtd_ago
                      "Price Target - High (1Y Ago)", -- alias: price_target_high_1y_ago
                      "Price Target - High (YTD Ago)", -- alias: price_target_high_ytd_ago
                      "Price Target - Low (1W Ago)", -- alias: price_target_low_1w_ago
                      "Price Target - Low (1M Ago)", -- alias: price_target_low_1m_ago
                      "Price Target - Low (3M Ago)", -- alias: price_target_low_3m_ago
                      "Price Target - Low (6M Ago)", -- alias: price_target_low_6m_ago
                      "Price Target - Low (MTD Ago)", -- alias: price_target_low_mtd_ago
                      "Price Target - Low (QTD Ago)", -- alias: price_target_low_qtd_ago
                      "Price Target - Low (YTD Ago)", -- alias: price_target_low_ytd_ago
                      "Price Target - Low (1Y Ago)", -- alias: price_target_low_1y_ago
                      "Price Target - Median (1W Ago)", -- alias: price_target_median_1w_ago
                      "Price Target - Median (1M Ago)", -- alias: price_target_median_1m_ago
                      "Price Target - Median (3M Ago)", -- alias: price_target_median_3m_ago
                      "Price Target - Median (6M Ago)", -- alias: price_target_median_6m_ago
                      "Price Target - Median (MTD Ago)", -- alias: price_target_median_mtd_ago
                      "Price Target - Median (QTD Ago)", -- alias: price_target_median_qtd_ago
                      "Price Target - Median (YTD Ago)", -- alias: price_target_median_ytd_ago
                      "Price Target - Median (1Y Ago)", -- alias: price_target_median_1y_ago
                      "Total Revenues (FQ)", -- alias: total_revenues_fq
                      "Total Revenues (-1FY)", -- alias: total_revenues_1fy
                      "Total Revenues (FY)", -- alias: total_revenues_fy
                      "Total Revenues (LTM)", -- alias: total_revenues_ltm
                      "Net Income/Adj. (-1FY)", -- alias: net_income_adj_1fy
                      "EBITDA (FQ)", -- alias: ebitda_fq
                      "EBITDA (LTM)", -- alias: ebitda_ltm
                      "EBITDA (FY)", -- alias: ebitda_fy
                      "EBITDA (-1FY)", -- alias: ebitda_1fy
                      "EBITDA/Adj. (LTM)", -- alias: ebitda_adj_ltm
                      "EBITDA/Adj. (FY)", -- alias: ebitda_adj_fy
                      "EBITDA/Adj. (-1FY)", -- alias: ebitda_adj_1fy
                      "EBIT (FQ)", -- alias: ebit_fq
                      "EBIT (LTM)", -- alias: ebit_ltm
                      "EBIT (FY)", -- alias: ebit_fy
                      "EBIT (-1FY)", -- alias: ebit_1fy
                      "EBIT/Adj. (-1FY)", -- alias: ebit_adj_1fy
                      "EBIT/Adj. (FY)", -- alias: ebit_adj_fy
                      "EBIT/Adj. (LTM)", -- alias: ebit_adj_ltm
                      "EBIT - Est Med (FY1E)", -- alias: ebit_est_med_fy1e
                      "EBIT - Est Med (NTM)", -- alias: ebit_est_med_ntm
                      "Net Income - (IS) (FY)", -- alias: net_income_is_fy
                      "Net Income - (IS) (LTM)", -- alias: net_income_is_ltm
                      "Normalized Net Income (FY)", -- alias: normalized_net_income_fy
                      "Normalized Net Income (LTM)", -- alias: normalized_net_income_ltm
                      "Net Income/Adj. (FY)", -- alias: net_income_adj_fy
                      "Net Income/Adj. (LTM)", -- alias: net_income_adj_ltm
                      "Gross Profit (LTM)", -- alias: gross_profit_ltm
                      "Gross Profit (FY)", -- alias: gross_profit_fy
                      "Cost Of Revenues (LTM)", -- alias: cost_of_revenues_ltm
                      "Operating Income (LTM)", -- alias: operating_income_ltm
                      "Operating Income (FY)", -- alias: operating_income_fy
                      "R&D Expenses (LTM)", -- alias: randd_expenses_ltm
                      "Interest Expense/Total (LTM)", -- alias: interest_expense_total_ltm
                      "Interest Income On Investments (LTM)", -- alias: interest_income_on_investments_ltm
                      "Net Income - (IS) (-1FY)", -- alias: net_income_is_1fy
                      "Normalized Net Income (-1FY)", -- alias: normalized_net_income_1fy
                      "Total Revenues (5YAVGFQ)", -- alias: total_revenues_5yavgfq
                      "EBITDA (5YAVGFQ)", -- alias: ebitda_5yavgfq
                      "EBIT (5YAVGFQ)", -- alias: ebit_5yavgfq
                      "Operating Income (FQ)", -- alias: operating_income_fq
                      "Operating Income (5YAVGFQ)", -- alias: operating_income_5yavgfq
                      "Normalized Net Income (FQ)", -- alias: normalized_net_income_fq
                      "Normalized Net Income (5YAVGFQ)", -- alias: normalized_net_income_5yavgfq
                      "Net Income/Adj. (FQ)", -- alias: net_income_adj_fq
                      "Net Income/Adj. (5YAVGFQ)", -- alias: net_income_adj_5yavgfq
                      "Net Income - (IS) (FQ)", -- alias: net_income_is_fq
                      "Net Income - (IS) (5YAVGFQ)", -- alias: net_income_is_5yavgfq
                      "Net Income - (IS) (5YAVGLTM)", -- alias: net_income_is_5yavgltm
                      "Normalized Net Income (5YAVGLTM)", -- alias: normalized_net_income_5yavgltm
                      "EBITDA (5YAVGLTM)", -- alias: ebitda_5yavgltm
                      "EBIT (5YAVGLTM)", -- alias: ebit_5yavgltm
                      "Total Revenues (5YAVGLTM)", -- alias: total_revenues_5yavgltm
                      "Selling General & Admin Expenses/Total (FQ)", -- alias: selling_general_and_admin_expenses_total_fq
                      "Selling General & Admin Expenses/Total (FY)", -- alias: selling_general_and_admin_expenses_total_fy
                      "Selling General & Admin Expenses/Total (-1FY)", -- alias: selling_general_and_admin_expenses_total_1fy
                      "Selling General & Admin Expenses/Total (5YAVGFQ)", -- alias: selling_general_and_admin_expenses_total_5yavgfq
                      "Marketing Expenses (FQ)", -- alias: marketing_expenses_fq
                      "Marketing Expenses (FY)", -- alias: marketing_expenses_fy
                      "Marketing Expenses (-1FY)", -- alias: marketing_expenses_1fy
                      "Marketing Expenses (5YAVGLTM)", -- alias: marketing_expenses_5yavgltm
                      "Revenues - Est Avg (NTM)", -- alias: revenues_est_avg_ntm
                      "Revenues - Est Avg (FY1E)", -- alias: revenues_est_avg_fy1e
                      "Revenues - Est Med (NTM)", -- alias: revenues_est_med_ntm
                      "Revenues - Est Med (FY1E)", -- alias: revenues_est_med_fy1e
                      "EBITDA - Est Avg (NTM)", -- alias: ebitda_est_avg_ntm
                      "EBITDA - Est Avg (FY1E)", -- alias: ebitda_est_avg_fy1e
                      "Total Revenues (-1FQFQ)", -- alias: total_revenues_1fqfq
                      "Total Revenues (-2FQFQ)", -- alias: total_revenues_2fqfq
                      "Total Revenues (-3FQFQ)", -- alias: total_revenues_3fqfq
                      "Total Revenues (-4FQFQ)", -- alias: total_revenues_4fqfq
                      "Total Revenues (-2FY)", -- alias: total_revenues_2fy
                      "Total Revenues (-3FY)", -- alias: total_revenues_3fy
                      "Total Revenues (-4FY)", -- alias: total_revenues_4fy
                      "Gross Profit (FQ)", -- alias: gross_profit_fq
                      "Gross Profit (-1FQFQ)", -- alias: gross_profit_1fqfq
                      "Gross Profit (-2FQFQ)", -- alias: gross_profit_2fqfq
                      "Gross Profit (-3FQFQ)", -- alias: gross_profit_3fqfq
                      "Gross Profit (-4FQFQ)", -- alias: gross_profit_4fqfq
                      "Gross Profit (-1FY)", -- alias: gross_profit_1fy
                      "Gross Profit (-2FY)", -- alias: gross_profit_2fy
                      "Gross Profit (-3FY)", -- alias: gross_profit_3fy
                      "Gross Profit (-4FY)", -- alias: gross_profit_4fy
                      "Operating Income (-1FQFQ)", -- alias: operating_income_1fqfq
                      "Operating Income (-2FQFQ)", -- alias: operating_income_2fqfq
                      "Operating Income (-3FQFQ)", -- alias: operating_income_3fqfq
                      "Operating Income (-4FQFQ)", -- alias: operating_income_4fqfq
                      "Operating Income (-1FY)", -- alias: operating_income_1fy
                      "Operating Income (-2FY)", -- alias: operating_income_2fy
                      "Operating Income (-3FY)", -- alias: operating_income_3fy
                      "Operating Income (-4FY)", -- alias: operating_income_4fy
                      "R&D Expenses (FQ)", -- alias: randd_expenses_fq
                      "R&D Expenses (FY)", -- alias: randd_expenses_fy
                      "R&D Expenses (-1FQFQ)", -- alias: randd_expenses_1fqfq
                      "R&D Expenses (-2FQFQ)", -- alias: randd_expenses_2fqfq
                      "R&D Expenses (-3FQFQ)", -- alias: randd_expenses_3fqfq
                      "R&D Expenses (-4FQFQ)", -- alias: randd_expenses_4fqfq
                      "R&D Expenses (-1FY)", -- alias: randd_expenses_1fy
                      "R&D Expenses (-2FY)", -- alias: randd_expenses_2fy
                      "R&D Expenses (-3FY)", -- alias: randd_expenses_3fy
                      "R&D Expenses (-4FY)", -- alias: randd_expenses_4fy
                      "Net Income - (IS) (-1FQFQ)", -- alias: net_income_is_1fqfq
                      "Net Income - (IS) (-2FQFQ)", -- alias: net_income_is_2fqfq
                      "Net Income - (IS) (-3FQFQ)", -- alias: net_income_is_3fqfq
                      "Net Income - (IS) (-4FQFQ)", -- alias: net_income_is_4fqfq
                      "Net Income - (IS) (-2FY)", -- alias: net_income_is_2fy
                      "Net Income - (IS) (-3FY)", -- alias: net_income_is_3fy
                      "Net Income - (IS) (-4FY)", -- alias: net_income_is_4fy
                      "Normalized Net Income (-1FQFQ)", -- alias: normalized_net_income_1fqfq
                      "Normalized Net Income (-2FQFQ)", -- alias: normalized_net_income_2fqfq
                      "Normalized Net Income (-3FQFQ)", -- alias: normalized_net_income_3fqfq
                      "Normalized Net Income (-4FQFQ)", -- alias: normalized_net_income_4fqfq
                      "Normalized Net Income (-2FY)", -- alias: normalized_net_income_2fy
                      "Normalized Net Income (-3FY)", -- alias: normalized_net_income_3fy
                      "Normalized Net Income (-4FY)", -- alias: normalized_net_income_4fy
                      "Net Income/Adj. (-1FQFQ)", -- alias: net_income_adj_1fqfq
                      "Net Income/Adj. (-2FQFQ)", -- alias: net_income_adj_2fqfq
                      "Net Income/Adj. (-3FQFQ)", -- alias: net_income_adj_3fqfq
                      "Net Income/Adj. (-4FQFQ)", -- alias: net_income_adj_4fqfq
                      "Net Income/Adj. (-2FY)", -- alias: net_income_adj_2fy
                      "Net Income/Adj. (-3FY)", -- alias: net_income_adj_3fy
                      "Net Income/Adj. (-4FY)", -- alias: net_income_adj_4fy
                      "EBIT (-1FQFQ)", -- alias: ebit_1fqfq
                      "EBIT (-2FQFQ)", -- alias: ebit_2fqfq
                      "EBIT (-3FQFQ)", -- alias: ebit_3fqfq
                      "EBIT (-4FQFQ)", -- alias: ebit_4fqfq
                      "EBIT (-2FY)", -- alias: ebit_2fy
                      "EBIT (-3FY)", -- alias: ebit_3fy
                      "EBIT (-4FY)", -- alias: ebit_4fy
                      "EBIT/Adj. (FQ)", -- alias: ebit_adj_fq
                      "EBIT/Adj. (-1FQFQ)", -- alias: ebit_adj_1fqfq
                      "EBIT/Adj. (-2FQFQ)", -- alias: ebit_adj_2fqfq
                      "EBIT/Adj. (-3FQFQ)", -- alias: ebit_adj_3fqfq
                      "EBIT/Adj. (-4FQFQ)", -- alias: ebit_adj_4fqfq
                      "EBIT/Adj. (-2FY)", -- alias: ebit_adj_2fy
                      "EBIT/Adj. (-3FY)", -- alias: ebit_adj_3fy
                      "EBIT/Adj. (-4FY)", -- alias: ebit_adj_4fy
                      "EBITDA (-1FQFQ)", -- alias: ebitda_1fqfq
                      "EBITDA (-2FQFQ)", -- alias: ebitda_2fqfq
                      "EBITDA (-3FQFQ)", -- alias: ebitda_3fqfq
                      "EBITDA (-4FQFQ)", -- alias: ebitda_4fqfq
                      "EBITDA (-2FY)", -- alias: ebitda_2fy
                      "EBITDA (-3FY)", -- alias: ebitda_3fy
                      "EBITDA (-4FY)", -- alias: ebitda_4fy
                      "EBITDA/Adj. (FQ)", -- alias: ebitda_adj_fq
                      "EBITDA/Adj. (-1FQFQ)", -- alias: ebitda_adj_1fqfq
                      "EBITDA/Adj. (-2FQFQ)", -- alias: ebitda_adj_2fqfq
                      "EBITDA/Adj. (-3FQFQ)", -- alias: ebitda_adj_3fqfq
                      "EBITDA/Adj. (-4FQFQ)", -- alias: ebitda_adj_4fqfq
                      "EBITDA/Adj. (-2FY)", -- alias: ebitda_adj_2fy
                      "EBITDA/Adj. (-3FY)", -- alias: ebitda_adj_3fy
                      "EBITDA/Adj. (-4FY)", -- alias: ebitda_adj_4fy
                      "TBV (FY)", -- alias: tbv_fy
                      "TBV (LTM)", -- alias: tbv_ltm
                      "Total Debt (FY)", -- alias: total_debt_fy
                      "Total Equity (FY)", -- alias: total_equity_fy
                      "Total Equity (LTM)", -- alias: total_equity_ltm
                      "Total Debt (LTM)", -- alias: total_debt_ltm
                      "Total Assets (LTM)", -- alias: total_assets_ltm
                      "Total Assets (FY)", -- alias: total_assets_fy
                      "Inventory (LTM)", -- alias: inventory_ltm
                      "Goodwill (FQ)", -- alias: goodwill_fq
                      "Goodwill (LTM)", -- alias: goodwill_ltm
                      "Goodwill (FY)", -- alias: goodwill_fy
                      "Goodwill (-1FY)", -- alias: goodwill_1fy
                      "Retained Earnings (LTM)", -- alias: retained_earnings_ltm
                      "Total Current Assets (LTM)", -- alias: total_current_assets_ltm
                      "Total Current Liabilities (LTM)", -- alias: total_current_liabilities_ltm
                      "Working Capital (LTM)", -- alias: working_capital_ltm
                      "Cash And Equivalents (LTM)", -- alias: cash_and_equivalents_ltm
                      "Cash And Equivalents (FQ)", -- alias: cash_and_equivalents_fq
                      "Cash And Equivalents (FY)", -- alias: cash_and_equivalents_fy
                      "Cash And Equivalents (5YAVGFQ)", -- alias: cash_and_equivalents_5yavgfq
                      "Inventory (FQ)", -- alias: inventory_fq
                      "Inventory (FY)", -- alias: inventory_fy
                      "Goodwill (5YAVGFQ)", -- alias: goodwill_5yavgfq
                      "Inventory (5YAVGFQ)", -- alias: inventory_5yavgfq
                      "Retained Earnings (FQ)", -- alias: retained_earnings_fq
                      "Retained Earnings (FY)", -- alias: retained_earnings_fy
                      "Retained Earnings (5YAVGFQ)", -- alias: retained_earnings_5yavgfq
                      "Working Capital (FQ)", -- alias: working_capital_fq
                      "Working Capital (FY)", -- alias: working_capital_fy
                      "Working Capital (5YAVGFY)", -- alias: working_capital_5yavgfy
                      "Gross Intangible Assets (LTM)", -- alias: gross_intangible_assets_ltm
                      "Gross Intangible Assets (FY)", -- alias: gross_intangible_assets_fy
                      "Gross Intangible Assets (5YAVGFQ)", -- alias: gross_intangible_assets_5yavgfq
                      "Accounts Receivable/Total (FY)", -- alias: accounts_receivable_total_fy
                      "Accounts Receivable/Total (-1FY)", -- alias: accounts_receivable_total_1fy
                      "Accounts Receivable/Total (5YAVGFQ)", -- alias: accounts_receivable_total_5yavgfq
                      "Working Capital (-1FQ)", -- alias: working_capital_1fq
                      "Working Capital (-2FQ)", -- alias: working_capital_2fq
                      "Working Capital (-3FQ)", -- alias: working_capital_3fq
                      "Working Capital (-4FQ)", -- alias: working_capital_4fq
                      "Working Capital (-1FY)", -- alias: working_capital_1fy
                      "Working Capital (-2FY)", -- alias: working_capital_2fy
                      "Working Capital (-3FY)", -- alias: working_capital_3fy
                      "Working Capital (-4FY)", -- alias: working_capital_4fy
                      "Total Debt (FQ)", -- alias: total_debt_fq
                      "Total Debt (-1FQ)", -- alias: total_debt_1fq
                      "Total Debt (-2FQ)", -- alias: total_debt_2fq
                      "Total Debt (-3FQ)", -- alias: total_debt_3fq
                      "Total Debt (-4FQ)", -- alias: total_debt_4fq
                      "Total Debt (-1FY)", -- alias: total_debt_1fy
                      "Total Debt (-2FY)", -- alias: total_debt_2fy
                      "Total Debt (-3FY)", -- alias: total_debt_3fy
                      "Total Debt (-4FY)", -- alias: total_debt_4fy
                      "Total Assets (FQ)", -- alias: total_assets_fq
                      "Total Assets (-1FQ)", -- alias: total_assets_1fq
                      "Total Assets (-2FQ)", -- alias: total_assets_2fq
                      "Total Assets (-3FQ)", -- alias: total_assets_3fq
                      "Total Assets (-4FQ)", -- alias: total_assets_4fq
                      "Total Assets (-1FY)", -- alias: total_assets_1fy
                      "Total Assets (-2FY)", -- alias: total_assets_2fy
                      "Total Assets (-3FY)", -- alias: total_assets_3fy
                      "Total Assets (-4FY)", -- alias: total_assets_4fy
                      "Inventory (-1FQ)", -- alias: inventory_1fq
                      "Inventory (-2FQ)", -- alias: inventory_2fq
                      "Inventory (-3FQ)", -- alias: inventory_3fq
                      "Inventory (-4FQ)", -- alias: inventory_4fq
                      "Inventory (-1FY)", -- alias: inventory_1fy
                      "Inventory (-2FY)", -- alias: inventory_2fy
                      "Inventory (-3FY)", -- alias: inventory_3fy
                      "Inventory (-4FY)", -- alias: inventory_4fy
                      "Goodwill (-1FQ)", -- alias: goodwill_1fq
                      "Goodwill (-2FQ)", -- alias: goodwill_2fq
                      "Goodwill (-3FQ)", -- alias: goodwill_3fq
                      "Goodwill (-4FQ)", -- alias: goodwill_4fq
                      "Goodwill (-2FY)", -- alias: goodwill_2fy
                      "Goodwill (-3FY)", -- alias: goodwill_3fy
                      "Goodwill (-4FY)", -- alias: goodwill_4fy
                      "Retained Earnings (-1FQ)", -- alias: retained_earnings_1fq
                      "Retained Earnings (-2FQ)", -- alias: retained_earnings_2fq
                      "Retained Earnings (-3FQ)", -- alias: retained_earnings_3fq
                      "Retained Earnings (-4FQ)", -- alias: retained_earnings_4fq
                      "Retained Earnings (-1FY)", -- alias: retained_earnings_1fy
                      "Retained Earnings (-2FY)", -- alias: retained_earnings_2fy
                      "Retained Earnings (-3FY)", -- alias: retained_earnings_3fy
                      "Retained Earnings (-4FY)", -- alias: retained_earnings_4fy
                      "Cash And Equivalents (-1FQ)", -- alias: cash_and_equivalents_1fq
                      "Cash And Equivalents (-2FQ)", -- alias: cash_and_equivalents_2fq
                      "Cash And Equivalents (-3FQ)", -- alias: cash_and_equivalents_3fq
                      "Cash And Equivalents (-4FQ)", -- alias: cash_and_equivalents_4fq
                      "Cash And Equivalents (-1FY)", -- alias: cash_and_equivalents_1fy
                      "Cash And Equivalents (-2FY)", -- alias: cash_and_equivalents_2fy
                      "Cash And Equivalents (-3FY)", -- alias: cash_and_equivalents_3fy
                      "Cash And Equivalents (-4FY)", -- alias: cash_and_equivalents_4fy
                      "Gross Intangible Assets (FQ)", -- alias: gross_intangible_assets_fq
                      "Gross Intangible Assets (-1FQ)", -- alias: gross_intangible_assets_1fq
                      "Gross Intangible Assets (-2FQ)", -- alias: gross_intangible_assets_2fq
                      "Gross Intangible Assets (-3FQ)", -- alias: gross_intangible_assets_3fq
                      "Gross Intangible Assets (-4FQ)", -- alias: gross_intangible_assets_4fq
                      "Gross Intangible Assets (-1FY)", -- alias: gross_intangible_assets_1fy
                      "Gross Intangible Assets (-2FY)", -- alias: gross_intangible_assets_2fy
                      "Gross Intangible Assets (-3FY)", -- alias: gross_intangible_assets_3fy
                      "Gross Intangible Assets (-4FY)", -- alias: gross_intangible_assets_4fy
                      "CFF (LTM)", -- alias: cff_ltm
                      "CFI (LTM)", -- alias: cfi_ltm
                      "FCF (LTM)", -- alias: fcf_ltm
                      "CFO (LTM)", -- alias: cfo_ltm
                      "Cash Acquisitions (LTM)", -- alias: cash_acquisitions_ltm
                      "Cash Acquisitions (FY)", -- alias: cash_acquisitions_fy
                      "Cash Acquisitions (-1FY)", -- alias: cash_acquisitions_1fy
                      "Capital Expenditure (LTM)", -- alias: capital_expenditure_ltm
                      "Capital Expenditure (-1FY)", -- alias: capital_expenditure_1fy
                      "Capital Expenditure (FY)", -- alias: capital_expenditure_fy
                      "CFF (FY)", -- alias: cff_fy
                      "CFF (-1FY)", -- alias: cff_1fy
                      "CFI (FY)", -- alias: cfi_fy
                      "CFI (-1FY)", -- alias: cfi_1fy
                      "CFO (FY)", -- alias: cfo_fy
                      "CFO (-1FY)", -- alias: cfo_1fy
                      "FCF (FY)", -- alias: fcf_fy
                      "FCF (-1FY)", -- alias: fcf_1fy
                      "Capital Expenditure (FQ)", -- alias: capital_expenditure_fq
                      "Capital Expenditure (5YAVGFQ)", -- alias: capital_expenditure_5yavgfq
                      "CFF (FQ)", -- alias: cff_fq
                      "CFI (FQ)", -- alias: cfi_fq
                      "CFO (FQ)", -- alias: cfo_fq
                      "FCF (FQ)", -- alias: fcf_fq
                      "FCF (5YAVGFQ)", -- alias: fcf_5yavgfq
                      "Cash Acquisitions (FQ)", -- alias: cash_acquisitions_fq
                      "Cash Acquisitions (5YAVGFQ)", -- alias: cash_acquisitions_5yavgfq
                      "Common Dividends Paid (LTM)", -- alias: common_dividends_paid_ltm
                      "Common Dividends Paid (FY)", -- alias: common_dividends_paid_fy
                      "CFO (-1FQFQ)", -- alias: cfo_1fqfq
                      "CFO (-2FQFQ)", -- alias: cfo_2fqfq
                      "CFO (-3FQFQ)", -- alias: cfo_3fqfq
                      "CFO (-4FQFQ)", -- alias: cfo_4fqfq
                      "CFI (-1FQFQ)", -- alias: cfi_1fqfq
                      "CFI (-2FQFQ)", -- alias: cfi_2fqfq
                      "CFI (-3FQFQ)", -- alias: cfi_3fqfq
                      "CFI (-4FQFQ)", -- alias: cfi_4fqfq
                      "CFI (-2FY)", -- alias: cfi_2fy
                      "CFI (-3FY)", -- alias: cfi_3fy
                      "CFI (-4FY)", -- alias: cfi_4fy
                      "FCF (-1FQFQ)", -- alias: fcf_1fqfq
                      "FCF (-2FQFQ)", -- alias: fcf_2fqfq
                      "FCF (-3FQFQ)", -- alias: fcf_3fqfq
                      "FCF (-4FQFQ)", -- alias: fcf_4fqfq
                      "CFF (-2FY)", -- alias: cff_2fy
                      "CFF (-3FY)", -- alias: cff_3fy
                      "CFF (-4FY)", -- alias: cff_4fy
                      "CFF (-1FQFQ)", -- alias: cff_1fqfq
                      "CFF (-2FQFQ)", -- alias: cff_2fqfq
                      "CFF (-3FQFQ)", -- alias: cff_3fqfq
                      "CFF (-4FQFQ)", -- alias: cff_4fqfq
                      "CFO (-2FY)", -- alias: cfo_2fy
                      "CFO (-3FY)", -- alias: cfo_3fy
                      "CFO (-4FY)", -- alias: cfo_4fy
                      "Cash Acquisitions (-1FQFQ)", -- alias: cash_acquisitions_1fqfq
                      "Cash Acquisitions (-2FQFQ)", -- alias: cash_acquisitions_2fqfq
                      "Cash Acquisitions (-3FQFQ)", -- alias: cash_acquisitions_3fqfq
                      "Cash Acquisitions (-4FQFQ)", -- alias: cash_acquisitions_4fqfq
                      "FCF (-2FY)", -- alias: fcf_2fy
                      "FCF (-3FY)", -- alias: fcf_3fy
                      "FCF (-4FY)", -- alias: fcf_4fy
                      "Cash Acquisitions (-2FY)", -- alias: cash_acquisitions_2fy
                      "Cash Acquisitions (-3FY)", -- alias: cash_acquisitions_3fy
                      "Cash Acquisitions (-4FY)", -- alias: cash_acquisitions_4fy
                      "Capital Expenditure (-1FQFQ)", -- alias: capital_expenditure_1fqfq
                      "Capital Expenditure (-2FQFQ)", -- alias: capital_expenditure_2fqfq
                      "Capital Expenditure (-3FQFQ)", -- alias: capital_expenditure_3fqfq
                      "Capital Expenditure (-4FQFQ)", -- alias: capital_expenditure_4fqfq
                      "Capital Expenditure (-2FY)", -- alias: capital_expenditure_2fy
                      "Capital Expenditure (-3FY)", -- alias: capital_expenditure_3fy
                      "Capital Expenditure (-4FY)", -- alias: capital_expenditure_4fy
                      "P/E (NTM)", -- alias: p_e_ntm
                      "P/E (LTM)", -- alias: p_e_ltm
                      "Altman Z-Score (FY)", -- alias: altman_z_score_fy
                      "Altman Z-Score (FQ)", -- alias: altman_z_score_fq
                      "Altman Z-Score (LTM)", -- alias: altman_z_score_ltm
                      "P/TBV (LTM)", -- alias: p_tbv_ltm
                      "Return On Equity % (LTM)", -- alias: return_on_equity_pct_ltm
                      "Return On Equity % (FY)", -- alias: return_on_equity_pct_fy
                      "Current Ratio (FY)", -- alias: current_ratio_fy
                      "Current Ratio (LTM)", -- alias: current_ratio_ltm
                      "Asset Turnover (FY)", -- alias: asset_turnover_fy
                      "Asset Turnover (LTM)", -- alias: asset_turnover_ltm
                      "EPS Norm - Est Avg (NTM)", -- alias: eps_norm_est_avg_ntm
                      "EPS/Adj. (-1FY)", -- alias: eps_adj_1fy
                      "EPS/Adj. (FY)", -- alias: eps_adj_fy
                      "EPS/Adj. (LTM)", -- alias: eps_adj_ltm
                      "EPS Norm - Est Avg (FY1E)", -- alias: eps_norm_est_avg_fy1e
                      "Return on Assets (ROA) % (LTM)", -- alias: return_on_assets_roa_pct_ltm
                      "Return on Assets (ROA) % (FY)", -- alias: return_on_assets_roa_pct_fy
                      "P/B (LTM)", -- alias: p_b_ltm
                      "P/B (-1FY)", -- alias: p_b_1fy
                      "P/B (5YAVG)", -- alias: p_b_5yavg
                      "EV/Sales (EST FY1)", -- alias: ev_sales_est_fy1
                      "EV/Sales (LTM)", -- alias: ev_sales_ltm
                      "EV/Sales (NTM)", -- alias: ev_sales_ntm
                      "EV/Sales (-1FYLTM)", -- alias: ev_sales_1fyltm
                      "EV/Sales (-2FYLTM)", -- alias: ev_sales_2fyltm
                      "EV/Sales (-3FYLTM)", -- alias: ev_sales_3fyltm
                      "EV/Sales (3YAVGLTM)", -- alias: ev_sales_3yavgltm
                      "EV/Sales (-1FQLTM)", -- alias: ev_sales_1fqltm
                      "EV/Sales (-2FQLTM)", -- alias: ev_sales_2fqltm
                      "EV/Sales (-3FQLTM)", -- alias: ev_sales_3fqltm
                      "EV/Sales (-4FQLTM)", -- alias: ev_sales_4fqltm
                      "EV/EBITDA (LTM)", -- alias: ev_ebitda_ltm
                      "EV/EBITDA (NTM)", -- alias: ev_ebitda_ntm
                      "EV/EBITDA (-1FYLTM)", -- alias: ev_ebitda_1fyltm
                      "EV/EBITDA (-1FQLTM)", -- alias: ev_ebitda_1fqltm
                      "EV/EBITDA (3YAVGLTM)", -- alias: ev_ebitda_3yavgltm
                      "EV/EBITDA (EST FY1)", -- alias: ev_ebitda_est_fy1
                      "P/E (EST FY1)", -- alias: p_e_est_fy1
                      "P/E (-1FYLTM)", -- alias: p_e_1fyltm
                      "P/E (-2FYLTM)", -- alias: p_e_2fyltm
                      "P/E (-3FYLTM)", -- alias: p_e_3fyltm
                      "P/E (3YAVGLTM)", -- alias: p_e_3yavgltm
                      "P/E (-1FQLTM)", -- alias: p_e_1fqltm
                      "P/E (-2FQLTM)", -- alias: p_e_2fqltm
                      "P/E (-3FQLTM)", -- alias: p_e_3fqltm
                      "P/E (5YAVGLTM)", -- alias: p_e_5yavgltm
                      "P/E (-0FQQoQLTM)", -- alias: p_e_0fqqoqltm
                      "P/E (-0FYYoYLTM)", -- alias: p_e_0fyyoyltm
                      "P/E (-1FYYoYLTM)", -- alias: p_e_1fyyoyltm
                      "P/E (-0FQYoYLTM)", -- alias: p_e_0fqyoyltm
                      "Net EPS - Basic (LTM)", -- alias: net_eps_basic_ltm
                      "Net EPS - Basic (FQ)", -- alias: net_eps_basic_fq
                      "Net EPS - Basic (FY)", -- alias: net_eps_basic_fy
                      "Net EPS - Basic (-1FQFQ)", -- alias: net_eps_basic_1fqfq
                      "Net EPS - Basic (-2FQFQ)", -- alias: net_eps_basic_2fqfq
                      "Net EPS - Basic (-3FQFQ)", -- alias: net_eps_basic_3fqfq
                      "Net EPS - Basic (-4FQFQ)", -- alias: net_eps_basic_4fqfq
                      "Net EPS - Basic (-1FY)", -- alias: net_eps_basic_1fy
                      "Net EPS - Basic (-2FY)", -- alias: net_eps_basic_2fy
                      "Net EPS - Basic (-3FY)", -- alias: net_eps_basic_3fy
                      "Net EPS - Basic (-4FY)", -- alias: net_eps_basic_4fy
                      "Net EPS - Basic (-5FY)", -- alias: net_eps_basic_5fy
                      "EPS GAAP - Est Avg (NTM)", -- alias: eps_gaap_est_avg_ntm
                      "EPS GAAP - Est Avg (FY1E)", -- alias: eps_gaap_est_avg_fy1e
                      "Basic EPS - Cont (LTM)", -- alias: basic_eps_cont_ltm
                      "Basic EPS - Cont (FQ)", -- alias: basic_eps_cont_fq
                      "Basic EPS - Cont (FY)", -- alias: basic_eps_cont_fy
                      "Basic EPS - Cont (-1FQFQ)", -- alias: basic_eps_cont_1fqfq
                      "Basic EPS - Cont (-2FQFQ)", -- alias: basic_eps_cont_2fqfq
                      "Basic EPS - Cont (-3FQFQ)", -- alias: basic_eps_cont_3fqfq
                      "Basic EPS - Cont (-4FQFQ)", -- alias: basic_eps_cont_4fqfq
                      "Basic EPS - Cont (-1FY)", -- alias: basic_eps_cont_1fy
                      "Basic EPS - Cont (-2FY)", -- alias: basic_eps_cont_2fy
                      "Basic EPS - Cont (-3FY)", -- alias: basic_eps_cont_3fy
                      "Basic EPS - Cont (-4FY)", -- alias: basic_eps_cont_4fy
                      "EPS/Adj. (FQ)", -- alias: eps_adj_fq
                      "EPS/Adj. (-1FQFQ)", -- alias: eps_adj_1fqfq
                      "EPS/Adj. (-2FQFQ)", -- alias: eps_adj_2fqfq
                      "EPS/Adj. (-3FQFQ)", -- alias: eps_adj_3fqfq
                      "EPS/Adj. (-4FQFQ)", -- alias: eps_adj_4fqfq
                      "EPS/Adj. (-2FY)", -- alias: eps_adj_2fy
                      "EPS/Adj. (-3FY)", -- alias: eps_adj_3fy
                      "EPS/Adj. (-4FY)", -- alias: eps_adj_4fy
                      "Total Return (YTD)", -- alias: total_return_ytd
                      "Beta (1Y)", -- alias: beta_1y
                      "Beta (2Y)", -- alias: beta_2y
                      "Beta (5Y)", -- alias: beta_5y
                      "Total Revenues/CAGR (5Y FY)", -- alias: total_revenues_cagr_5y_fy
                      "Tot. Return %/CAGR (3Y)", -- alias: tot_return_pct_cagr_3y
                      "Tot. Return %/CAGR (10Y)", -- alias: tot_return_pct_cagr_10y
                      "Total Return (5Y)", -- alias: total_return_5y
                      "Total Return (10Y)", -- alias: total_return_10y
                      "Net Income Margin % (FY)", -- alias: net_income_margin_pct_fy
                      "Net Income Margin % (LTM)", -- alias: net_income_margin_pct_ltm
                      "Volatility (1M)", -- alias: volatility_1m
                      "Volatility (3M)", -- alias: volatility_3m
                      "Volatility (6M)", -- alias: volatility_6m
                      "Volatility (1Y)", -- alias: volatility_1y
                      "Div Yield (Ind)", -- alias: div_yield_ind
                      "Div Yield (LTM)", -- alias: div_yield_ltm
                      "Gross Profit Margin % (FY)", -- alias: gross_profit_margin_pct_fy
                      "Gross Profit Margin % (LTM)", -- alias: gross_profit_margin_pct_ltm
                      "Buyback Yield (LTM)", -- alias: buyback_yield_ltm
                      "Div Yield (-1FYInd)", -- alias: div_yield_1fyind
                      "Div Yield (TTM)", -- alias: div_yield_ttm
                      "Div Yield (NTM)", -- alias: div_yield_ntm
                      "Div Yield (5YAVGLTM)", -- alias: div_yield_5yavgltm
                      "Revenues - Est YoY % (FY1E)", -- alias: revenues_est_yoy_pct_fy1e
                      "Price Chg. % (1M)", -- alias: price_chg_pct_1m
                      "Price Chg. % (3M)", -- alias: price_chg_pct_3m
                      "1-Day %", -- alias: one_day_pct
                      "EPS Est Avg Rev % (FY1E - 1W)", -- alias: eps_est_avg_rev_pct_fy1e_1w
                      "EPS Est Avg Rev % (FY1E - 1M)", -- alias: eps_est_avg_rev_pct_fy1e_1m
                      "EPS Est Avg Rev % (FY1E - 3M)", -- alias: eps_est_avg_rev_pct_fy1e_3m
                      "EPS Est Avg Rev % (FY1E - 6M)", -- alias: eps_est_avg_rev_pct_fy1e_6m
                      "EPS Est Avg Rev % (FY1E - 1Y)", -- alias: eps_est_avg_rev_pct_fy1e_1y
                      "Div Yield (-2FYInd)", -- alias: div_yield_2fyind
                      "Div Yield (-3FYInd)", -- alias: div_yield_3fyind
                      "Div Yield (-4FYInd)", -- alias: div_yield_4fyind
                      "Div Yield (-5FYInd)", -- alias: div_yield_5fyind
                      "EPS GAAP Est Avg Rev % (FY1E - 1M)", -- alias: eps_gaap_est_avg_rev_pct_fy1e_1m
                      "EPS GAAP Est Avg Rev % (FY1E - 3M)", -- alias: eps_gaap_est_avg_rev_pct_fy1e_3m
                      "EPS GAAP Est Avg Rev % (FY1E - 6M)", -- alias: eps_gaap_est_avg_rev_pct_fy1e_6m
                      "EPS GAAP Est Avg Rev % (FY1E - 1Y)", -- alias: eps_gaap_est_avg_rev_pct_fy1e_1y
                      "Dividend Streak", -- alias: dividend_streak
                      "Price Target - #", -- alias: price_target_count
                      "Analyst Rating", -- alias: analyst_rating
                      "# Strong Sell Ratings", -- alias: num_strong_sell_ratings
                      "# Strong Buys Ratings", -- alias: num_strong_buys_ratings
                      "# Hold Ratings", -- alias: num_hold_ratings
                      "# Buys Ratings", -- alias: num_buys_ratings
                      "# Sell Ratings", -- alias: num_sell_ratings
                      "# No Opinion Ratings", -- alias: num_no_opinion_ratings
                      "Shrs Out", -- alias: shares_outstanding
                      "Shrs Out (-1FY)", -- alias: shrs_out_1fy
                      "Full Time Employees (FQ)", -- alias: full_time_employees_fq
                      "Full Time Employees (FY)", -- alias: full_time_employees_fy
                      "Full Time Employees (-1FY)", -- alias: full_time_employees_1fy
                      "Full Time Employees (-2FY)", -- alias: full_time_employees_2fy
                      "Full Time Employees (-3FY)", -- alias: full_time_employees_3fy
                      "Avg Employees (5YAVGFY)", -- alias: avg_employees_5yavgfy
                      "EPS Norm - Est # (FY1E)", -- alias: eps_norm_est_num_fy1e
                      "Price Target - # (3M Ago)", -- alias: price_target_num_3m_ago
                      "Price Target - # (6M Ago)", -- alias: price_target_num_6m_ago
                      "Price Target - # (YTD Ago)", -- alias: price_target_num_ytd_ago
                      "Price Target - # (1Y Ago)", -- alias: price_target_num_1y_ago
                      "Price Target - # (1W Ago)", -- alias: price_target_num_1w_ago
                      "Price Target - # (1M Ago)", -- alias: price_target_num_1m_ago
                      "Price Target - # (MTD Ago)", -- alias: price_target_num_mtd_ago
                      "Price Target - # (QTD Ago)", -- alias: price_target_num_qtd_ago
                      "Gain (Loss) On Sale Of Assets (LTM)", -- alias: gain_loss_on_sale_of_assets_ltm
                      "Impairment of Goodwill (FQ)", -- alias: impairment_of_goodwill_fq
                      "Impairment of Goodwill (LTM)", -- alias: impairment_of_goodwill_ltm
                      "Impairment of Goodwill (-1FY)", -- alias: impairment_of_goodwill_1fy
                      "Impairment of Goodwill (FY)", -- alias: impairment_of_goodwill_fy
                      "Asset Writedown (LTM)", -- alias: asset_writedown_ltm
                      "Asset Writedown (FY)", -- alias: asset_writedown_fy
                      "Asset Writedown (-1FY)", -- alias: asset_writedown_1fy
                      "Restructuring Charges (LTM)", -- alias: restructuring_charges_ltm
                      "Restructuring Charges (FQ)", -- alias: restructuring_charges_fq
                      "Restructuring Charges (-1FY)", -- alias: restructuring_charges_1fy
                      "Restructuring Charges (FY)", -- alias: restructuring_charges_fy
                      "Merger & Restructuring Charges (LTM)", -- alias: merger_and_restructuring_charges_ltm
                      "Other Unusual Items/Total (LTM)", -- alias: other_unusual_items_total_ltm
                      "Asset Writedown (FQ)", -- alias: asset_writedown_fq
                      "Asset Writedown (5YAVGFQ)", -- alias: asset_writedown_5yavgfq
                      "Impairment of Goodwill (5YAVGFQ)", -- alias: impairment_of_goodwill_5yavgfq
                      "Restructuring Charges (5YAVGFQ)", -- alias: restructuring_charges_5yavgfq
                      "Merger & Restructuring Charges (FQ)", -- alias: merger_and_restructuring_charges_fq
                      "Merger & Restructuring Charges (FY)", -- alias: merger_and_restructuring_charges_fy
                      "Merger & Restructuring Charges (5YAVGFQ)", -- alias: merger_and_restructuring_charges_5yavgfq
                      "Merger & Restructuring Charges (-1FQFQ)", -- alias: merger_and_restructuring_charges_1fqfq
                      "Merger & Restructuring Charges (-2FQFQ)", -- alias: merger_and_restructuring_charges_2fqfq
                      "Merger & Restructuring Charges (-3FQFQ)", -- alias: merger_and_restructuring_charges_3fqfq
                      "Merger & Restructuring Charges (-4FQFQ)", -- alias: merger_and_restructuring_charges_4fqfq
                      "Merger & Restructuring Charges (-1FY)", -- alias: merger_and_restructuring_charges_1fy
                      "Merger & Restructuring Charges (-2FY)", -- alias: merger_and_restructuring_charges_2fy
                      "Merger & Restructuring Charges (-3FY)", -- alias: merger_and_restructuring_charges_3fy
                      "Merger & Restructuring Charges (-4FY)", -- alias: merger_and_restructuring_charges_4fy
                      "Impairment of Goodwill (-1FQFQ)", -- alias: impairment_of_goodwill_1fqfq
                      "Impairment of Goodwill (-2FQFQ)", -- alias: impairment_of_goodwill_2fqfq
                      "Impairment of Goodwill (-3FQFQ)", -- alias: impairment_of_goodwill_3fqfq
                      "Impairment of Goodwill (-4FQFQ)", -- alias: impairment_of_goodwill_4fqfq
                      "Impairment of Goodwill (-2FY)", -- alias: impairment_of_goodwill_2fy
                      "Impairment of Goodwill (-3FY)", -- alias: impairment_of_goodwill_3fy
                      "Impairment of Goodwill (-4FY)", -- alias: impairment_of_goodwill_4fy
                      "Asset Writedown (-1FQFQ)", -- alias: asset_writedown_1fqfq
                      "Asset Writedown (-2FQFQ)", -- alias: asset_writedown_2fqfq
                      "Asset Writedown (-3FQFQ)", -- alias: asset_writedown_3fqfq
                      "Asset Writedown (-4FQFQ)", -- alias: asset_writedown_4fqfq
                      "Asset Writedown (-2FY)", -- alias: asset_writedown_2fy
                      "Asset Writedown (-3FY)", -- alias: asset_writedown_3fy
                      "Asset Writedown (-4FY)", -- alias: asset_writedown_4fy
                      "Asset Writedown (-5FY)", -- alias: asset_writedown_5fy
                      "Gain (Loss) On Sale Of Assets (FQ)", -- alias: gain_loss_on_sale_of_assets_fq
                      "Gain (Loss) On Sale Of Assets (FY)", -- alias: gain_loss_on_sale_of_assets_fy
                      "Gain (Loss) On Sale Of Assets (-1FQFQ)", -- alias: gain_loss_on_sale_of_assets_1fqfq
                      "Gain (Loss) On Sale Of Assets (-2FQFQ)", -- alias: gain_loss_on_sale_of_assets_2fqfq
                      "Gain (Loss) On Sale Of Assets (-3FQFQ)", -- alias: gain_loss_on_sale_of_assets_3fqfq
                      "Gain (Loss) On Sale Of Assets (-4FQFQ)", -- alias: gain_loss_on_sale_of_assets_4fqfq
                      "Gain (Loss) On Sale Of Assets (-1FY)", -- alias: gain_loss_on_sale_of_assets_1fy
                      "Gain (Loss) On Sale Of Assets (-2FY)", -- alias: gain_loss_on_sale_of_assets_2fy
                      "Gain (Loss) On Sale Of Assets (-3FY)", -- alias: gain_loss_on_sale_of_assets_3fy
                      "Gain (Loss) On Sale Of Assets (-4FY)", -- alias: gain_loss_on_sale_of_assets_4fy
                      "Restructuring Charges (-1FQFQ)", -- alias: restructuring_charges_1fqfq
                      "Restructuring Charges (-2FQFQ)", -- alias: restructuring_charges_2fqfq
                      "Restructuring Charges (-3FQFQ)", -- alias: restructuring_charges_3fqfq
                      "Restructuring Charges (-4FQFQ)", -- alias: restructuring_charges_4fqfq
                      "Restructuring Charges (-2FY)", -- alias: restructuring_charges_2fy
                      "Restructuring Charges (-3FY)", -- alias: restructuring_charges_3fy
                      "Restructuring Charges (-4FY)", -- alias: restructuring_charges_4fy
                      "Interest And Investment Income (LTM)", -- alias: interest_and_investment_income_ltm
                      "Interest And Investment Income (FQ)", -- alias: interest_and_investment_income_fq
                      "Interest And Investment Income (FY)", -- alias: interest_and_investment_income_fy
                      "Interest And Investment Income (-1FQFQ)", -- alias: interest_and_investment_income_1fqfq
                      "Interest And Investment Income (-2FQFQ)", -- alias: interest_and_investment_income_2fqfq
                      "Interest And Investment Income (-3FQFQ)", -- alias: interest_and_investment_income_3fqfq
                      "Interest And Investment Income (-4FQFQ)", -- alias: interest_and_investment_income_4fqfq
                      "Interest And Investment Income (-1FY)", -- alias: interest_and_investment_income_1fy
                      "Interest And Investment Income (-2FY)", -- alias: interest_and_investment_income_2fy
                      "Interest And Investment Income (-3FY)", -- alias: interest_and_investment_income_3fy
                      "Interest And Investment Income (-4FY)", -- alias: interest_and_investment_income_4fy
                      "Effective Tax Rate - (Ratio) (LTM)",
                      "Effective Tax Rate - (Ratio) (FQ)",
                      "Effective Tax Rate - (Ratio) (-1FQFQ)",
                      "Effective Tax Rate - (Ratio) (-2FQFQ)",
                      "Effective Tax Rate - (Ratio) (-4FQFQ)",
                      "Effective Tax Rate - (Ratio) (-3FQFQ)",
                      "Effective Tax Rate - (Ratio) (FY)",
                      "Effective Tax Rate - (Ratio) (-1FY)",
                      "Effective Tax Rate - (Ratio) (-2FY)",
                      "Effective Tax Rate - (Ratio) (-3FY)",
                      "Effective Tax Rate - (Ratio) (-4FY)",
                      "FCF - Est Avg (FY1E)", -- alias: fcf_est_avg_fy1e
                      "FCF - Est Avg (FY2E)", -- alias: fcf_est_avg_fy2e
                      "FCF - Est Avg (FY3E)", -- alias: fcf_est_avg_fy3e
                      "FCF - Est Avg (FY4E)", -- alias: fcf_est_avg_fy4e
                      "FCF - Est Avg (FY5E)", -- alias: fcf_est_avg_fy5e
                      "Total Operating Expenses (LTM)", -- alias: total_operating_expenses_ltm
                      "Total Operating Expenses (FQ)", -- alias: total_operating_expenses_fq
                      "Total Operating Expenses (FY)", -- alias: total_operating_expenses_fy
                      "Total Operating Expenses (-1FQFQ)", -- alias: total_operating_expenses_1fqfq
                      "Total Operating Expenses (-2FQFQ)", -- alias: total_operating_expenses_2fqfq
                      "Total Operating Expenses (-3FQFQ)", -- alias: total_operating_expenses_3fqfq
                      "Total Operating Expenses (-4FQFQ)", -- alias: total_operating_expenses_4fqfq
                      "Total Operating Expenses (-1FY)", -- alias: total_operating_expenses_1fy
                      "Total Operating Expenses (-2FY)", -- alias: total_operating_expenses_2fy
                      "Total Operating Expenses (-3FY)", -- alias: total_operating_expenses_3fy
                      "Total Operating Expenses (-4FY)", -- alias: total_operating_expenses_4fy
                      "Fiscal Month", -- alias: fiscal_month
                      "Fiscal Quarter", -- alias: fiscal_quarter
                      "Fiscal Year", -- alias: fiscal_year
                      "Reporting Lag") -- alias: reporting_lag
SELECT NULLIF(TRIM(s."Ticker"), '')                                              AS ticker,
       NULLIF(TRIM(s."ISIN"), '')                                                AS isin,
       NULLIF(TRIM(s."Name"), '')                                                AS name,
       NULLIF(TRIM(s."Description"), '')                                         AS description,
       COALESCE(NULLIF(TRIM(s."Region"), ''), 'n/a')                             AS region,
       COALESCE(NULLIF(TRIM(s."Country"), ''), 'n/a')                            AS country,
       COALESCE(NULLIF(TRIM(s."Trading Country"), ''), 'n/a')                    AS trading_country,
       COALESCE(NULLIF(TRIM(s."Exchange"), ''), 'n/a')                           AS exchange,
       COALESCE(NULLIF(TRIM(s."Unit"), ''), 'n/a')                               AS unit,
       COALESCE(NULLIF(TRIM(s."Sector"), ''), 'n/a')                             AS sector,
       COALESCE(NULLIF(TRIM(s."Industry"), ''), 'n/a')                           AS industry,
       COALESCE(NULLIF(TRIM(s."Style Class"), ''), 'n/a')                        AS style_class,
       COALESCE(NULLIF(TRIM(s."Size Class"), ''), 'n/a')                         AS size_class,
       COALESCE(NULLIF(TRIM(s."FY End"), ''), 'n/a')                             AS fy_end,
       COALESCE(NULLIF(TRIM(s."Next Earnings (When)"), ''), 'n/a')               AS next_earnings_when,
       COALESCE(NULLIF(TRIM(s."Next Earnings (Status)"), ''), 'n/a')             AS next_earnings_status,
       COALESCE(NULLIF(TRIM(s."Dividend Record (Currency)"), ''), 'n/a')         AS dividend_record_currency,
       COALESCE(NULLIF(TRIM(s."Dividend Record (Frequency)"), ''), 'n/a')        AS dividend_record_frequency,
       'Q' || current_fiscal.fiscal_quarter || ' ' || current_fiscal.fiscal_year AS current_fiscal_quarter,
       'Q' || calculate_next_fiscal_quarter(
               NULLIF(TRIM(s."Next Earnings"), '')::DATE,
               NULLIF(TRIM(s."Income Statement Report Date"), '')::DATE,
               parsed.fy_end_date,
               report_fiscal.earnings_report_frequency
              ) || ' ' || report_fiscal.next_quarter_year                        AS next_fiscal_quarter,
       report_fiscal.next_earnings_report_type                                   AS next_earnings_report,
       report_fiscal.reporting_interval                                          AS reporting_interval,
       report_fiscal.earnings_report_frequency                                   AS earnings_report_frequency,
       NULLIF(TRIM(s."Last Updated"), '')::DATE                                  AS last_updated,
       NULLIF(TRIM(s."Income Statement Report Date"), '')::DATE                  AS income_statement_report_date,
       NULLIF(TRIM(s."Next Earnings"), '')::DATE                                 AS next_earnings,
       NULLIF(TRIM(s."Dividend Record (Announce Date)"), '')::DATE               AS dividend_record_announce_date,
       NULLIF(TRIM(s."Dividend Record (Payable Date)"), '')::DATE                AS dividend_record_payable_date,
       NULLIF(TRIM(s."Dividend Record (Record Date)"), '')::DATE                 AS dividend_record_record_date,
       NULLIF(TRIM(s."Dividend Record (Ex Date)"), '')::DATE                     AS dividend_record_ex_date,
       CURRENT_DATE                                                              AS reference_date,
       parsed.fy_end_date                                                        AS fy_end_date,
       next_fy.next_fy_end_date                                                  AS next_fy_end_date,
       calculate_next_income_statement_report_date(
               NULLIF(TRIM(s."Income Statement Report Date"), '')::DATE,
               report_fiscal.earnings_report_frequency
       )                                                                         AS next_income_statement_report_date,
       text_to_numeric_safe(s."Price Target")                                    AS price_target,
       text_to_numeric_safe(s."Price Target - Median")                           AS price_target_median,
       COALESCE(text_to_numeric_safe(s."Dividend Record (Amount)"), 0)           AS dividend_record_amount,
       text_to_numeric_safe(s."Market Cap")                                      AS market_cap,
       text_to_numeric_safe(s."Enterprise Value")                                AS enterprise_value,
       text_to_numeric_safe(s."Last Price")                                      AS last_price,
       text_to_numeric_safe(s."Price Target (YTD Ago)")                          AS price_target_ytd_ago,
       text_to_numeric_safe(s."Price Target - Low")                              AS price_target_low,
       text_to_numeric_safe(s."Price Target - High")                             AS price_target_high,
       text_to_numeric_safe(s."Market Cap (Country R)")                          AS market_cap_country_r,
       text_to_numeric_safe(s."Volume (Shrs)")                                   AS volume_shrs,
       COALESCE(text_to_numeric_safe(s."Dividend Per Share (LTM)"), 0)           AS dividend_per_share_ltm,
       text_to_numeric_safe(s."Price (5D Ago)")                                  AS price_5d_ago,
       text_to_numeric_safe(s."Price (1W Ago)")                                  AS price_1w_ago,
       text_to_numeric_safe(s."Price (1M Ago)")                                  AS price_1m_ago,
       text_to_numeric_safe(s."Price (3M Ago)")                                  AS price_3m_ago,
       text_to_numeric_safe(s."Price (6M Ago)")                                  AS price_6m_ago,
       text_to_numeric_safe(s."Price (1Y Ago)")                                  AS price_1y_ago,
       text_to_numeric_safe(s."Price (3Y Ago)")                                  AS price_3y_ago,
       text_to_numeric_safe(s."Price (5Y Ago)")                                  AS price_5y_ago,
       text_to_numeric_safe(s."Price (QTD Ago)")                                 AS price_qtd_ago,
       text_to_numeric_safe(s."Rel. Volume")                                     AS rel_volume,
       text_to_numeric_safe(s."52W High/Adj")                                    AS w52_high_adj,
       text_to_numeric_safe(s."52W Low/Adj")                                     AS w52_low_adj,
       text_to_numeric_safe(s."EMA (20D)")                                       AS ema_20d,
       text_to_numeric_safe(s."EMA (50D)")                                       AS ema_50d,
       text_to_numeric_safe(s."EMA (100D)")                                      AS ema_100d,
       text_to_numeric_safe(s."EMA (250D)")                                      AS ema_250d,
       text_to_numeric_safe(s."Price Target (1W Ago)")                           AS price_target_1w_ago,
       text_to_numeric_safe(s."Price Target (1M Ago)")                           AS price_target_1m_ago,
       text_to_numeric_safe(s."Price Target (3M Ago)")                           AS price_target_3m_ago,
       text_to_numeric_safe(s."Price Target (6M Ago)")                           AS price_target_6m_ago,
       text_to_numeric_safe(s."Price Target (MTD Ago)")                          AS price_target_mtd_ago,
       text_to_numeric_safe(s."Price Target (QTD Ago)")                          AS price_target_qtd_ago,
       text_to_numeric_safe(s."Price Target (1Y Ago)")                           AS price_target_1y_ago,
       text_to_numeric_safe(s."Price Target - High (1W Ago)")                    AS price_target_high_1w_ago,
       text_to_numeric_safe(s."Price Target - High (1M Ago)")                    AS price_target_high_1m_ago,
       text_to_numeric_safe(s."Price Target - High (6M Ago)")                    AS price_target_high_6m_ago,
       text_to_numeric_safe(s."Price Target - High (MTD Ago)")                   AS price_target_high_mtd_ago,
       text_to_numeric_safe(s."Price Target - High (3M Ago)")                    AS price_target_high_3m_ago,
       text_to_numeric_safe(s."Price Target - High (QTD Ago)")                   AS price_target_high_qtd_ago,
       text_to_numeric_safe(s."Price Target - High (1Y Ago)")                    AS price_target_high_1y_ago,
       text_to_numeric_safe(s."Price Target - High (YTD Ago)")                   AS price_target_high_ytd_ago,
       text_to_numeric_safe(s."Price Target - Low (1W Ago)")                     AS price_target_low_1w_ago,
       text_to_numeric_safe(s."Price Target - Low (1M Ago)")                     AS price_target_low_1m_ago,
       text_to_numeric_safe(s."Price Target - Low (3M Ago)")                     AS price_target_low_3m_ago,
       text_to_numeric_safe(s."Price Target - Low (6M Ago)")                     AS price_target_low_6m_ago,
       text_to_numeric_safe(s."Price Target - Low (MTD Ago)")                    AS price_target_low_mtd_ago,
       text_to_numeric_safe(s."Price Target - Low (QTD Ago)")                    AS price_target_low_qtd_ago,
       text_to_numeric_safe(s."Price Target - Low (YTD Ago)")                    AS price_target_low_ytd_ago,
       text_to_numeric_safe(s."Price Target - Low (1Y Ago)")                     AS price_target_low_1y_ago,
       text_to_numeric_safe(s."Price Target - Median (1W Ago)")                  AS price_target_median_1w_ago,
       text_to_numeric_safe(s."Price Target - Median (1M Ago)")                  AS price_target_median_1m_ago,
       text_to_numeric_safe(s."Price Target - Median (3M Ago)")                  AS price_target_median_3m_ago,
       text_to_numeric_safe(s."Price Target - Median (6M Ago)")                  AS price_target_median_6m_ago,
       text_to_numeric_safe(s."Price Target - Median (MTD Ago)")                 AS price_target_median_mtd_ago,
       text_to_numeric_safe(s."Price Target - Median (QTD Ago)")                 AS price_target_median_qtd_ago,
       text_to_numeric_safe(s."Price Target - Median (YTD Ago)")                 AS price_target_median_ytd_ago,
       text_to_numeric_safe(s."Price Target - Median (1Y Ago)")                  AS price_target_median_1y_ago,
       COALESCE(text_to_numeric_safe(s."Total Revenues (FQ)"), 0)                AS total_revenues_fq,
       COALESCE(text_to_numeric_safe(s."Total Revenues (-1FY)"), 0)              AS total_revenues_1fy,
       COALESCE(text_to_numeric_safe(s."Total Revenues (FY)"), 0)                AS total_revenues_fy,
       COALESCE(text_to_numeric_safe(s."Total Revenues (LTM)"), 0)               AS total_revenues_ltm,
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-1FY)"), 0)             AS net_income_adj_1fy,
       COALESCE(text_to_numeric_safe(s."EBITDA (FQ)"), 0)                        AS ebitda_fq,
       COALESCE(text_to_numeric_safe(s."EBITDA (LTM)"), 0)                       AS ebitda_ltm,
       COALESCE(text_to_numeric_safe(s."EBITDA (FY)"), 0)                        AS ebitda_fy,
       COALESCE(text_to_numeric_safe(s."EBITDA (-1FY)"), 0)                      AS ebitda_1fy,
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (LTM)"), 0)                  AS ebitda_adj_ltm,
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (FY)"), 0)                   AS ebitda_adj_fy,
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-1FY)"), 0)                 AS ebitda_adj_1fy,
       COALESCE(text_to_numeric_safe(s."EBIT (FQ)"), 0)                          AS ebit_fq,
       COALESCE(text_to_numeric_safe(s."EBIT (LTM)"), 0)                         AS ebit_ltm,
       COALESCE(text_to_numeric_safe(s."EBIT (FY)"), 0)                          AS ebit_fy,
       COALESCE(text_to_numeric_safe(s."EBIT (-1FY)"), 0)                        AS ebit_1fy,
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-1FY)"), 0)                   AS ebit_adj_1fy,
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (FY)"), 0)                     AS ebit_adj_fy,
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (LTM)"), 0)                    AS ebit_adj_ltm,
       COALESCE(text_to_numeric_safe(s."EBIT - Est Med (FY1E)"), 0)              AS ebit_est_med_fy1e,
       COALESCE(text_to_numeric_safe(s."EBIT - Est Med (NTM)"), 0)               AS ebit_est_med_ntm,
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (FY)"), 0)             AS net_income_is_fy,
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (LTM)"), 0)            AS net_income_is_ltm,
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (FY)"), 0)         AS normalized_net_income_fy,
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (LTM)"), 0)        AS normalized_net_income_ltm,
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (FY)"), 0)               AS net_income_adj_fy,
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (LTM)"), 0)              AS net_income_adj_ltm,
       COALESCE(text_to_numeric_safe(s."Gross Profit (LTM)"), 0)                 AS gross_profit_ltm,
       COALESCE(text_to_numeric_safe(s."Gross Profit (FY)"), 0)                  AS gross_profit_fy,
       COALESCE(text_to_numeric_safe(s."Cost Of Revenues (LTM)"), 0)             AS cost_of_revenues_ltm,
       COALESCE(text_to_numeric_safe(s."Operating Income (LTM)"), 0)             AS operating_income_ltm,
       COALESCE(text_to_numeric_safe(s."Operating Income (FY)"), 0)              AS operating_income_fy,
       COALESCE(text_to_numeric_safe(s."R&D Expenses (LTM)"), 0)                 AS randd_expenses_ltm,
       COALESCE(text_to_numeric_safe(s."Interest Expense/Total (LTM)"), 0)       AS interest_expense_total_ltm,
       COALESCE(text_to_numeric_safe(s."Interest Income On Investments (LTM)"),
                0)                                                               AS interest_income_on_investments_ltm,
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-1FY)"), 0)           AS net_income_is_1fy,
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-1FY)"), 0)       AS normalized_net_income_1fy,
       COALESCE(text_to_numeric_safe(s."Total Revenues (5YAVGFQ)"), 0)           AS total_revenues_5yavgfq,
       COALESCE(text_to_numeric_safe(s."EBITDA (5YAVGFQ)"), 0)                   AS ebitda_5yavgfq,
       COALESCE(text_to_numeric_safe(s."EBIT (5YAVGFQ)"), 0)                     AS ebit_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Operating Income (FQ)"), 0)              AS operating_income_fq,
       COALESCE(text_to_numeric_safe(s."Operating Income (5YAVGFQ)"), 0)         AS operating_income_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (FQ)"), 0)         AS normalized_net_income_fq,
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (5YAVGFQ)"),
                0)                                                               AS normalized_net_income_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (FQ)"), 0)               AS net_income_adj_fq,
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (5YAVGFQ)"), 0)          AS net_income_adj_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (FQ)"), 0)             AS net_income_is_fq,
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (5YAVGFQ)"), 0)        AS net_income_is_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (5YAVGLTM)"), 0)       AS net_income_is_5yavgltm,
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (5YAVGLTM)"),
                0)                                                               AS normalized_net_income_5yavgltm,
       COALESCE(text_to_numeric_safe(s."EBITDA (5YAVGLTM)"), 0)                  AS ebitda_5yavgltm,
       COALESCE(text_to_numeric_safe(s."EBIT (5YAVGLTM)"), 0)                    AS ebit_5yavgltm,
       COALESCE(text_to_numeric_safe(s."Total Revenues (5YAVGLTM)"), 0)          AS total_revenues_5yavgltm,
       COALESCE(text_to_numeric_safe(s."Selling General & Admin Expenses/Total (FQ)"),
                0)                                                               AS selling_general_and_admin_expenses_total_fq,
       COALESCE(text_to_numeric_safe(s."Selling General & Admin Expenses/Total (FY)"),
                0)                                                               AS selling_general_and_admin_expenses_total_fy,
       COALESCE(text_to_numeric_safe(s."Selling General & Admin Expenses/Total (-1FY)"),
                0)                                                               AS selling_general_and_admin_expenses_total_1fy,
       COALESCE(text_to_numeric_safe(s."Selling General & Admin Expenses/Total (5YAVGFQ)"),
                0)                                                               AS selling_general_and_admin_expenses_total_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Marketing Expenses (FQ)"), 0)            AS marketing_expenses_fq,
       COALESCE(text_to_numeric_safe(s."Marketing Expenses (FY)"), 0)            AS marketing_expenses_fy,
       COALESCE(text_to_numeric_safe(s."Marketing Expenses (-1FY)"), 0)          AS marketing_expenses_1fy,
       COALESCE(text_to_numeric_safe(s."Marketing Expenses (5YAVGLTM)"),
                0)                                                               AS marketing_expenses_5yavgltm,
       COALESCE(text_to_numeric_safe(s."Revenues - Est Avg (NTM)"), 0)           AS revenues_est_avg_ntm,
       COALESCE(text_to_numeric_safe(s."Revenues - Est Avg (FY1E)"), 0)          AS revenues_est_avg_fy1e,
       COALESCE(text_to_numeric_safe(s."Revenues - Est Med (NTM)"), 0)           AS revenues_est_med_ntm,
       COALESCE(text_to_numeric_safe(s."Revenues - Est Med (FY1E)"), 0)          AS revenues_est_med_fy1e,
       COALESCE(text_to_numeric_safe(s."EBITDA - Est Avg (NTM)"), 0)             AS ebitda_est_avg_ntm,
       COALESCE(text_to_numeric_safe(s."EBITDA - Est Avg (FY1E)"), 0)            AS ebitda_est_avg_fy1e,
       -- NEW: Total Revenues Historical
       text_to_numeric_safe(s."Total Revenues (-1FQFQ)")                         AS total_revenues_1fqfq,
       text_to_numeric_safe(s."Total Revenues (-2FQFQ)")                         AS total_revenues_2fqfq,
       text_to_numeric_safe(s."Total Revenues (-3FQFQ)")                         AS total_revenues_3fqfq,
       text_to_numeric_safe(s."Total Revenues (-4FQFQ)")                         AS total_revenues_4fqfq,
       text_to_numeric_safe(s."Total Revenues (-2FY)")                           AS total_revenues_2fy,
       text_to_numeric_safe(s."Total Revenues (-3FY)")                           AS total_revenues_3fy,
       text_to_numeric_safe(s."Total Revenues (-4FY)")                           AS total_revenues_4fy,
       -- NEW: Gross Profit Historical
       COALESCE(text_to_numeric_safe(s."Gross Profit (FQ)"), 0)                  AS gross_profit_fq,
       COALESCE(text_to_numeric_safe(s."Gross Profit (-1FQFQ)"), 0)              AS gross_profit_1fqfq,
       COALESCE(text_to_numeric_safe(s."Gross Profit (-2FQFQ)"), 0)              AS gross_profit_2fqfq,
       COALESCE(text_to_numeric_safe(s."Gross Profit (-3FQFQ)"), 0)              AS gross_profit_3fqfq,
       COALESCE(text_to_numeric_safe(s."Gross Profit (-4FQFQ)"), 0)              AS gross_profit_4fqfq,
       COALESCE(text_to_numeric_safe(s."Gross Profit (-1FY)"), 0)                AS gross_profit_1fy,
       COALESCE(text_to_numeric_safe(s."Gross Profit (-2FY)"), 0)                AS gross_profit_2fy,
       COALESCE(text_to_numeric_safe(s."Gross Profit (-3FY)"), 0)                AS gross_profit_3fy,
       COALESCE(text_to_numeric_safe(s."Gross Profit (-4FY)"), 0)                AS gross_profit_4fy,
       -- NEW: Operating Income Historical
       COALESCE(text_to_numeric_safe(s."Operating Income (-1FQFQ)"), 0)          AS operating_income_1fqfq,
       COALESCE(text_to_numeric_safe(s."Operating Income (-2FQFQ)"), 0)          AS operating_income_2fqfq,
       COALESCE(text_to_numeric_safe(s."Operating Income (-3FQFQ)"), 0)          AS operating_income_3fqfq,
       COALESCE(text_to_numeric_safe(s."Operating Income (-4FQFQ)"), 0)          AS operating_income_4fqfq,
       COALESCE(text_to_numeric_safe(s."Operating Income (-1FY)"), 0)            AS operating_income_1fy,
       COALESCE(text_to_numeric_safe(s."Operating Income (-2FY)"), 0)            AS operating_income_2fy,
       COALESCE(text_to_numeric_safe(s."Operating Income (-3FY)"), 0)            AS operating_income_3fy,
       COALESCE(text_to_numeric_safe(s."Operating Income (-4FY)"), 0)            AS operating_income_4fy,
       -- NEW: R&D Expenses Historical
       COALESCE(text_to_numeric_safe(s."R&D Expenses (FQ)"), 0)                  AS randd_expenses_fq,
       COALESCE(text_to_numeric_safe(s."R&D Expenses (FY)"), 0)                  AS randd_expenses_fy,
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-1FQFQ)"), 0)              AS randd_expenses_1fqfq,
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-2FQFQ)"), 0)              AS randd_expenses_2fqfq,
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-3FQFQ)"), 0)              AS randd_expenses_3fqfq,
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-4FQFQ)"), 0)              AS randd_expenses_4fqfq,
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-1FY)"), 0)                AS randd_expenses_1fy,
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-2FY)"), 0)                AS randd_expenses_2fy,
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-3FY)"), 0)                AS randd_expenses_3fy,
       COALESCE(text_to_numeric_safe(s."R&D Expenses (-4FY)"), 0)                AS randd_expenses_4fy,
       -- NEW: Net Income - (IS) Historical
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-1FQFQ)"), 0)         AS net_income_is_1fqfq,
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-2FQFQ)"), 0)         AS net_income_is_2fqfq,
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-3FQFQ)"), 0)         AS net_income_is_3fqfq,
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-4FQFQ)"), 0)         AS net_income_is_4fqfq,
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-2FY)"), 0)           AS net_income_is_2fy,
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-3FY)"), 0)           AS net_income_is_3fy,
       COALESCE(text_to_numeric_safe(s."Net Income - (IS) (-4FY)"), 0)           AS net_income_is_4fy,
       -- NEW: Normalized Net Income Historical
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-1FQFQ)"),
                0)                                                               AS normalized_net_income_1fqfq,
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-2FQFQ)"),
                0)                                                               AS normalized_net_income_2fqfq,
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-3FQFQ)"),
                0)                                                               AS normalized_net_income_3fqfq,
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-4FQFQ)"),
                0)                                                               AS normalized_net_income_4fqfq,
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-2FY)"), 0)       AS normalized_net_income_2fy,
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-3FY)"), 0)       AS normalized_net_income_3fy,
       COALESCE(text_to_numeric_safe(s."Normalized Net Income (-4FY)"), 0)       AS normalized_net_income_4fy,
       -- NEW: Net Income/Adj. Historical
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-1FQFQ)"), 0)           AS net_income_adj_1fqfq,
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-2FQFQ)"), 0)           AS net_income_adj_2fqfq,
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-3FQFQ)"), 0)           AS net_income_adj_3fqfq,
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-4FQFQ)"), 0)           AS net_income_adj_4fqfq,
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-2FY)"), 0)             AS net_income_adj_2fy,
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-3FY)"), 0)             AS net_income_adj_3fy,
       COALESCE(text_to_numeric_safe(s."Net Income/Adj. (-4FY)"), 0)             AS net_income_adj_4fy,
       -- NEW: EBIT Historical
       COALESCE(text_to_numeric_safe(s."EBIT (-1FQFQ)"), 0)                      AS ebit_1fqfq,
       COALESCE(text_to_numeric_safe(s."EBIT (-2FQFQ)"), 0)                      AS ebit_2fqfq,
       COALESCE(text_to_numeric_safe(s."EBIT (-3FQFQ)"), 0)                      AS ebit_3fqfq,
       COALESCE(text_to_numeric_safe(s."EBIT (-4FQFQ)"), 0)                      AS ebit_4fqfq,
       COALESCE(text_to_numeric_safe(s."EBIT (-2FY)"), 0)                        AS ebit_2fy,
       COALESCE(text_to_numeric_safe(s."EBIT (-3FY)"), 0)                        AS ebit_3fy,
       COALESCE(text_to_numeric_safe(s."EBIT (-4FY)"), 0)                        AS ebit_4fy,
       -- NEW: EBIT/Adj. Historical
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (FQ)"), 0)                     AS ebit_adj_fq,
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-1FQFQ)"), 0)                 AS ebit_adj_1fqfq,
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-2FQFQ)"), 0)                 AS ebit_adj_2fqfq,
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-3FQFQ)"), 0)                 AS ebit_adj_3fqfq,
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-4FQFQ)"), 0)                 AS ebit_adj_4fqfq,
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-2FY)"), 0)                   AS ebit_adj_2fy,
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-3FY)"), 0)                   AS ebit_adj_3fy,
       COALESCE(text_to_numeric_safe(s."EBIT/Adj. (-4FY)"), 0)                   AS ebit_adj_4fy,
       -- NEW: EBITDA Historical
       COALESCE(text_to_numeric_safe(s."EBITDA (-1FQFQ)"), 0)                    AS ebitda_1fqfq,
       COALESCE(text_to_numeric_safe(s."EBITDA (-2FQFQ)"), 0)                    AS ebitda_2fqfq,
       COALESCE(text_to_numeric_safe(s."EBITDA (-3FQFQ)"), 0)                    AS ebitda_3fqfq,
       COALESCE(text_to_numeric_safe(s."EBITDA (-4FQFQ)"), 0)                    AS ebitda_4fqfq,
       COALESCE(text_to_numeric_safe(s."EBITDA (-2FY)"), 0)                      AS ebitda_2fy,
       COALESCE(text_to_numeric_safe(s."EBITDA (-3FY)"), 0)                      AS ebitda_3fy,
       COALESCE(text_to_numeric_safe(s."EBITDA (-4FY)"), 0)                      AS ebitda_4fy,
       -- NEW: EBITDA/Adj. Historical
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (FQ)"), 0)                   AS ebitda_adj_fq,
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-1FQFQ)"), 0)               AS ebitda_adj_1fqfq,
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-2FQFQ)"), 0)               AS ebitda_adj_2fqfq,
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-3FQFQ)"), 0)               AS ebitda_adj_3fqfq,
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-4FQFQ)"), 0)               AS ebitda_adj_4fqfq,
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-2FY)"), 0)                 AS ebitda_adj_2fy,
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-3FY)"), 0)                 AS ebitda_adj_3fy,
       COALESCE(text_to_numeric_safe(s."EBITDA/Adj. (-4FY)"), 0)                 AS ebitda_adj_4fy,
       COALESCE(text_to_numeric_safe(s."TBV (FY)"), 0)                           AS tbv_fy,
       COALESCE(text_to_numeric_safe(s."TBV (LTM)"), 0)                          AS tbv_ltm,
       COALESCE(text_to_numeric_safe(s."Total Debt (FY)"), 0)                    AS total_debt_fy,
       COALESCE(text_to_numeric_safe(s."Total Equity (FY)"), 0)                  AS total_equity_fy,
       COALESCE(text_to_numeric_safe(s."Total Equity (LTM)"), 0)                 AS total_equity_ltm,
       COALESCE(text_to_numeric_safe(s."Total Debt (LTM)"), 0)                   AS total_debt_ltm,
       COALESCE(text_to_numeric_safe(s."Total Assets (LTM)"), 0)                 AS total_assets_ltm,
       COALESCE(text_to_numeric_safe(s."Total Assets (FY)"), 0)                  AS total_assets_fy,
       COALESCE(text_to_numeric_safe(s."Inventory (LTM)"), 0)                    AS inventory_ltm,
       COALESCE(text_to_numeric_safe(s."Goodwill (FQ)"), 0)                      AS goodwill_fq,
       COALESCE(text_to_numeric_safe(s."Goodwill (LTM)"), 0)                     AS goodwill_ltm,
       COALESCE(text_to_numeric_safe(s."Goodwill (FY)"), 0)                      AS goodwill_fy,
       COALESCE(text_to_numeric_safe(s."Goodwill (-1FY)"), 0)                    AS goodwill_1fy,
       COALESCE(text_to_numeric_safe(s."Retained Earnings (LTM)"), 0)            AS retained_earnings_ltm,
       COALESCE(text_to_numeric_safe(s."Total Current Assets (LTM)"), 0)         AS total_current_assets_ltm,
       COALESCE(text_to_numeric_safe(s."Total Current Liabilities (LTM)"),
                0)                                                               AS total_current_liabilities_ltm,
       COALESCE(text_to_numeric_safe(s."Working Capital (LTM)"), 0)              AS working_capital_ltm,
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (LTM)"), 0)         AS cash_and_equivalents_ltm,
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (FQ)"), 0)          AS cash_and_equivalents_fq,
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (FY)"), 0)          AS cash_and_equivalents_fy,
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (5YAVGFQ)"),
                0)                                                               AS cash_and_equivalents_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Inventory (FQ)"), 0)                     AS inventory_fq,
       COALESCE(text_to_numeric_safe(s."Inventory (FY)"), 0)                     AS inventory_fy,
       COALESCE(text_to_numeric_safe(s."Goodwill (5YAVGFQ)"), 0)                 AS goodwill_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Inventory (5YAVGFQ)"), 0)                AS inventory_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Retained Earnings (FQ)"), 0)             AS retained_earnings_fq,
       COALESCE(text_to_numeric_safe(s."Retained Earnings (FY)"), 0)             AS retained_earnings_fy,
       COALESCE(text_to_numeric_safe(s."Retained Earnings (5YAVGFQ)"), 0)        AS retained_earnings_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Working Capital (FQ)"), 0)               AS working_capital_fq,
       COALESCE(text_to_numeric_safe(s."Working Capital (FY)"), 0)               AS working_capital_fy,
       COALESCE(text_to_numeric_safe(s."Working Capital (5YAVGFY)"), 0)          AS working_capital_5yavgfy,
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (LTM)"),
                0)                                                               AS gross_intangible_assets_ltm,
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (FY)"), 0)       AS gross_intangible_assets_fy,
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (5YAVGFQ)"),
                0)                                                               AS gross_intangible_assets_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Accounts Receivable/Total (FY)"),
                0)                                                               AS accounts_receivable_total_fy,
       COALESCE(text_to_numeric_safe(s."Accounts Receivable/Total (-1FY)"),
                0)                                                               AS accounts_receivable_total_1fy,
       COALESCE(text_to_numeric_safe(s."Accounts Receivable/Total (5YAVGFQ)"),
                0)                                                               AS accounts_receivable_total_5yavgfq,
       -- NEW: Working Capital Historical
       COALESCE(text_to_numeric_safe(s."Working Capital (-1FQ)"), 0)             AS working_capital_1fq,
       COALESCE(text_to_numeric_safe(s."Working Capital (-2FQ)"), 0)             AS working_capital_2fq,
       COALESCE(text_to_numeric_safe(s."Working Capital (-3FQ)"), 0)             AS working_capital_3fq,
       COALESCE(text_to_numeric_safe(s."Working Capital (-4FQ)"), 0)             AS working_capital_4fq,
       COALESCE(text_to_numeric_safe(s."Working Capital (-1FY)"), 0)             AS working_capital_1fy,
       COALESCE(text_to_numeric_safe(s."Working Capital (-2FY)"), 0)             AS working_capital_2fy,
       COALESCE(text_to_numeric_safe(s."Working Capital (-3FY)"), 0)             AS working_capital_3fy,
       COALESCE(text_to_numeric_safe(s."Working Capital (-4FY)"), 0)             AS working_capital_4fy,
       -- NEW: Total Debt Historical
       COALESCE(text_to_numeric_safe(s."Total Debt (FQ)"), 0)                    AS total_debt_fq,
       COALESCE(text_to_numeric_safe(s."Total Debt (-1FQ)"), 0)                  AS total_debt_1fq,
       COALESCE(text_to_numeric_safe(s."Total Debt (-2FQ)"), 0)                  AS total_debt_2fq,
       COALESCE(text_to_numeric_safe(s."Total Debt (-3FQ)"), 0)                  AS total_debt_3fq,
       COALESCE(text_to_numeric_safe(s."Total Debt (-4FQ)"), 0)                  AS total_debt_4fq,
       COALESCE(text_to_numeric_safe(s."Total Debt (-1FY)"), 0)                  AS total_debt_1fy,
       COALESCE(text_to_numeric_safe(s."Total Debt (-2FY)"), 0)                  AS total_debt_2fy,
       COALESCE(text_to_numeric_safe(s."Total Debt (-3FY)"), 0)                  AS total_debt_3fy,
       COALESCE(text_to_numeric_safe(s."Total Debt (-4FY)"), 0)                  AS total_debt_4fy,
       -- NEW: Total Assets Historical
       COALESCE(text_to_numeric_safe(s."Total Assets (FQ)"), 0)                  AS total_assets_fq,
       COALESCE(text_to_numeric_safe(s."Total Assets (-1FQ)"), 0)                AS total_assets_1fq,
       COALESCE(text_to_numeric_safe(s."Total Assets (-2FQ)"), 0)                AS total_assets_2fq,
       COALESCE(text_to_numeric_safe(s."Total Assets (-3FQ)"), 0)                AS total_assets_3fq,
       COALESCE(text_to_numeric_safe(s."Total Assets (-4FQ)"), 0)                AS total_assets_4fq,
       COALESCE(text_to_numeric_safe(s."Total Assets (-1FY)"), 0)                AS total_assets_1fy,
       COALESCE(text_to_numeric_safe(s."Total Assets (-2FY)"), 0)                AS total_assets_2fy,
       COALESCE(text_to_numeric_safe(s."Total Assets (-3FY)"), 0)                AS total_assets_3fy,
       COALESCE(text_to_numeric_safe(s."Total Assets (-4FY)"), 0)                AS total_assets_4fy,
       -- NEW: Inventory Historical
       COALESCE(text_to_numeric_safe(s."Inventory (-1FQ)"), 0)                   AS inventory_1fq,
       COALESCE(text_to_numeric_safe(s."Inventory (-2FQ)"), 0)                   AS inventory_2fq,
       COALESCE(text_to_numeric_safe(s."Inventory (-3FQ)"), 0)                   AS inventory_3fq,
       COALESCE(text_to_numeric_safe(s."Inventory (-4FQ)"), 0)                   AS inventory_4fq,
       COALESCE(text_to_numeric_safe(s."Inventory (-1FY)"), 0)                   AS inventory_1fy,
       COALESCE(text_to_numeric_safe(s."Inventory (-2FY)"), 0)                   AS inventory_2fy,
       COALESCE(text_to_numeric_safe(s."Inventory (-3FY)"), 0)                   AS inventory_3fy,
       COALESCE(text_to_numeric_safe(s."Inventory (-4FY)"), 0)                   AS inventory_4fy,
       -- NEW: Goodwill Historical
       COALESCE(text_to_numeric_safe(s."Goodwill (-1FQ)"), 0)                    AS goodwill_1fq,
       COALESCE(text_to_numeric_safe(s."Goodwill (-2FQ)"), 0)                    AS goodwill_2fq,
       COALESCE(text_to_numeric_safe(s."Goodwill (-3FQ)"), 0)                    AS goodwill_3fq,
       COALESCE(text_to_numeric_safe(s."Goodwill (-4FQ)"), 0)                    AS goodwill_4fq,
       COALESCE(text_to_numeric_safe(s."Goodwill (-2FY)"), 0)                    AS goodwill_2fy,
       COALESCE(text_to_numeric_safe(s."Goodwill (-3FY)"), 0)                    AS goodwill_3fy,
       COALESCE(text_to_numeric_safe(s."Goodwill (-4FY)"), 0)                    AS goodwill_4fy,
       -- NEW: Retained Earnings Historical
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-1FQ)"), 0)           AS retained_earnings_1fq,
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-2FQ)"), 0)           AS retained_earnings_2fq,
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-3FQ)"), 0)           AS retained_earnings_3fq,
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-4FQ)"), 0)           AS retained_earnings_4fq,
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-1FY)"), 0)           AS retained_earnings_1fy,
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-2FY)"), 0)           AS retained_earnings_2fy,
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-3FY)"), 0)           AS retained_earnings_3fy,
       COALESCE(text_to_numeric_safe(s."Retained Earnings (-4FY)"), 0)           AS retained_earnings_4fy,
       -- NEW: Cash And Equivalents Historical
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-1FQ)"), 0)        AS cash_and_equivalents_1fq,
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-2FQ)"), 0)        AS cash_and_equivalents_2fq,
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-3FQ)"), 0)        AS cash_and_equivalents_3fq,
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-4FQ)"), 0)        AS cash_and_equivalents_4fq,
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-1FY)"), 0)        AS cash_and_equivalents_1fy,
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-2FY)"), 0)        AS cash_and_equivalents_2fy,
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-3FY)"), 0)        AS cash_and_equivalents_3fy,
       COALESCE(text_to_numeric_safe(s."Cash And Equivalents (-4FY)"), 0)        AS cash_and_equivalents_4fy,
       -- NEW: Gross Intangible Assets Historical
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (FQ)"), 0)       AS gross_intangible_assets_fq,
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-1FQ)"),
                0)                                                               AS gross_intangible_assets_1fq,
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-2FQ)"),
                0)                                                               AS gross_intangible_assets_2fq,
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-3FQ)"),
                0)                                                               AS gross_intangible_assets_3fq,
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-4FQ)"),
                0)                                                               AS gross_intangible_assets_4fq,
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-1FY)"),
                0)                                                               AS gross_intangible_assets_1fy,
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-2FY)"),
                0)                                                               AS gross_intangible_assets_2fy,
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-3FY)"),
                0)                                                               AS gross_intangible_assets_3fy,
       COALESCE(text_to_numeric_safe(s."Gross Intangible Assets (-4FY)"),
                0)                                                               AS gross_intangible_assets_4fy,
       COALESCE(text_to_numeric_safe(s."CFF (LTM)"), 0)                          AS cff_ltm,
       COALESCE(text_to_numeric_safe(s."CFI (LTM)"), 0)                          AS cfi_ltm,
       COALESCE(text_to_numeric_safe(s."FCF (LTM)"), 0)                          AS fcf_ltm,
       COALESCE(text_to_numeric_safe(s."CFO (LTM)"), 0)                          AS cfo_ltm,
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (LTM)"), 0)            AS cash_acquisitions_ltm,
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (FY)"), 0)             AS cash_acquisitions_fy,
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-1FY)"), 0)           AS cash_acquisitions_1fy,
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (LTM)"), 0)          AS capital_expenditure_ltm,
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-1FY)"), 0)         AS capital_expenditure_1fy,
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (FY)"), 0)           AS capital_expenditure_fy,
       COALESCE(text_to_numeric_safe(s."CFF (FY)"), 0)                           AS cff_fy,
       COALESCE(text_to_numeric_safe(s."CFF (-1FY)"), 0)                         AS cff_1fy,
       COALESCE(text_to_numeric_safe(s."CFI (FY)"), 0)                           AS cfi_fy,
       COALESCE(text_to_numeric_safe(s."CFI (-1FY)"), 0)                         AS cfi_1fy,
       COALESCE(text_to_numeric_safe(s."CFO (FY)"), 0)                           AS cfo_fy,
       COALESCE(text_to_numeric_safe(s."CFO (-1FY)"), 0)                         AS cfo_1fy,
       COALESCE(text_to_numeric_safe(s."FCF (FY)"), 0)                           AS fcf_fy,
       COALESCE(text_to_numeric_safe(s."FCF (-1FY)"), 0)                         AS fcf_1fy,
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (FQ)"), 0)           AS capital_expenditure_fq,
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (5YAVGFQ)"),
                0)                                                               AS capital_expenditure_5yavgfq,
       COALESCE(text_to_numeric_safe(s."CFF (FQ)"), 0)                           AS cff_fq,
       COALESCE(text_to_numeric_safe(s."CFI (FQ)"), 0)                           AS cfi_fq,
       COALESCE(text_to_numeric_safe(s."CFO (FQ)"), 0)                           AS cfo_fq,
       COALESCE(text_to_numeric_safe(s."FCF (FQ)"), 0)                           AS fcf_fq,
       COALESCE(text_to_numeric_safe(s."FCF (5YAVGFQ)"), 0)                      AS fcf_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (FQ)"), 0)             AS cash_acquisitions_fq,
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (5YAVGFQ)"), 0)        AS cash_acquisitions_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Common Dividends Paid (LTM)"), 0)        AS common_dividends_paid_ltm,
       COALESCE(text_to_numeric_safe(s."Common Dividends Paid (FY)"), 0)         AS common_dividends_paid_fy,
       COALESCE(text_to_numeric_safe(s."CFO (-1FQFQ)"), 0)                       AS cfo_1fqfq,
       COALESCE(text_to_numeric_safe(s."CFO (-2FQFQ)"), 0)                       AS cfo_2fqfq,
       COALESCE(text_to_numeric_safe(s."CFO (-3FQFQ)"), 0)                       AS cfo_3fqfq,
       COALESCE(text_to_numeric_safe(s."CFO (-4FQFQ)"), 0)                       AS cfo_4fqfq,
       COALESCE(text_to_numeric_safe(s."CFI (-1FQFQ)"), 0)                       AS cfi_1fqfq,
       COALESCE(text_to_numeric_safe(s."CFI (-2FQFQ)"), 0)                       AS cfi_2fqfq,
       COALESCE(text_to_numeric_safe(s."CFI (-3FQFQ)"), 0)                       AS cfi_3fqfq,
       COALESCE(text_to_numeric_safe(s."CFI (-4FQFQ)"), 0)                       AS cfi_4fqfq,
       COALESCE(text_to_numeric_safe(s."CFI (-2FY)"), 0)                         AS cfi_2fy,
       COALESCE(text_to_numeric_safe(s."CFI (-3FY)"), 0)                         AS cfi_3fy,
       COALESCE(text_to_numeric_safe(s."CFI (-4FY)"), 0)                         AS cfi_4fy,
       COALESCE(text_to_numeric_safe(s."FCF (-1FQFQ)"), 0)                       AS fcf_1fqfq,
       COALESCE(text_to_numeric_safe(s."FCF (-2FQFQ)"), 0)                       AS fcf_2fqfq,
       COALESCE(text_to_numeric_safe(s."FCF (-3FQFQ)"), 0)                       AS fcf_3fqfq,
       COALESCE(text_to_numeric_safe(s."FCF (-4FQFQ)"), 0)                       AS fcf_4fqfq,
       COALESCE(text_to_numeric_safe(s."CFF (-2FY)"), 0)                         AS cff_2fy,
       COALESCE(text_to_numeric_safe(s."CFF (-3FY)"), 0)                         AS cff_3fy,
       COALESCE(text_to_numeric_safe(s."CFF (-4FY)"), 0)                         AS cff_4fy,
       COALESCE(text_to_numeric_safe(s."CFF (-1FQFQ)"), 0)                       AS cff_1fqfq,
       COALESCE(text_to_numeric_safe(s."CFF (-2FQFQ)"), 0)                       AS cff_2fqfq,
       COALESCE(text_to_numeric_safe(s."CFF (-3FQFQ)"), 0)                       AS cff_3fqfq,
       COALESCE(text_to_numeric_safe(s."CFF (-4FQFQ)"), 0)                       AS cff_4fqfq,
       COALESCE(text_to_numeric_safe(s."CFO (-2FY)"), 0)                         AS cfo_2fy,
       COALESCE(text_to_numeric_safe(s."CFO (-3FY)"), 0)                         AS cfo_3fy,
       COALESCE(text_to_numeric_safe(s."CFO (-4FY)"), 0)                         AS cfo_4fy,
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-1FQFQ)"), 0)         AS cash_acquisitions_1fqfq,
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-2FQFQ)"), 0)         AS cash_acquisitions_2fqfq,
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-3FQFQ)"), 0)         AS cash_acquisitions_3fqfq,
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-4FQFQ)"), 0)         AS cash_acquisitions_4fqfq,
       COALESCE(text_to_numeric_safe(s."FCF (-2FY)"), 0)                         AS fcf_2fy,
       COALESCE(text_to_numeric_safe(s."FCF (-3FY)"), 0)                         AS fcf_3fy,
       COALESCE(text_to_numeric_safe(s."FCF (-4FY)"), 0)                         AS fcf_4fy,
       -- NEW: Cash Acquisitions Historical (FY)
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-2FY)"), 0)           AS cash_acquisitions_2fy,
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-3FY)"), 0)           AS cash_acquisitions_3fy,
       COALESCE(text_to_numeric_safe(s."Cash Acquisitions (-4FY)"), 0)           AS cash_acquisitions_4fy,
       -- NEW: Capital Expenditure Historical
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-1FQFQ)"), 0)       AS capital_expenditure_1fqfq,
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-2FQFQ)"), 0)       AS capital_expenditure_2fqfq,
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-3FQFQ)"), 0)       AS capital_expenditure_3fqfq,
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-4FQFQ)"), 0)       AS capital_expenditure_4fqfq,
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-2FY)"), 0)         AS capital_expenditure_2fy,
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-3FY)"), 0)         AS capital_expenditure_3fy,
       COALESCE(text_to_numeric_safe(s."Capital Expenditure (-4FY)"), 0)         AS capital_expenditure_4fy,
       -- Continue with existing P/E columns
       text_to_numeric_safe(s."P/E (NTM)")                                       AS p_e_ntm,
       text_to_numeric_safe(s."P/E (LTM)")                                       AS p_e_ltm,
       text_to_numeric_safe(s."Altman Z-Score (FY)")                             AS altman_z_score_fy,
       text_to_numeric_safe(s."Altman Z-Score (FQ)")                             AS altman_z_score_fq,
       text_to_numeric_safe(s."Altman Z-Score (LTM)")                            AS altman_z_score_ltm,
       text_to_numeric_safe(s."P/TBV (LTM)")                                     AS p_tbv_ltm,
       text_to_numeric_safe(s."Return On Equity % (LTM)")                        AS return_on_equity_pct_ltm,
       text_to_numeric_safe(s."Return On Equity % (FY)")                         AS return_on_equity_pct_fy,
       text_to_numeric_safe(s."Current Ratio (FY)")                              AS current_ratio_fy,
       text_to_numeric_safe(s."Current Ratio (LTM)")                             AS current_ratio_ltm,
       text_to_numeric_safe(s."Asset Turnover (FY)")                             AS asset_turnover_fy,
       text_to_numeric_safe(s."Asset Turnover (LTM)")                            AS asset_turnover_ltm,
       text_to_numeric_safe(s."EPS Norm - Est Avg (NTM)")                        AS eps_norm_est_avg_ntm,
       text_to_numeric_safe(s."EPS/Adj. (-1FY)")                                 AS eps_adj_1fy,
       text_to_numeric_safe(s."EPS/Adj. (FY)")                                   AS eps_adj_fy,
       text_to_numeric_safe(s."EPS/Adj. (LTM)")                                  AS eps_adj_ltm,
       text_to_numeric_safe(s."EPS Norm - Est Avg (FY1E)")                       AS eps_norm_est_avg_fy1e,
       text_to_numeric_safe(s."Return on Assets (ROA) % (LTM)")                  AS return_on_assets_roa_pct_ltm,
       text_to_numeric_safe(s."Return on Assets (ROA) % (FY)")                   AS return_on_assets_roa_pct_fy,
       text_to_numeric_safe(s."P/B (LTM)")                                       AS p_b_ltm,
       text_to_numeric_safe(s."P/B (-1FY)")                                      AS p_b_1fy,
       text_to_numeric_safe(s."P/B (5YAVG)")                                     AS p_b_5yavg,
       text_to_numeric_safe(s."EV/Sales (EST FY1)")                              AS ev_sales_est_fy1,
       text_to_numeric_safe(s."EV/Sales (LTM)")                                  AS ev_sales_ltm,
       text_to_numeric_safe(s."EV/Sales (NTM)")                                  AS ev_sales_ntm,
       text_to_numeric_safe(s."EV/Sales (-1FYLTM)")                              AS ev_sales_1fyltm,
       text_to_numeric_safe(s."EV/Sales (-2FYLTM)")                              AS ev_sales_2fyltm,
       text_to_numeric_safe(s."EV/Sales (-3FYLTM)")                              AS ev_sales_3fyltm,
       text_to_numeric_safe(s."EV/Sales (3YAVGLTM)")                             AS ev_sales_3yavgltm,
       text_to_numeric_safe(s."EV/Sales (-1FQLTM)")                              AS ev_sales_1fqltm,
       text_to_numeric_safe(s."EV/Sales (-2FQLTM)")                              AS ev_sales_2fqltm,
       text_to_numeric_safe(s."EV/Sales (-3FQLTM)")                              AS ev_sales_3fqltm,
       text_to_numeric_safe(s."EV/Sales (-4FQLTM)")                              AS ev_sales_4fqltm,
       text_to_numeric_safe(s."EV/EBITDA (LTM)")                                 AS ev_ebitda_ltm,
       text_to_numeric_safe(s."EV/EBITDA (NTM)")                                 AS ev_ebitda_ntm,
       text_to_numeric_safe(s."EV/EBITDA (-1FYLTM)")                             AS ev_ebitda_1fyltm,
       text_to_numeric_safe(s."EV/EBITDA (-1FQLTM)")                             AS ev_ebitda_1fqltm,
       text_to_numeric_safe(s."EV/EBITDA (3YAVGLTM)")                            AS ev_ebitda_3yavgltm,
       text_to_numeric_safe(s."EV/EBITDA (EST FY1)")                             AS ev_ebitda_est_fy1,
       text_to_numeric_safe(s."P/E (EST FY1)")                                   AS p_e_est_fy1,
       text_to_numeric_safe(s."P/E (-1FYLTM)")                                   AS p_e_1fyltm,
       text_to_numeric_safe(s."P/E (-2FYLTM)")                                   AS p_e_2fyltm,
       text_to_numeric_safe(s."P/E (-3FYLTM)")                                   AS p_e_3fyltm,
       text_to_numeric_safe(s."P/E (3YAVGLTM)")                                  AS p_e_3yavgltm,
       text_to_numeric_safe(s."P/E (-1FQLTM)")                                   AS p_e_1fqltm,
       text_to_numeric_safe(s."P/E (-2FQLTM)")                                   AS p_e_2fqltm,
       text_to_numeric_safe(s."P/E (-3FQLTM)")                                   AS p_e_3fqltm,
       text_to_numeric_safe(s."P/E (5YAVGLTM)")                                  AS p_e_5yavgltm,
       text_to_numeric_safe(s."P/E (-0FQQoQLTM)")                                AS p_e_0fqqoqltm,
       text_to_numeric_safe(s."P/E (-0FYYoYLTM)")                                AS p_e_0fyyoyltm,
       text_to_numeric_safe(s."P/E (-1FYYoYLTM)")                                AS p_e_1fyyoyltm,
       text_to_numeric_safe(s."P/E (-0FQYoYLTM)")                                AS p_e_0fqyoyltm,
       text_to_numeric_safe(s."Net EPS - Basic (LTM)")                           AS net_eps_basic_ltm,
       text_to_numeric_safe(s."Net EPS - Basic (FQ)")                            AS net_eps_basic_fq,
       text_to_numeric_safe(s."Net EPS - Basic (FY)")                            AS net_eps_basic_fy,
       text_to_numeric_safe(s."Net EPS - Basic (-1FQFQ)")                        AS net_eps_basic_1fqfq,
       text_to_numeric_safe(s."Net EPS - Basic (-2FQFQ)")                        AS net_eps_basic_2fqfq,
       text_to_numeric_safe(s."Net EPS - Basic (-3FQFQ)")                        AS net_eps_basic_3fqfq,
       text_to_numeric_safe(s."Net EPS - Basic (-4FQFQ)")                        AS net_eps_basic_4fqfq,
       text_to_numeric_safe(s."Net EPS - Basic (-1FY)")                          AS net_eps_basic_1fy,
       text_to_numeric_safe(s."Net EPS - Basic (-2FY)")                          AS net_eps_basic_2fy,
       text_to_numeric_safe(s."Net EPS - Basic (-3FY)")                          AS net_eps_basic_3fy,
       text_to_numeric_safe(s."Net EPS - Basic (-4FY)")                          AS net_eps_basic_4fy,
       text_to_numeric_safe(s."Net EPS - Basic (-5FY)")                          AS net_eps_basic_5fy,
       text_to_numeric_safe(s."EPS GAAP - Est Avg (NTM)")                        AS eps_gaap_est_avg_ntm,
       text_to_numeric_safe(s."EPS GAAP - Est Avg (FY1E)")                       AS eps_gaap_est_avg_fy1e,
       -- NEW: Basic EPS - Cont Historical
       text_to_numeric_safe(s."Basic EPS - Cont (LTM)")                          AS basic_eps_cont_ltm,
       text_to_numeric_safe(s."Basic EPS - Cont (FQ)")                           AS basic_eps_cont_fq,
       text_to_numeric_safe(s."Basic EPS - Cont (FY)")                           AS basic_eps_cont_fy,
       text_to_numeric_safe(s."Basic EPS - Cont (-1FQFQ)")                       AS basic_eps_cont_1fqfq,
       text_to_numeric_safe(s."Basic EPS - Cont (-2FQFQ)")                       AS basic_eps_cont_2fqfq,
       text_to_numeric_safe(s."Basic EPS - Cont (-3FQFQ)")                       AS basic_eps_cont_3fqfq,
       text_to_numeric_safe(s."Basic EPS - Cont (-4FQFQ)")                       AS basic_eps_cont_4fqfq,
       text_to_numeric_safe(s."Basic EPS - Cont (-1FY)")                         AS basic_eps_cont_1fy,
       text_to_numeric_safe(s."Basic EPS - Cont (-2FY)")                         AS basic_eps_cont_2fy,
       text_to_numeric_safe(s."Basic EPS - Cont (-3FY)")                         AS basic_eps_cont_3fy,
       text_to_numeric_safe(s."Basic EPS - Cont (-4FY)")                         AS basic_eps_cont_4fy,
       -- NEW: EPS/Adj. Historical
       text_to_numeric_safe(s."EPS/Adj. (FQ)")                                   AS eps_adj_fq,
       text_to_numeric_safe(s."EPS/Adj. (-1FQFQ)")                               AS eps_adj_1fqfq,
       text_to_numeric_safe(s."EPS/Adj. (-2FQFQ)")                               AS eps_adj_2fqfq,
       text_to_numeric_safe(s."EPS/Adj. (-3FQFQ)")                               AS eps_adj_3fqfq,
       text_to_numeric_safe(s."EPS/Adj. (-4FQFQ)")                               AS eps_adj_4fqfq,
       text_to_numeric_safe(s."EPS/Adj. (-2FY)")                                 AS eps_adj_2fy,
       text_to_numeric_safe(s."EPS/Adj. (-3FY)")                                 AS eps_adj_3fy,
       text_to_numeric_safe(s."EPS/Adj. (-4FY)")                                 AS eps_adj_4fy,
       text_to_numeric_safe(s."Total Return (YTD)")                              AS total_return_ytd,
       text_to_numeric_safe(s."Beta (1Y)")                                       AS beta_1y,
       text_to_numeric_safe(s."Beta (2Y)")                                       AS beta_2y,
       text_to_numeric_safe(s."Beta (5Y)")                                       AS beta_5y,
       text_to_numeric_safe(s."Total Revenues/CAGR (5Y FY)")                     AS total_revenues_cagr_5y_fy,
       text_to_numeric_safe(s."Tot. Return %/CAGR (3Y)")                         AS tot_return_pct_cagr_3y,
       text_to_numeric_safe(s."Tot. Return %/CAGR (10Y)")                        AS tot_return_pct_cagr_10y,
       text_to_numeric_safe(s."Total Return (5Y)")                               AS total_return_5y,
       text_to_numeric_safe(s."Total Return (10Y)")                              AS total_return_10y,
       text_to_numeric_safe(s."Net Income Margin % (FY)")                        AS net_income_margin_pct_fy,
       text_to_numeric_safe(s."Net Income Margin % (LTM)")                       AS net_income_margin_pct_ltm,
       text_to_numeric_safe(s."Volatility (1M)")                                 AS volatility_1m,
       text_to_numeric_safe(s."Volatility (3M)")                                 AS volatility_3m,
       text_to_numeric_safe(s."Volatility (6M)")                                 AS volatility_6m,
       text_to_numeric_safe(s."Volatility (1Y)")                                 AS volatility_1y,
       text_to_numeric_safe(s."Div Yield (Ind)")                                 AS div_yield_ind,
       text_to_numeric_safe(s."Div Yield (LTM)")                                 AS div_yield_ltm,
       text_to_numeric_safe(s."Gross Profit Margin % (FY)")                      AS gross_profit_margin_pct_fy,
       text_to_numeric_safe(s."Gross Profit Margin % (LTM)")                     AS gross_profit_margin_pct_ltm,
       text_to_numeric_safe(s."Buyback Yield (LTM)")                             AS buyback_yield_ltm,
       text_to_numeric_safe(s."Div Yield (-1FYInd)")                             AS div_yield_1fyind,
       text_to_numeric_safe(s."Div Yield (TTM)")                                 AS div_yield_ttm,
       text_to_numeric_safe(s."Div Yield (NTM)")                                 AS div_yield_ntm,
       text_to_numeric_safe(s."Div Yield (5YAVGLTM)")                            AS div_yield_5yavgltm,
       text_to_numeric_safe(s."Revenues - Est YoY % (FY1E)")                     AS revenues_est_yoy_pct_fy1e,
       text_to_numeric_safe(s."Price Chg. % (1M)")                               AS price_chg_pct_1m,
       text_to_numeric_safe(s."Price Chg. % (3M)")                               AS price_chg_pct_3m,
       text_to_numeric_safe(s."1-Day %")                                         AS one_day_pct,
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 1W)")                   AS eps_est_avg_rev_pct_fy1e_1w,
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 1M)")                   AS eps_est_avg_rev_pct_fy1e_1m,
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 3M)")                   AS eps_est_avg_rev_pct_fy1e_3m,
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 6M)")                   AS eps_est_avg_rev_pct_fy1e_6m,
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 1Y)")                   AS eps_est_avg_rev_pct_fy1e_1y,
       text_to_numeric_safe(s."Div Yield (-2FYInd)")                             AS div_yield_2fyind,
       text_to_numeric_safe(s."Div Yield (-3FYInd)")                             AS div_yield_3fyind,
       text_to_numeric_safe(s."Div Yield (-4FYInd)")                             AS div_yield_4fyind,
       text_to_numeric_safe(s."Div Yield (-5FYInd)")                             AS div_yield_5fyind,
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 1M)")              AS eps_gaap_est_avg_rev_pct_fy1e_1m,
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 3M)")              AS eps_gaap_est_avg_rev_pct_fy1e_3m,
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 6M)")              AS eps_gaap_est_avg_rev_pct_fy1e_6m,
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 1Y)")              AS eps_gaap_est_avg_rev_pct_fy1e_1y,
       COALESCE(text_to_numeric_safe(s."Dividend Streak"), 0)                    AS dividend_streak,
       COALESCE(text_to_numeric_safe(s."Price Target - #"), 0)                   AS price_target_count,
       COALESCE(text_to_numeric_safe(s."Analyst Rating"), 0)                     AS analyst_rating,
       COALESCE(text_to_numeric_safe(s."# Strong Sell Ratings"), 0)              AS num_strong_sell_ratings,
       COALESCE(text_to_numeric_safe(s."# Strong Buys Ratings"), 0)              AS num_strong_buys_ratings,
       COALESCE(text_to_numeric_safe(s."# Hold Ratings"), 0)                     AS num_hold_ratings,
       COALESCE(text_to_numeric_safe(s."# Buys Ratings"), 0)                     AS num_buys_ratings,
       COALESCE(text_to_numeric_safe(s."# Sell Ratings"), 0)                     AS num_sell_ratings,
       COALESCE(text_to_numeric_safe(s."# No Opinion Ratings"), 0)               AS num_no_opinion_ratings,
       COALESCE(text_to_numeric_safe(s."Shrs Out"), 0)                           AS shares_outstanding,
       COALESCE(text_to_numeric_safe(s."Shrs Out (-1FY)"), 0)                    AS shrs_out_1fy,
       COALESCE(text_to_numeric_safe(s."Full Time Employees (FQ)"), 0)           AS full_time_employees_fq,
       COALESCE(text_to_numeric_safe(s."Full Time Employees (FY)"), 0)           AS full_time_employees_fy,
       COALESCE(text_to_numeric_safe(s."Full Time Employees (-1FY)"), 0)         AS full_time_employees_1fy,
       COALESCE(text_to_numeric_safe(s."Full Time Employees (-2FY)"), 0)         AS full_time_employees_2fy,
       COALESCE(text_to_numeric_safe(s."Full Time Employees (-3FY)"), 0)         AS full_time_employees_3fy,
       COALESCE(text_to_numeric_safe(s."Avg Employees (5YAVGFY)"), 0)            AS avg_employees_5yavgfy,
       COALESCE(text_to_numeric_safe(s."EPS Norm - Est # (FY1E)"), 0)            AS eps_norm_est_num_fy1e,
       COALESCE(text_to_numeric_safe(s."Price Target - # (3M Ago)"), 0)          AS price_target_num_3m_ago,
       COALESCE(text_to_numeric_safe(s."Price Target - # (6M Ago)"), 0)          AS price_target_num_6m_ago,
       COALESCE(text_to_numeric_safe(s."Price Target - # (YTD Ago)"), 0)         AS price_target_num_ytd_ago,
       COALESCE(text_to_numeric_safe(s."Price Target - # (1Y Ago)"), 0)          AS price_target_num_1y_ago,
       COALESCE(text_to_numeric_safe(s."Price Target - # (1W Ago)"), 0)          AS price_target_num_1w_ago,
       COALESCE(text_to_numeric_safe(s."Price Target - # (1M Ago)"), 0)          AS price_target_num_1m_ago,
       COALESCE(text_to_numeric_safe(s."Price Target - # (MTD Ago)"), 0)         AS price_target_num_mtd_ago,
       COALESCE(text_to_numeric_safe(s."Price Target - # (QTD Ago)"), 0)         AS price_target_num_qtd_ago,
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (LTM)"),
                0)                                                               AS gain_loss_on_sale_of_assets_ltm,
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (FQ)"), 0)        AS impairment_of_goodwill_fq,
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (LTM)"), 0)       AS impairment_of_goodwill_ltm,
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-1FY)"),
                0)                                                               AS impairment_of_goodwill_1fy,
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (FY)"), 0)        AS impairment_of_goodwill_fy,
       COALESCE(text_to_numeric_safe(s."Asset Writedown (LTM)"), 0)              AS asset_writedown_ltm,
       COALESCE(text_to_numeric_safe(s."Asset Writedown (FY)"), 0)               AS asset_writedown_fy,
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-1FY)"), 0)             AS asset_writedown_1fy,
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (LTM)"), 0)        AS restructuring_charges_ltm,
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (FQ)"), 0)         AS restructuring_charges_fq,
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-1FY)"), 0)       AS restructuring_charges_1fy,
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (FY)"), 0)         AS restructuring_charges_fy,
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (LTM)"),
                0)                                                               AS merger_and_restructuring_charges_ltm,
       COALESCE(text_to_numeric_safe(s."Other Unusual Items/Total (LTM)"),
                0)                                                               AS other_unusual_items_total_ltm,
       COALESCE(text_to_numeric_safe(s."Asset Writedown (FQ)"), 0)               AS asset_writedown_fq,
       COALESCE(text_to_numeric_safe(s."Asset Writedown (5YAVGFQ)"), 0)          AS asset_writedown_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (5YAVGFQ)"),
                0)                                                               AS impairment_of_goodwill_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (5YAVGFQ)"),
                0)                                                               AS restructuring_charges_5yavgfq,
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (FQ)"),
                0)                                                               AS merger_and_restructuring_charges_fq,
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (FY)"),
                0)                                                               AS merger_and_restructuring_charges_fy,
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (5YAVGFQ)"),
                0)                                                               AS merger_and_restructuring_charges_5yavgfq,
       -- NEW: Merger & Restructuring Charges Historical
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-1FQFQ)"),
                0)                                                               AS merger_and_restructuring_charges_1fqfq,
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-2FQFQ)"),
                0)                                                               AS merger_and_restructuring_charges_2fqfq,
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-3FQFQ)"),
                0)                                                               AS merger_and_restructuring_charges_3fqfq,
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-4FQFQ)"),
                0)                                                               AS merger_and_restructuring_charges_4fqfq,
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-1FY)"),
                0)                                                               AS merger_and_restructuring_charges_1fy,
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-2FY)"),
                0)                                                               AS merger_and_restructuring_charges_2fy,
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-3FY)"),
                0)                                                               AS merger_and_restructuring_charges_3fy,
       COALESCE(text_to_numeric_safe(s."Merger & Restructuring Charges (-4FY)"),
                0)                                                               AS merger_and_restructuring_charges_4fy,
       -- NEW: Impairment of Goodwill Historical
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-1FQFQ)"),
                0)                                                               AS impairment_of_goodwill_1fqfq,
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-2FQFQ)"),
                0)                                                               AS impairment_of_goodwill_2fqfq,
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-3FQFQ)"),
                0)                                                               AS impairment_of_goodwill_3fqfq,
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-4FQFQ)"),
                0)                                                               AS impairment_of_goodwill_4fqfq,
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-2FY)"),
                0)                                                               AS impairment_of_goodwill_2fy,
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-3FY)"),
                0)                                                               AS impairment_of_goodwill_3fy,
       COALESCE(text_to_numeric_safe(s."Impairment of Goodwill (-4FY)"),
                0)                                                               AS impairment_of_goodwill_4fy,
       -- NEW: Asset Writedown Historical
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-1FQFQ)"), 0)           AS asset_writedown_1fqfq,
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-2FQFQ)"), 0)           AS asset_writedown_2fqfq,
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-3FQFQ)"), 0)           AS asset_writedown_3fqfq,
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-4FQFQ)"), 0)           AS asset_writedown_4fqfq,
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-2FY)"), 0)             AS asset_writedown_2fy,
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-3FY)"), 0)             AS asset_writedown_3fy,
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-4FY)"), 0)             AS asset_writedown_4fy,
       COALESCE(text_to_numeric_safe(s."Asset Writedown (-5FY)"), 0)             AS asset_writedown_5fy,
       -- NEW: Gain (Loss) On Sale Of Assets Historical
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (FQ)"),
                0)                                                               AS gain_loss_on_sale_of_assets_fq,
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (FY)"),
                0)                                                               AS gain_loss_on_sale_of_assets_fy,
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-1FQFQ)"),
                0)                                                               AS gain_loss_on_sale_of_assets_1fqfq,
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-2FQFQ)"),
                0)                                                               AS gain_loss_on_sale_of_assets_2fqfq,
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-3FQFQ)"),
                0)                                                               AS gain_loss_on_sale_of_assets_3fqfq,
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-4FQFQ)"),
                0)                                                               AS gain_loss_on_sale_of_assets_4fqfq,
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-1FY)"),
                0)                                                               AS gain_loss_on_sale_of_assets_1fy,
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-2FY)"),
                0)                                                               AS gain_loss_on_sale_of_assets_2fy,
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-3FY)"),
                0)                                                               AS gain_loss_on_sale_of_assets_3fy,
       COALESCE(text_to_numeric_safe(s."Gain (Loss) On Sale Of Assets (-4FY)"),
                0)                                                               AS gain_loss_on_sale_of_assets_4fy,
       -- NEW: Restructuring Charges Historical
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-1FQFQ)"),
                0)                                                               AS restructuring_charges_1fqfq,
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-2FQFQ)"),
                0)                                                               AS restructuring_charges_2fqfq,
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-3FQFQ)"),
                0)                                                               AS restructuring_charges_3fqfq,
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-4FQFQ)"),
                0)                                                               AS restructuring_charges_4fqfq,
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-2FY)"), 0)       AS restructuring_charges_2fy,
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-3FY)"), 0)       AS restructuring_charges_3fy,
       COALESCE(text_to_numeric_safe(s."Restructuring Charges (-4FY)"), 0)       AS restructuring_charges_4fy,
       COALESCE(text_to_numeric_safe(s."Interest And Investment Income (LTM)"),
                0)                                                               AS interest_and_investment_income_ltm,
       COALESCE(text_to_numeric_safe(s."Interest And Investment Income (FQ)"),
                0)                                                               AS interest_and_investment_income_fq,
       COALESCE(text_to_numeric_safe(s."Interest And Investment Income (FY)"),
                0)                                                               AS interest_and_investment_income_fy,
       COALESCE(text_to_numeric_safe(s."Interest And Investment Income (-1FQFQ)"),
                0)                                                               AS interest_and_investment_income_1fqfq,
       COALESCE(text_to_numeric_safe(s."Interest And Investment Income (-2FQFQ)"),
                0)                                                               AS interest_and_investment_income_2fqfq,
       COALESCE(text_to_numeric_safe(s."Interest And Investment Income (-3FQFQ)"),
                0)                                                               AS interest_and_investment_income_3fqfq,
       COALESCE(text_to_numeric_safe(s."Interest And Investment Income (-4FQFQ)"),
                0)                                                               AS interest_and_investment_income_4fqfq,
       COALESCE(text_to_numeric_safe(s."Interest And Investment Income (-1FY)"),
                0)                                                               AS interest_and_investment_income_1fy,
       COALESCE(text_to_numeric_safe(s."Interest And Investment Income (-2FY)"),
                0)                                                               AS interest_and_investment_income_2fy,
       COALESCE(text_to_numeric_safe(s."Interest And Investment Income (-3FY)"),
                0)                                                               AS interest_and_investment_income_3fy,
       COALESCE(text_to_numeric_safe(s."Interest And Investment Income (-4FY)"),
                0)                                                               AS interest_and_investment_income_4fy,
       COALESCE(text_to_numeric_safe(s."Effective Tax Rate - (Ratio) (LTM)"),
                0)                                                               AS effective_tax_rate_ltm,
       COALESCE(text_to_numeric_safe(s."Effective Tax Rate - (Ratio) (FQ)"),
                0)                                                               AS effective_tax_rate_fq,
       COALESCE(text_to_numeric_safe(s."Effective Tax Rate - (Ratio) (FY)"),
                0)                                                               AS effective_tax_rate_fy,
       COALESCE(text_to_numeric_safe(s."Effective Tax Rate - (Ratio) (-1FQFQ)"),
                0)                                                               AS effective_tax_rate_1fqfq,
       COALESCE(text_to_numeric_safe(s."Effective Tax Rate - (Ratio) (-2FQFQ)"),
                0)                                                               AS effective_tax_rate_2fqfq,
       COALESCE(text_to_numeric_safe(s."Effective Tax Rate - (Ratio) (-3FQFQ)"),
                0)                                                               AS effective_tax_rate_3fqfq,
       COALESCE(text_to_numeric_safe(s."Effective Tax Rate - (Ratio) (-4FQFQ)"),
                0)                                                               AS effective_tax_rate_4fqfq,
       COALESCE(text_to_numeric_safe(s."Effective Tax Rate - (Ratio) (-1FY)"),
                0)                                                               AS effective_tax_rate_1fy,
       COALESCE(text_to_numeric_safe(s."Effective Tax Rate - (Ratio) (-2FY)"),
                0)                                                               AS effective_tax_rate_2fy,
       COALESCE(text_to_numeric_safe(s."Effective Tax Rate - (Ratio) (-3FY)"),
                0)                                                               AS effective_tax_rate_3fy,
       COALESCE(text_to_numeric_safe(s."Effective Tax Rate - (Ratio) (-4FY)"),
                0)                                                               AS effective_tax_rate_4fy,
       COALESCE(text_to_numeric_safe(s."FCF - Est Avg (FY1E)"),
                0)                                                               AS fcf_est_avg_fy1e,
       COALESCE(text_to_numeric_safe(s."FCF - Est Avg (FY2E)"),
                0)                                                               AS fcf_est_avg_fy2e,
       COALESCE(text_to_numeric_safe(s."FCF - Est Avg (FY3E)"),
                0)                                                               AS fcf_est_avg_fy3e,
       COALESCE(text_to_numeric_safe(s."FCF - Est Avg (FY4E)"),
                0)                                                               AS fcf_est_avg_fy4e,
       COALESCE(text_to_numeric_safe(s."FCF - Est Avg (FY5E)"),
                0)                                                               AS fcf_est_avg_fy5e,
       COALESCE(text_to_numeric_safe(s."Total Operating Expenses (LTM)"),
                0)                                                               AS total_operating_expenses_ltm,
       COALESCE(text_to_numeric_safe(s."Total Operating Expenses (FQ)"),
                0)                                                               AS total_operating_expenses_fq,
       COALESCE(text_to_numeric_safe(s."Total Operating Expenses (FY)"),
                0)                                                               AS total_operating_expenses_fy,
       COALESCE(text_to_numeric_safe(s."Total Operating Expenses (-1FQFQ)"),
                0)                                                               AS total_operating_expenses_1fqfq,
       COALESCE(text_to_numeric_safe(s."Total Operating Expenses (-2FQFQ)"),
                0)                                                               AS total_operating_expenses_2fqfq,
       COALESCE(text_to_numeric_safe(s."Total Operating Expenses (-3FQFQ)"),
                0)                                                               AS total_operating_expenses_3fqfq,
       COALESCE(text_to_numeric_safe(s."Total Operating Expenses (-4FQFQ)"),
                0)                                                               AS total_operating_expenses_4fqfq,
       COALESCE(text_to_numeric_safe(s."Total Operating Expenses (-1FY)"),
                0)                                                               AS total_operating_expenses_1fy,
       COALESCE(text_to_numeric_safe(s."Total Operating Expenses (-2FY)"),
                0)                                                               AS total_operating_expenses_2fy,
       COALESCE(text_to_numeric_safe(s."Total Operating Expenses (-3FY)"),
                0)                                                               AS total_operating_expenses_3fy,
       COALESCE(text_to_numeric_safe(s."Total Operating Expenses (-4FY)"),
                0)                                                               AS total_operating_expenses_4fy,
       report_fiscal.fiscal_month                                                AS fiscal_month,
       report_fiscal.fiscal_quarter                                              AS fiscal_quarter,
       report_fiscal.fiscal_year                                                 AS fiscal_year,
       calculate_reporting_lag(
               NULLIF(TRIM(s."Next Earnings"), '')::DATE,
               NULLIF(TRIM(s."Income Statement Report Date"), '')::DATE,
               report_fiscal.earnings_report_frequency
       )                                                                         AS reporting_lag
FROM screening_staging s,
     LATERAL (
         SELECT parse_fiscal_year_end_date(NULLIF(TRIM(s."FY End"), '')) AS fy_end_date
         )             parsed,
     LATERAL (
         SELECT calculate_next_fy_end_date(parsed.fy_end_date) AS next_fy_end_date
         )             next_fy,
     LATERAL (
         SELECT * FROM calculate_fiscal_info(CURRENT_DATE::DATE, parsed.fy_end_date, NULL::TEXT)
         )             current_fiscal,
     LATERAL (
         SELECT *
         FROM calculate_fiscal_info(NULLIF(TRIM(s."Income Statement Report Date"), '')::DATE, parsed.fy_end_date,
                                    NULL::TEXT)
         )             report_fiscal
ON CONFLICT DO NOTHING;

-- FINAL VALIDATION
-- ===================================================================
\echo 'Final validation...'
SELECT 'Total rows in equities:' AS info, COUNT(*) AS count
FROM equities;
SELECT 'Rows by Region:' AS info, "Region", COUNT(*) AS count
FROM equities
GROUP BY "Region"
ORDER BY "Region";
SELECT 'Rows by Sector (top 10):' AS info, "Sector", COUNT(*) AS count
FROM equities
GROUP BY "Sector"
ORDER BY COUNT(*) DESC
LIMIT 10;

-- ===================================================================
-- CLEANUP
-- ===================================================================
DROP TABLE IF EXISTS screening_staging;
DROP FUNCTION IF EXISTS text_to_numeric_safe(TEXT);
DROP FUNCTION IF EXISTS text_to_date_safe(TEXT, TEXT);
DROP FUNCTION IF EXISTS month_abbrev_to_number(TEXT);
DROP FUNCTION IF EXISTS get_expected_reporting_lag_days(TEXT);
DROP FUNCTION IF EXISTS parse_fiscal_year_end_date(TEXT);
DROP FUNCTION IF EXISTS frequency_to_months(TEXT);
DROP FUNCTION IF EXISTS months_to_frequency(INTEGER);
DROP FUNCTION IF EXISTS derive_earnings_report_frequency(DATE, DATE);
DROP FUNCTION IF EXISTS calculate_fiscal_info(DATE, DATE, TEXT);
DROP FUNCTION IF EXISTS calculate_next_income_statement_report_date(DATE, TEXT);
DROP FUNCTION IF EXISTS calculate_next_fy_end_date(DATE);
DROP FUNCTION IF EXISTS calculate_next_fiscal_quarter(INTEGER, TEXT);
DROP FUNCTION IF EXISTS calculate_reporting_lag(DATE, DATE, TEXT);
DROP FUNCTION IF EXISTS calculate_expected_report_date(DATE, TEXT);
DROP FUNCTION IF EXISTS validate_fiscal_dates(DATE, DATE, DATE);

\echo 'Import complete!'
