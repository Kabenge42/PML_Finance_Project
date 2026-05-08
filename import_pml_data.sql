-- ===================================================================
-- Equities Data Import Script
-- ===================================================================
-- Documentation: See docs/column_mapping_reference.md for column aliases
-- Usage: psql -h localhost -p 5432 -U postgres -d postgres -f import_pml_data.sql
\echo 'Starting pml_df data import...'

-- ===================================================================
-- SESSION-LEVEL TUNING FOR BULK IMPORT
-- ===================================================================
SET search_path = pml, public; -- Resolve pml_df / pml_us without schema prefix
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
SELECT 'Current pml_df table row count:' AS status, COUNT(*) AS row_count
FROM pml_df;

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
SELECT CASE UPPER(LEFT(TRIM(COALESCE(month_abbrev, '')), 3))
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
           END
$$ LANGUAGE SQL IMMUTABLE
                PARALLEL SAFE;

-- ===================================================================
-- HELPER FUNCTION: Get Expected Reporting Lag Days
-- ===================================================================
-- Returns the typical number of days between period end and earnings release
CREATE OR REPLACE FUNCTION get_expected_reporting_lag_days(earnings_report_frequency TEXT)
    RETURNS INTEGER AS
$$
SELECT CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'QUARTERLY')))
           WHEN 'QUARTERLY' THEN 45
           WHEN 'SEMI-ANNUAL' THEN 60
           WHEN 'SEMI-ANNUALLY' THEN 60
           WHEN 'ANNUAL' THEN 90
           WHEN 'ANNUALLY' THEN 90
           ELSE 45
           END
$$ LANGUAGE SQL IMMUTABLE
                PARALLEL SAFE;

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

-- Converts TEXT to DATE safely, returns NULL for invalid input.
-- Auto-detects common formats found in vendor CSV exports, including:
--   * 'Mon-DD-YYYY'  (e.g. 'Mar-16-2026')   <-- primary PML CSV format
--   * 'YYYY-MM-DD'   (ISO 8601)
--   * 'MM/DD/YYYY'   (US)
--   * 'DD/MM/YYYY'   (EU)
-- If `date_format` is supplied explicitly (and not the default sentinel
-- 'AUTO'), it is honored as-is via TO_DATE().
CREATE OR REPLACE FUNCTION text_to_date_safe(input_text TEXT, date_format TEXT DEFAULT 'AUTO')
    RETURNS DATE AS
$$
DECLARE
    v TEXT;
BEGIN
    IF input_text IS NULL THEN
        RETURN NULL;
    END IF;
    v := TRIM(input_text);
    IF v IN ('', '-', '--', 'N/A', 'NA', 'NULL', 'NONE', 'n/a', 'na', 'null', 'none') THEN
        RETURN NULL;
    END IF;

    -- Explicit format requested by caller
    IF date_format IS NOT NULL AND date_format <> 'AUTO' THEN
        BEGIN
            RETURN TO_DATE(v, date_format);
        EXCEPTION WHEN OTHERS THEN
            RETURN NULL;
        END;
    END IF;

    -- Auto-detect by shape
    -- 'Mon-DD-YYYY' e.g. Mar-16-2026
    IF v ~ '^[A-Za-z]{3}-\d{1,2}-\d{4}$' THEN
        BEGIN
            RETURN TO_DATE(v, 'Mon-DD-YYYY');
        EXCEPTION WHEN OTHERS THEN
            RETURN NULL;
        END;
    END IF;

    -- 'Mon DD, YYYY' e.g. 'Mar 16, 2026'
    IF v ~ '^[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}$' THEN
        BEGIN
            RETURN TO_DATE(v, 'Mon DD, YYYY');
        EXCEPTION WHEN OTHERS THEN
            RETURN NULL;
        END;
    END IF;

    -- ISO 'YYYY-MM-DD'
    IF v ~ '^\d{4}-\d{2}-\d{2}$' THEN
        BEGIN
            RETURN TO_DATE(v, 'YYYY-MM-DD');
        EXCEPTION WHEN OTHERS THEN
            RETURN NULL;
        END;
    END IF;

    -- 'MM/DD/YYYY' (US)
    IF v ~ '^\d{1,2}/\d{1,2}/\d{4}$' THEN
        BEGIN
            RETURN TO_DATE(v, 'MM/DD/YYYY');
        EXCEPTION WHEN OTHERS THEN
            RETURN NULL;
        END;
    END IF;

    -- Last-resort: let PostgreSQL's input parser try
    BEGIN
        RETURN v::DATE;
    EXCEPTION WHEN OTHERS THEN
        RETURN NULL;
    END;
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
    parts      TEXT[];
    month_num  INT;
    year_value INT;
BEGIN
    IF fy_end_text IS NULL OR TRIM(fy_end_text) = '' THEN
        RETURN NULL;
    END IF;

    parts := regexp_split_to_array(TRIM(fy_end_text), '\s+');
    IF array_length(parts, 1) < 2 OR parts[2] !~ '^\d{4}$' THEN
        RETURN NULL;
    END IF;

    year_value := parts[2]::INT;
    month_num := month_abbrev_to_number(parts[1]);

    IF month_num IS NULL OR year_value NOT BETWEEN 1900 AND 2100 THEN
        RETURN NULL;
    END IF;

    -- Last day of month via single interval literal date-math idiom
    RETURN (MAKE_DATE(year_value, month_num, 1)
        + INTERVAL '1 month - 1 day')::DATE;
END;
$$ LANGUAGE plpgsql IMMUTABLE
                    STRICT;

-- ===================================================================
-- HELPER FUNCTION: Convert Frequency to Interval Months
-- ===================================================================
CREATE OR REPLACE FUNCTION frequency_to_months(
    earnings_report_frequency TEXT,
    fy_end_date DATE DEFAULT NULL,
    next_fy_end_date DATE DEFAULT NULL
)
    RETURNS INTEGER AS
$$
DECLARE
    fy_range_months INT := 12;
BEGIN
    -- Use AGE() for month arithmetic — correct across year boundaries
    IF fy_end_date IS NOT NULL AND next_fy_end_date IS NOT NULL THEN
        fy_range_months := (DATE_PART('year', AGE(next_fy_end_date, fy_end_date)) * 12
            + DATE_PART('month', AGE(next_fy_end_date, fy_end_date)))::INT;
    END IF;

    RETURN CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'QUARTERLY')))
               WHEN 'QUARTERLY' THEN fy_range_months / 4
               WHEN 'SEMI-ANNUAL' THEN fy_range_months / 2
               WHEN 'SEMI-ANNUALLY' THEN fy_range_months / 2
               WHEN 'ANNUAL' THEN fy_range_months
               WHEN 'ANNUALLY' THEN fy_range_months
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
SELECT CASE
           WHEN interval_months IS NULL THEN 'Quarterly'
           WHEN interval_months <= 3 THEN 'Quarterly'
           WHEN interval_months <= 6 THEN 'Semi-Annually'
           ELSE 'Annually'
           END
$$ LANGUAGE SQL IMMUTABLE
                PARALLEL SAFE;

-- ===================================================================
-- HELPER FUNCTION: Derive Earnings Report Frequency
-- ===================================================================
CREATE OR REPLACE FUNCTION derive_earnings_report_frequency(
    income_statement_report_date DATE,
    fy_end_date DATE
)
    RETURNS TEXT AS
$$
DECLARE
    months_diff INT;
BEGIN
    IF income_statement_report_date IS NULL OR fy_end_date IS NULL THEN
        RETURN 'Quarterly';
    END IF;

    -- AGE() handles direction & year wrap automatically; it also
    -- respects the DAY component, unlike raw EXTRACT() subtraction.
    months_diff := ABS(
            (DATE_PART('year', AGE(income_statement_report_date, fy_end_date)) * 12
                + DATE_PART('month', AGE(income_statement_report_date, fy_end_date)))::INT
                   );

    -- Normalize within a 12-month window, but treat exact FY-end (0) as Annually
    -- rather than conflating it with Semi-Annually.
    IF months_diff = 0 THEN
        RETURN 'Annually';
    END IF;

    months_diff := months_diff % 12;
    IF months_diff = 0 THEN
        months_diff := 12;
    END IF;

    RETURN CASE
               WHEN months_diff = 12 THEN 'Annually'
               WHEN months_diff = 6 THEN 'Semi-Annually'
               ELSE 'Quarterly'
        END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- Unified Fiscal Date Calculator
-- Derives all calculations based on FY End Date reporting ranges
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_fiscal_info(
    reference_date DATE,
    fy_end_date DATE,
    input_earnings_frequency TEXT DEFAULT NULL,
    OUT fiscal_month INTEGER,
    OUT fiscal_quarter INTEGER,
    OUT fiscal_year INTEGER,
    OUT next_quarter INTEGER,
    OUT next_quarter_year INTEGER,
    OUT reporting_interval INTEGER,
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

    -- Use make_interval + AGE() for date-math-correct boundaries
    next_fy_end_date := (fy_end_date + make_interval(years => 1))::DATE;

    -- Fiscal year range in months via AGE() (respects day component)
    fy_range_months := (DATE_PART('year', AGE(next_fy_end_date, fy_end_date)) * 12
        + DATE_PART('month', AGE(next_fy_end_date, fy_end_date)))::INTEGER;

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

    -- Months since fiscal year end via AGE() (correct across year boundaries
    -- AND accounts for the day of month, unlike raw EXTRACT subtraction).
    months_since_fy_end := (DATE_PART('year', AGE(reference_date, fy_end_date)) * 12
        + DATE_PART('month', AGE(reference_date, fy_end_date)))::INTEGER;

    -- Fiscal month (1..fy_range_months); safe for negative months_since_fy_end too
    fiscal_month := ((months_since_fy_end - 1) % fy_range_months + fy_range_months) % fy_range_months + 1;

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
    earnings_report_frequency TEXT
)
    RETURNS DATE AS
$$
SELECT CASE
           WHEN income_statement_report_date IS NULL THEN NULL
           ELSE (income_statement_report_date
               + make_interval(months => frequency_to_months(earnings_report_frequency)))::DATE
           END
$$ LANGUAGE SQL IMMUTABLE
                PARALLEL SAFE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next Fiscal Year End Date
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_next_fy_end_date(fy_end_date DATE)
    RETURNS DATE AS
$$
SELECT (fy_end_date + make_interval(years => 1))::DATE
$$ LANGUAGE SQL IMMUTABLE
                STRICT
                PARALLEL SAFE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next Fiscal Quarter
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_next_fiscal_quarter(
    next_earnings_date DATE,
    income_statement_report_date DATE,
    fy_end_date DATE,
    earnings_report_frequency TEXT DEFAULT 'Quarterly'
)
    RETURNS INTEGER AS
$$
DECLARE
    reference_date   DATE;
    interval_months  INT;
    years_ahead      INT;
    current_fy_start DATE;
    months_into_fy   INT;
BEGIN
    IF fy_end_date IS NULL THEN
        RETURN NULL;
    END IF;

    interval_months := frequency_to_months(earnings_report_frequency);

    -- Choose reference date
    IF income_statement_report_date IS NOT NULL THEN
        reference_date := income_statement_report_date;
    ELSIF next_earnings_date IS NOT NULL THEN
        reference_date := next_earnings_date;
    ELSE
        RETURN NULL;
    END IF;

    -- How many whole fiscal years between fy_end and reference, using AGE()
    -- so that day-of-month is respected. FLOOR + 1 keeps us inside the CURRENT
    -- fiscal year even when reference_date falls exactly on an FY boundary.
    IF reference_date <= fy_end_date THEN
        years_ahead := 0;
    ELSE
        years_ahead := FLOOR(
                               (DATE_PART('year', AGE(reference_date, fy_end_date)) * 12
                                   + DATE_PART('month', AGE(reference_date, fy_end_date)))::NUMERIC / 12
                       )::INT + 1;
    END IF;

    -- Start of the current fiscal year = (fy_end + (years_ahead - 1) years) + 1 day
    current_fy_start := (fy_end_date + make_interval(years => years_ahead - 1)
        + INTERVAL '1 day')::DATE;

    -- Months into FY using AGE (handles month-length variations correctly)
    months_into_fy := (DATE_PART('year', AGE(reference_date, current_fy_start)) * 12
        + DATE_PART('month', AGE(reference_date, current_fy_start)))::INT + 1;

    -- Safe 1–12 normalization even for negative values
    months_into_fy := ((months_into_fy - 1) % 12 + 12) % 12 + 1;

    RETURN LEAST(4, GREATEST(1, CEIL(months_into_fy / 3.0)::INT));
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Reporting Lag
-- ===================================================================
-- Returns the ACTUAL lag in days between the next earnings date and the
-- most recent income-statement report date, along with the deviation from
-- the EXPECTED lag for the given frequency. Using date subtraction (Date Math)
-- returns an integer number of days directly.
CREATE OR REPLACE FUNCTION calculate_reporting_lag(
    next_earnings DATE,
    income_statement_report_date DATE,
    earnings_report_frequency TEXT DEFAULT 'Quarterly'
)
    RETURNS INTEGER AS
$$
SELECT CASE
           WHEN next_earnings IS NULL OR income_statement_report_date IS NULL THEN NULL
           -- Date - Date returns an INTEGER number of days in PostgreSQL.
           -- We compare against the expected reporting lag for the given frequency
           -- to produce the deviation (positive = late, negative = early).
           ELSE (next_earnings - income_statement_report_date)
               - get_expected_reporting_lag_days(earnings_report_frequency)
           END
$$ LANGUAGE SQL IMMUTABLE
                PARALLEL SAFE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Expected Report Date
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_expected_report_date(
    period_end_date DATE,
    earnings_report_frequency TEXT
)
    RETURNS DATE AS
$$
SELECT CASE
           WHEN period_end_date IS NULL THEN NULL
           ELSE (period_end_date
               + make_interval(days => get_expected_reporting_lag_days(earnings_report_frequency)))::DATE
           END
$$ LANGUAGE SQL IMMUTABLE
                PARALLEL SAFE;

-- ===================================================================
-- HELPER FUNCTION: Validate Fiscal Dates
-- ===================================================================
CREATE OR REPLACE FUNCTION validate_fiscal_dates(
    fy_end_date DATE,
    report_date DATE,
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
        RETURN QUERY SELECT 'FY End Date is in the future'::TEXT, 'WARNING'::TEXT;
    END IF;

    -- A report that is MORE THAN a full fiscal year before the FY end
    -- cannot belong to the current fiscal period. Use strict inequality
    -- with make_interval for clarity.
    IF report_date IS NOT NULL AND report_date < (fy_end_date - make_interval(years => 1))::DATE THEN
        RETURN QUERY SELECT 'Report date predates fiscal year'::TEXT, 'ERROR'::TEXT;
    END IF;

    -- Allow at most 1 day of clock skew; anything beyond is a future-date error
    IF report_date IS NOT NULL AND report_date > reference_date + INTERVAL '1 day' THEN
        RETURN QUERY SELECT 'Report date is in the future'::TEXT, 'WARNING'::TEXT;
    END IF;

    -- Idiomatic end-of-month test via DATE_TRUNC
    IF fy_end_date IS NOT NULL
        AND fy_end_date <> (DATE_TRUNC('month', fy_end_date)
            + INTERVAL '1 month - 1 day')::DATE THEN
        RETURN QUERY SELECT 'FY End is not last day of month'::TEXT, 'INFO'::TEXT;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- Importing US Region Data...
-- ===================================================================
-- STAGING TABLE CREATION (DYNAMIC, CSV-DRIVEN)
-- ===================================================================
-- Why dynamic? \copy maps CSV → table BY POSITION, not by header name.
-- Hard-coding the staging columns therefore breaks whenever the source
-- CSV adds, removes, or reorders columns.
--
-- Strategy:
--   1. Read the FIRST CSV's header line into a one-row, one-column buffer.
--   2. Split it on commas (respecting quoted column names) to get the
--      authoritative ordered list of CSV columns.
--   3. Build a TEMP TABLE whose columns are TEXT and exactly match that
--      list (in order). Every subsequent regional file MUST share this
--      header (a precondition we verify below).
--   4. \copy each regional file into the staging table.
-- ===================================================================
\echo 'Discovering CSV header from data/pml/pml_us.csv ...'

DROP TABLE IF EXISTS staging_header_buf;

CREATE TEMP TABLE staging_header_buf
(
    header_line TEXT
);

-- Load ONLY the first line of the US file into the buffer.
-- We disable HEADER so the line is captured verbatim, and use a delimiter
-- that won't appear in a CSV header line (\b = backspace) so the entire
-- line lands in the single column.
\copy staging_header_buf FROM PROGRAM 'powershell -NoProfile -Command "Get-Content -Path data/pml/pml_us.csv -TotalCount 1"' WITH (FORMAT text)

-- Build the staging table from the discovered header.
DO
$$
    DECLARE
        v_header  TEXT;
        v_cols    TEXT[];
        v_col     TEXT;
        v_ddl     TEXT := 'CREATE TEMP TABLE staging_header_buf (';
        v_first   BOOLEAN := TRUE;
    BEGIN
        SELECT header_line INTO v_header FROM staging_header_buf LIMIT 1;
        IF v_header IS NULL THEN
            RAISE EXCEPTION 'Could not read header line from data/pml/pml_us.csv';
        END IF;

        -- Strip a possible UTF-8 BOM
        v_header := regexp_replace(v_header, E'^\uFEFF', '');

        -- Split the header on commas that are NOT inside double quotes.
        -- This regex matches a comma followed by an even number of quotes
        -- to the end of the string -- i.e., a comma outside any quoted run.
        v_cols := regexp_split_to_array(v_header, ',(?=(?:[^"]*"[^"]*")*[^"]*$)');

        FOREACH v_col IN ARRAY v_cols
            LOOP
                -- Trim whitespace and surrounding quotes
                v_col := btrim(v_col);
                IF length(v_col) >= 2
                    AND left(v_col, 1) = '"'
                    AND right(v_col, 1) = '"' THEN
                    v_col := substring(v_col FROM 2 FOR length(v_col) - 2);
                    -- Un-escape doubled quotes ("" -> ")
                    v_col := replace(v_col, '""', '"');
                END IF;

                IF NOT v_first THEN
                    v_ddl := v_ddl || ', ';
                END IF;
                v_first := FALSE;

                -- Quote the column name and declare it TEXT
                v_ddl := v_ddl || quote_ident(v_col) || ' TEXT';
            END LOOP;

        v_ddl := v_ddl || ')';

        EXECUTE 'DROP TABLE IF EXISTS staging_header_buf';
        EXECUTE v_ddl;

        RAISE NOTICE 'Staging table created with % columns.', array_length(v_cols, 1);
    END
$$;

-- ===================================================================
-- DATA IMPORT EXECUTION
-- ===================================================================
-- All four regional files MUST share the same header (same column set,
-- same order) as data/pml/pml_us.csv. If a vendor ever ships a file
-- with a divergent header, we want a hard, immediate failure rather
-- than silent column misalignment -- which is exactly what \copy
-- gives us, since it validates field count per row.

\echo 'Importing US data...'
\copy staging_header_buf FROM 'data/pml/pml_us.csv'   WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description"))

\echo 'Importing EU data...'
\copy staging_header_buf FROM 'data/pml/pml_eu.csv'   WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description"))

\echo 'Importing APAC data...'
\copy staging_header_buf FROM 'data/pml/pml_apac.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description"))

\echo 'Importing ROTW data...'
\copy staging_header_buf FROM 'data/pml/pml_rotw.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description"))

-- ===================================================================
-- DATA VALIDATION (PRE-INSERT)
-- ===================================================================
\echo 'Validating imported data...'
SELECT 'Total rows in staging:' AS info, COUNT(*) AS count
FROM staging_header_buf;

TRUNCATE TABLE pml_df;
INSERT INTO pml_df (
                    "Ticker",
                    "ISIN",
                    "Name",
                    "Description",
                    "Region",
                    "Country",
                    "Trading Country",
                    "Exchange",
                    "Unit",
                    "Sector",
                    "Industry",
                    "Style Class",
                    "Size Class",
                    "Last Updated",
                    "Income Statement Report Date",
                    "FY End",
                    "Next Earnings",
                    "Next Earnings (When)",
                    "Next Earnings (Status)",
                    "Dividend Record (Currency)",
                    "Dividend Record (Amount)",
                    "Dividend Record (Frequency)",
                    "Dividend Streak",
                    "Dividend Record (Announce Date)",
                    "Dividend Record (Payable Date)",
                    "Dividend Record (Record Date)",
                    "Dividend Record (Ex Date)",
                    "Market Cap",
                    "Enterprise Value",
                    "Last Price",
                    "Price Target (YTD Ago)",
                    "Total Return (YTD)",
                    "Price Target",
                    "Price Target - Low",
                    "Price Target - Median",
                    "Price Target - High",
                    "Price Target - #",
                    "P/E (NTM)",
                    "P/E (LTM)",
                    "Altman Z-Score (FY)",
                    "Altman Z-Score (FQ)",
                    "Altman Z-Score (LTM)",
                    "Beta (1Y)",
                    "Beta (2Y)",
                    "Beta (5Y)",
                    "Analyst Rating",
                    "# Strong Sell Ratings",
                    "# Strong Buys Ratings",
                    "# Hold Ratings",
                    "# Buys Ratings",
                    "# Sell Ratings",
                    "# No Opinion Ratings",
                    "Market Cap (Country R)",
                    "Tot. Return %/CAGR (3Y)",
                    "Tot. Return %/CAGR (10Y)",
                    "Total Return (5Y)",
                    "Total Return (10Y)",
                    "Volume (Shrs)",
                    "Dividend Per Share (LTM)",
                    "Div Yield (Ind)",
                    "Div Yield (LTM)",
                    "Gross Profit Margin % (FY)",
                    "Gross Profit Margin % (LTM)",
                    "EPS Norm - Est Avg (NTM)",
                    "EPS/Adj. (-1FY)",
                    "EPS/Adj. (FY)",
                    "EPS/Adj. (LTM)",
                    "EPS Norm - Est Avg (FY1E)",
                    "Buyback Yield (LTM)",
                    "Return on Assets (ROA) % (LTM)",
                    "Return on Assets (ROA) % (FY)",
                    "Div Yield (-1FYInd)",
                    "P/B (LTM)",
                    "P/B (-1FY)",
                    "P/B (5YAVG)",
                    "Div Yield (TTM)",
                    "Div Yield (NTM)",
                    "Div Yield (5YAVGLTM)",
                    "Price Chg. % (3M)",
                    "1-Day %",
                    "Price (5D Ago)",
                    "Price (1W Ago)",
                    "Price (1M Ago)",
                    "Price (3M Ago)",
                    "Price (6M Ago)",
                    "Price (1Y Ago)",
                    "Price (3Y Ago)",
                    "Price (5Y Ago)",
                    "Price (QTD Ago)",
                    "Rel. Volume",
                    "Shrs Out",
                    "Shrs Out (-1FY)",
                    "Common Dividends Paid (LTM)",
                    "Common Dividends Paid (FY)",
                    "EV/Sales (LTM)",
                    "EV/Sales (NTM)",
                    "EV/Sales (-1FYLTM)",
                    "EV/Sales (-2FYLTM)",
                    "EV/Sales (-3FYLTM)",
                    "EV/Sales (3YAVGLTM)",
                    "EV/Sales (-1FQLTM)",
                    "EV/Sales (-2FQLTM)",
                    "EV/Sales (-3FQLTM)",
                    "EV/Sales (-4FQLTM)",
                    "52W High/Adj",
                    "52W Low/Adj",
                    "EMA (20D)",
                    "EMA (50D)",
                    "EMA (100D)",
                    "EMA (250D)",
                    "EV/EBITDA (LTM)",
                    "EV/EBITDA (NTM)",
                    "EV/EBITDA (-1FYLTM)",
                    "EV/EBITDA (-1FQLTM)",
                    "EV/EBITDA (3YAVGLTM)",
                    "EV/EBITDA (EST FY1)",
                    "P/E (EST FY1)",
                    "P/E (-1FYLTM)",
                    "P/E (-2FYLTM)",
                    "P/E (-3FYLTM)",
                    "P/E (3YAVGLTM)",
                    "P/E (-1FQLTM)",
                    "P/E (-2FQLTM)",
                    "P/E (-3FQLTM)",
                    "P/E (5YAVGLTM)",
                    "P/E (-0FQQoQLTM)",
                    "P/E (-0FYYoYLTM)",
                    "P/E (-1FYYoYLTM)",
                    "P/E (-0FQYoYLTM)",
                    "Full Time Employees (FQ)",
                    "Full Time Employees (FY)",
                    "Full Time Employees (-1FY)",
                    "Full Time Employees (-2FY)",
                    "Full Time Employees (-3FY)",
                    "Avg Employees (5YAVGFY)",
                    "Net EPS - Basic (LTM)",
                    "Net EPS - Basic (FQ)",
                    "Net EPS - Basic (FY)",
                    "Net EPS - Basic (-1FQFQ)",
                    "Net EPS - Basic (-2FQFQ)",
                    "Net EPS - Basic (-3FQFQ)",
                    "Net EPS - Basic (-4FQFQ)",
                    "Net EPS - Basic (-1FY)",
                    "Net EPS - Basic (-2FY)",
                    "Net EPS - Basic (-3FY)",
                    "Net EPS - Basic (-4FY)",
                    "Net EPS - Basic (-5FY)",
                    "EPS Est Avg Rev % (FY1E - 1W)",
                    "EPS Est Avg Rev % (FY1E - 1M)",
                    "EPS Est Avg Rev % (FY1E - 3M)",
                    "EPS Est Avg Rev % (FY1E - 6M)",
                    "EPS Est Avg Rev % (FY1E - 1Y)",
                    "Div Yield (-2FYInd)",
                    "Div Yield (-3FYInd)",
                    "Div Yield (-4FYInd)",
                    "Div Yield (-5FYInd)",
                    "EBITDA - Est Avg (NTM)",
                    "EBITDA - Est Avg (FY1E)",
                    "EPS GAAP - Est Avg (NTM)",
                    "EPS GAAP - Est Avg (FY1E)",
                    "EPS GAAP Est Avg Rev % (FY1E - 1M)",
                    "EPS GAAP Est Avg Rev % (FY1E - 3M)",
                    "EPS GAAP Est Avg Rev % (FY1E - 6M)",
                    "EPS GAAP Est Avg Rev % (FY1E - 1Y)",
                    "EPS Norm - Est # (FY1E)",
                    "Price Target (1W Ago)",
                    "Price Target (1M Ago)",
                    "Price Target (3M Ago)",
                    "Price Target (6M Ago)",
                    "Price Target (MTD Ago)",
                    "Price Target (QTD Ago)",
                    "Price Target (1Y Ago)",
                    "Price Target - # (3M Ago)",
                    "Price Target - # (6M Ago)",
                    "Price Target - # (YTD Ago)",
                    "Price Target - # (1Y Ago)",
                    "Price Target - # (1W Ago)",
                    "Price Target - # (1M Ago)",
                    "Price Target - # (MTD Ago)",
                    "Price Target - # (QTD Ago)",
                    "Price Target - High (1W Ago)",
                    "Price Target - High (1M Ago)",
                    "Price Target - High (6M Ago)",
                    "Price Target - High (MTD Ago)",
                    "Price Target - High (3M Ago)",
                    "Price Target - High (QTD Ago)",
                    "Price Target - High (1Y Ago)",
                    "Price Target - High (YTD Ago)",
                    "Price Target - Low (1W Ago)",
                    "Price Target - Low (1M Ago)",
                    "Price Target - Low (3M Ago)",
                    "Price Target - Low (6M Ago)",
                    "Price Target - Low (MTD Ago)",
                    "Price Target - Low (QTD Ago)",
                    "Price Target - Low (YTD Ago)",
                    "Price Target - Low (1Y Ago)",
                    "Price Target - Median (1W Ago)",
                    "Price Target - Median (1M Ago)",
                    "Price Target - Median (3M Ago)",
                    "Price Target - Median (6M Ago)",
                    "Price Target - Median (MTD Ago)",
                    "Price Target - Median (QTD Ago)",
                    "Price Target - Median (YTD Ago)",
                    "Price Target - Median (1Y Ago)",
                    "Basic EPS - Cont (LTM)",
                    "Basic EPS - Cont (FQ)",
                    "Basic EPS - Cont (FY)",
                    "Basic EPS - Cont (-1FQFQ)",
                    "Basic EPS - Cont (-2FQFQ)",
                    "Basic EPS - Cont (-4FQFQ)",
                    "Basic EPS - Cont (-3FQFQ)",
                    "Basic EPS - Cont (-1FY)",
                    "Basic EPS - Cont (-2FY)",
                    "Basic EPS - Cont (-3FY)",
                    "Basic EPS - Cont (-4FY)",
                    "EPS/Adj. (FQ)",
                    "EPS/Adj. (-1FQFQ)",
                    "EPS/Adj. (-3FQFQ)",
                    "EPS/Adj. (-4FQFQ)",
                    "EPS/Adj. (-2FQFQ)",
                    "EPS/Adj. (-2FY)",
                    "EPS/Adj. (-3FY)",
                    "EPS/Adj. (-4FY)",
                    "Gross Profit (-1FQFQ)",
                    "Gross Profit (-3FQFQ)",
                    "Gross Profit (-4FQFQ)",
                    "Gross Profit (-2FQFQ)",
                    "Gross Profit (-1FY)",
                    "Gross Profit (-2FY)",
                    "Gross Profit (-3FY)",
                    "Gross Profit (-4FY)",
                    "FCF - Est Avg (FY1E)",
                    "FCF - Est Avg (FY2E)",
                    "FCF - Est Avg (FY3E)",
                    "FCF - Est Avg (FY4E)",
                    "FCF - Est Avg (FY5E)",
                    "EPS (-0FYEstimate)",
                    "EPS (-0FYActual)",
                    "EPS (-0FYSurprise %)",
                    "EPS (-1FYEstimate)",
                    "EPS (-1FYActual)",
                    "EPS (-1FYSurprise %)",
                    "EPS (-2FYEstimate)",
                    "EPS (-2FYActual)",
                    "EPS (-2FYSurprise %)",
                    "EPS (-3FYEstimate)",
                    "EPS (-3FYActual)",
                    "EPS (-3FYSurprise %)",
                    "EPS (-4FYActual)",
                    "EPS (-4FYEstimate)",
                    "EPS (-4FYSurprise %)",
                    "EPS (-5FYEstimate)",
                    "EPS (-5FYActual)",
                    "EPS (-5FYSurprise %)",
                    "EPS (-0FQEstimate)",
                    "EPS (-0FQActual)",
                    "EPS (-0FQSurprise %)",
                    "EPS (-1FQEstimate)",
                    "EPS (-1FQActual)",
                    "EPS (-1FQSurprise %)",
                    "EPS (-2FQEstimate)",
                    "EPS (-2FQActual)",
                    "EPS (-2FQSurprise %)",
                    "EPS (-3FQEstimate)",
                    "EPS (-3FQActual)",
                    "EPS (-3FQSurprise %)",
                    "EPS (-4FQEstimate)",
                    "EPS (-4FQActual)",
                    "EPS (-4FQSurprise %)",
                    "FCF (LTM)",
                    "FCF (FQ)",
                    "FCF (-1FQFQ)",
                    "FCF (-3FQFQ)",
                    "FCF (-4FQFQ)",
                    "FCF (-2FQFQ)",
                    "FCF (FY)",
                    "FCF (-1FY)",
                    "FCF (-3FY)",
                    "FCF (-2FY)",
                    "FCF (-4FY)",
                    "FCF (-5FY)",
                    "Target % (Avg)",
                    "Target % (Med)",
                    "Target % (Low)",
                    "Target % (High)",
                    "Price Target - StdDev",
                    "Price Target - StdDev (1W Ago)",
                    "Price Target - StdDev (1M Ago)",
                    "Price Target - StdDev (3M Ago)",
                    "Price Target - StdDev (6M Ago)",
                    "Price Target - StdDev (1Y Ago)",
                    "Altman Z-Score (-1FY)",
                    "Altman Z-Score (-2FY)",
                    "Altman Z-Score (-3FY)",
                    "Altman Z-Score (-4FY)",
                    "Altman Z-Score (-5FY)",
                    "Altman Z-Score (-1FQFQ)",
                    "Altman Z-Score (-2FQFQ)",
                    "Altman Z-Score (-3FQFQ)",
                    "Altman Z-Score (-4FQFQ)",
                    "Altman Z-Score (-0FYYoYLTM)",
                    "Altman Z-Score (-1FYYoYLTM)",
                    "Altman Z-Score (-3FYYoYLTM)",
                    "Altman Z-Score (-4FYYoYLTM)",
                    "Altman Z-Score (-5FYYoYLTM)",
                    "Altman Z-Score (-2FYYoYLTM)",
                    "P/E (EST FY2)",
                    "P/E (EST FY3)",
                    "P/E (EST FY4)",
                    "P/E (EST FY5)",
                    "P/E (-4FYLTM)",
                    "P/E (-4FQLTM)",
                    "P/E (3YAVGNTM)",
                    "P/E (5YAVGNTM)",
                    "EPS Norm - Est Avg (FQ1E)",
                    "EPS Norm - Est Avg (FQ2E)",
                    "EPS Norm - Est Avg (FQ3E)",
                    "EPS Norm - Est Avg (FQ4E)",
                    "EPS Norm - Est Avg (FY2E)",
                    "EPS Norm - Est Avg (FY3E)",
                    "EPS Norm - Est Avg (FY4E)",
                    "EPS Norm - Est Avg (FY5E)",
                    "Capital Expenditure (LTM)",
                    "Capital Expenditure (FQ)",
                    "Capital Expenditure (FY)",
                    "Capital Expenditure (-1FQFQ)",
                    "Capital Expenditure (-2FQFQ)",
                    "Capital Expenditure (-3FQFQ)",
                    "Capital Expenditure (-4FQFQ)",
                    "Capital Expenditure (-1FY)",
                    "Capital Expenditure (-2FY)",
                    "Capital Expenditure (-4FY)",
                    "Capital Expenditure (-3FY)",
                    "Capital Expenditure (-5FY)",
                    "CFF (LTM)",
                    "CFF (FQ)",
                    "CFF (FY)",
                    "CFF (-1FQFQ)",
                    "CFF (-2FQFQ)",
                    "CFF (-3FQFQ)",
                    "CFF (-4FQFQ)",
                    "CFF (-1FY)",
                    "CFF (-2FY)",
                    "CFF (-3FY)",
                    "CFF (-4FY)",
                    "CFI (LTM)",
                    "CFI (FQ)",
                    "CFI (FY)",
                    "CFI (-1FQFQ)",
                    "CFI (-2FQFQ)",
                    "CFI (-3FQFQ)",
                    "CFI (-4FQFQ)",
                    "CFI (-1FY)",
                    "CFI (-2FY)",
                    "CFI (-3FY)",
                    "CFI (-5FY)",
                    "CFI (-4FY)",
                    "CFO (LTM)",
                    "CFO (FQ)",
                    "CFO (FY)",
                    "CFO (-1FQFQ)",
                    "CFO (-2FQFQ)",
                    "CFO (-4FQFQ)",
                    "CFO (-3FQFQ)",
                    "CFO (-1FY)",
                    "CFO (-2FY)",
                    "CFO (-3FY)",
                    "CFO (-4FY)",
                    "CFO (-5FY)",
                    "Dividend Per Share (FQ)",
                    "Dividend Per Share (FY)",
                    "Dividend Per Share (-1FQFQ)",
                    "Dividend Per Share (-2FQFQ)",
                    "Dividend Per Share (-3FQFQ)",
                    "Dividend Per Share (-4FQFQ)",
                    "Dividend Per Share (-1FY)",
                    "Dividend Per Share (-2FY)",
                    "Dividend Per Share (-3FY)",
                    "Dividend Per Share (-4FY)",
                    "Dividend Per Share (-5FY)",
                    "Enterprise Value (-1FQ)",
                    "Enterprise Value (-2FQ)",
                    "Enterprise Value (-3FQ)",
                    "Enterprise Value (-4FQ)",
                    "Enterprise Value (-1FY)",
                    "Enterprise Value (-2FY)",
                    "Enterprise Value (-3FY)",
                    "Enterprise Value (-4FY)",
                    "Enterprise Value (-5FY)",
                    "Volatility (1M)",
                    "Volatility (3M)",
                    "Volatility (6M)",
                    "Volatility (1Y)",
                    "EPS GAAP Est Avg Rev % (FY1E - 1W)",
                    "EPS GAAP Est Avg Rev % (FY1E - MTD)",
                    "EPS GAAP Est Avg Rev % (FY1E - QTD)",
                    "EPS GAAP Est Avg Rev % (FY1E - YTD)",
                    "Price Target - StdDev (MTD Ago)",
                    "Price Target - StdDev (QTD Ago)",
                    "Price Target - StdDev (YTD Ago)")
SELECT
       NULLIF(TRIM(s."Ticker"), '')                                       AS "Ticker",
       NULLIF(TRIM(s."ISIN"), '')                                         AS "ISIN",
       NULLIF(TRIM(s."Name"), '')                                         AS "Name",
       NULLIF(TRIM(s."Description"), '')                                  AS "Description",
       COALESCE(NULLIF(TRIM(s."Region"), ''), 'n/a')                      AS "Region",
       COALESCE(NULLIF(TRIM(s."Country"), ''), 'n/a')                     AS "Country",
       COALESCE(NULLIF(TRIM(s."Trading Country"), ''), 'n/a')             AS "Trading Country",
       COALESCE(NULLIF(TRIM(s."Exchange"), ''), 'n/a')                    AS "Exchange",
       COALESCE(NULLIF(TRIM(s."Unit"), ''), 'n/a')                        AS "Unit",
       COALESCE(NULLIF(TRIM(s."Sector"), ''), 'n/a')                      AS "Sector",
       COALESCE(NULLIF(TRIM(s."Industry"), ''), 'n/a')                    AS "Industry",
       COALESCE(NULLIF(TRIM(s."Style Class"), ''), 'n/a')                 AS "Style Class",
       COALESCE(NULLIF(TRIM(s."Size Class"), ''), 'n/a')                  AS "Size Class",
       text_to_date_safe(s."Last Updated")                                AS "Last Updated",
       text_to_date_safe(s."Income Statement Report Date")                AS "Income Statement Report Date",
       COALESCE(NULLIF(TRIM(s."FY End"), ''), 'n/a')                      AS "FY End",
       text_to_date_safe(s."Next Earnings")                               AS "Next Earnings",
       COALESCE(NULLIF(TRIM(s."Next Earnings (When)"), ''), 'n/a')        AS "Next Earnings (When)",
       COALESCE(NULLIF(TRIM(s."Next Earnings (Status)"), ''), 'n/a')      AS "Next Earnings (Status)",
       COALESCE(NULLIF(TRIM(s."Dividend Record (Currency)"), ''), 'n/a')  AS "Dividend Record (Currency)",
       text_to_numeric_safe(s."Dividend Record (Amount)")                 AS "Dividend Record (Amount)",
       COALESCE(NULLIF(TRIM(s."Dividend Record (Frequency)"), ''), 'n/a') AS "Dividend Record (Frequency)",
       text_to_numeric_safe(s."Dividend Streak")                          AS "Dividend Streak",
       text_to_date_safe(s."Dividend Record (Announce Date)")             AS "Dividend Record (Announce Date)",
       text_to_date_safe(s."Dividend Record (Payable Date)")              AS "Dividend Record (Payable Date)",
       text_to_date_safe(s."Dividend Record (Record Date)")               AS "Dividend Record (Record Date)",
       text_to_date_safe(s."Dividend Record (Ex Date)")                   AS "Dividend Record (Ex Date)",
       text_to_numeric_safe(s."Market Cap")                               AS "Market Cap",
       text_to_numeric_safe(s."Enterprise Value")                         AS "Enterprise Value",
       text_to_numeric_safe(s."Last Price")                               AS "Last Price",
       text_to_numeric_safe(s."Price Target (YTD Ago)")                   AS "Price Target (YTD Ago)",
       text_to_numeric_safe(s."Total Return (YTD)")                       AS "Total Return (YTD)",
       text_to_numeric_safe(s."Price Target")                             AS "Price Target",
       text_to_numeric_safe(s."Price Target - Low")                       AS "Price Target - Low",
       text_to_numeric_safe(s."Price Target - Median")                    AS "Price Target - Median",
       text_to_numeric_safe(s."Price Target - High")                      AS "Price Target - High",
       text_to_numeric_safe(s."Price Target - #")                         AS "Price Target - #",
       text_to_numeric_safe(s."P/E (NTM)")                                AS "P/E (NTM)",
       text_to_numeric_safe(s."P/E (LTM)")                                AS "P/E (LTM)",
       text_to_numeric_safe(s."Altman Z-Score (FY)")                      AS "Altman Z-Score (FY)",
       text_to_numeric_safe(s."Altman Z-Score (FQ)")                      AS "Altman Z-Score (FQ)",
       text_to_numeric_safe(s."Altman Z-Score (LTM)")                     AS "Altman Z-Score (LTM)",
       text_to_numeric_safe(s."Beta (1Y)")                                AS "Beta (1Y)",
       text_to_numeric_safe(s."Beta (2Y)")                                AS "Beta (2Y)",
       text_to_numeric_safe(s."Beta (5Y)")                                AS "Beta (5Y)",
       text_to_numeric_safe(s."Analyst Rating")                           AS "Analyst Rating",
       COALESCE(text_to_numeric_safe(s."# Strong Sell Ratings"), 0)::INT  AS "# Strong Sell Ratings",
       COALESCE(text_to_numeric_safe(s."# Strong Buys Ratings"), 0)::INT  AS "# Strong Buys Ratings",
       COALESCE(text_to_numeric_safe(s."# Hold Ratings"), 0)::INT         AS "# Hold Ratings",
       COALESCE(text_to_numeric_safe(s."# Buys Ratings"), 0)::INT         AS "# Buys Ratings",
       COALESCE(text_to_numeric_safe(s."# Sell Ratings"), 0)::INT         AS "# Sell Ratings",
       COALESCE(text_to_numeric_safe(s."# No Opinion Ratings"), 0)::INT   AS "# No Opinion Ratings",
       COALESCE(text_to_numeric_safe(s."Market Cap (Country R)"), 0)::INT AS "Market Cap (Country R)",
       text_to_numeric_safe(s."Tot. Return %/CAGR (3Y)")                  AS "Tot. Return %/CAGR (3Y)",
       text_to_numeric_safe(s."Tot. Return %/CAGR (10Y)")                 AS "Tot. Return %/CAGR (10Y)",
       text_to_numeric_safe(s."Total Return (5Y)")                        AS "Total Return (5Y)",
       text_to_numeric_safe(s."Total Return (10Y)")                       AS "Total Return (10Y)",
       text_to_numeric_safe(s."Volume (Shrs)")                            AS "Volume (Shrs)",
       text_to_numeric_safe(s."Dividend Per Share (LTM)")                 AS "Dividend Per Share (LTM)",
       text_to_numeric_safe(s."Div Yield (Ind)")                          AS "Div Yield (Ind)",
       text_to_numeric_safe(s."Div Yield (LTM)")                          AS "Div Yield (LTM)",
       text_to_numeric_safe(s."Gross Profit Margin % (FY)")               AS "Gross Profit Margin % (FY)",
       text_to_numeric_safe(s."Gross Profit Margin % (LTM)")              AS "Gross Profit Margin % (LTM)",
       text_to_numeric_safe(s."EPS Norm - Est Avg (NTM)")                 AS "EPS Norm - Est Avg (NTM)",
       text_to_numeric_safe(s."EPS/Adj. (-1FY)")                          AS "EPS/Adj. (-1FY)",
       text_to_numeric_safe(s."EPS/Adj. (FY)")                            AS "EPS/Adj. (FY)",
       text_to_numeric_safe(s."EPS/Adj. (LTM)")                           AS "EPS/Adj. (LTM)",
       text_to_numeric_safe(s."EPS Norm - Est Avg (FY1E)")                AS "EPS Norm - Est Avg (FY1E)",
       text_to_numeric_safe(s."Buyback Yield (LTM)")                      AS "Buyback Yield (LTM)",
       text_to_numeric_safe(s."Return on Assets (ROA) % (LTM)")           AS "Return on Assets (ROA) % (LTM)",
       text_to_numeric_safe(s."Return on Assets (ROA) % (FY)")            AS "Return on Assets (ROA) % (FY)",
       text_to_numeric_safe(s."Div Yield (-1FYInd)")                      AS "Div Yield (-1FYInd)",
       text_to_numeric_safe(s."P/B (LTM)")                                AS "P/B (LTM)",
       text_to_numeric_safe(s."P/B (-1FY)")                               AS "P/B (-1FY)",
       text_to_numeric_safe(s."P/B (5YAVG)")                              AS "P/B (5YAVG)",
       text_to_numeric_safe(s."Div Yield (TTM)")                          AS "Div Yield (TTM)",
       text_to_numeric_safe(s."Div Yield (NTM)")                          AS "Div Yield (NTM)",
       text_to_numeric_safe(s."Div Yield (5YAVGLTM)")                     AS "Div Yield (5YAVGLTM)",
       text_to_numeric_safe(s."Price Chg. % (3M)")                        AS "Price Chg. % (3M)",
       text_to_numeric_safe(s."1-Day %")                                  AS "1-Day %",
       text_to_numeric_safe(s."Price (5D Ago)")                           AS "Price (5D Ago)",
       text_to_numeric_safe(s."Price (1W Ago)")                           AS "Price (1W Ago)",
       text_to_numeric_safe(s."Price (1M Ago)")                           AS "Price (1M Ago)",
       text_to_numeric_safe(s."Price (3M Ago)")                           AS "Price (3M Ago)",
       text_to_numeric_safe(s."Price (6M Ago)")                           AS "Price (6M Ago)",
       text_to_numeric_safe(s."Price (1Y Ago)")                           AS "Price (1Y Ago)",
       text_to_numeric_safe(s."Price (3Y Ago)")                           AS "Price (3Y Ago)",
       text_to_numeric_safe(s."Price (5Y Ago)")                           AS "Price (5Y Ago)",
       text_to_numeric_safe(s."Price (QTD Ago)")                          AS "Price (QTD Ago)",
       text_to_numeric_safe(s."Rel. Volume")                              AS "Rel. Volume",
       text_to_numeric_safe(s."Shrs Out")                                 AS "Shrs Out",
       text_to_numeric_safe(s."Shrs Out (-1FY)")                          AS "Shrs Out (-1FY)",
       text_to_numeric_safe(s."Common Dividends Paid (LTM)")              AS "Common Dividends Paid (LTM)",
       text_to_numeric_safe(s."Common Dividends Paid (FY)")               AS "Common Dividends Paid (FY)",
       text_to_numeric_safe(s."EV/Sales (LTM)")                           AS "EV/Sales (LTM)",
       text_to_numeric_safe(s."EV/Sales (NTM)")                           AS "EV/Sales (NTM)",
       text_to_numeric_safe(s."EV/Sales (-1FYLTM)")                       AS "EV/Sales (-1FYLTM)",
       text_to_numeric_safe(s."EV/Sales (-2FYLTM)")                       AS "EV/Sales (-2FYLTM)",
       text_to_numeric_safe(s."EV/Sales (-3FYLTM)")                       AS "EV/Sales (-3FYLTM)",
       text_to_numeric_safe(s."EV/Sales (3YAVGLTM)")                      AS "EV/Sales (3YAVGLTM)",
       text_to_numeric_safe(s."EV/Sales (-1FQLTM)")                       AS "EV/Sales (-1FQLTM)",
       text_to_numeric_safe(s."EV/Sales (-2FQLTM)")                       AS "EV/Sales (-2FQLTM)",
       text_to_numeric_safe(s."EV/Sales (-3FQLTM)")                       AS "EV/Sales (-3FQLTM)",
       text_to_numeric_safe(s."EV/Sales (-4FQLTM)")                       AS "EV/Sales (-4FQLTM)",
       text_to_numeric_safe(s."52W High/Adj")                             AS "52W High/Adj",
       text_to_numeric_safe(s."52W Low/Adj")                              AS "52W Low/Adj",
       text_to_numeric_safe(s."EMA (20D)")                                AS "EMA (20D)",
       text_to_numeric_safe(s."EMA (50D)")                                AS "EMA (50D)",
       text_to_numeric_safe(s."EMA (100D)")                               AS "EMA (100D)",
       text_to_numeric_safe(s."EMA (250D)")                               AS "EMA (250D)",
       text_to_numeric_safe(s."EV/EBITDA (LTM)")                          AS "EV/EBITDA (LTM)",
       text_to_numeric_safe(s."EV/EBITDA (NTM)")                          AS "EV/EBITDA (NTM)",
       text_to_numeric_safe(s."EV/EBITDA (-1FYLTM)")                      AS "EV/EBITDA (-1FYLTM)",
       text_to_numeric_safe(s."EV/EBITDA (-1FQLTM)")                      AS "EV/EBITDA (-1FQLTM)",
       text_to_numeric_safe(s."EV/EBITDA (3YAVGLTM)")                     AS "EV/EBITDA (3YAVGLTM)",
       text_to_numeric_safe(s."EV/EBITDA (EST FY1)")                      AS "EV/EBITDA (EST FY1)",
       text_to_numeric_safe(s."P/E (EST FY1)")                            AS "P/E (EST FY1)",
       text_to_numeric_safe(s."P/E (-1FYLTM)")                            AS "P/E (-1FYLTM)",
       text_to_numeric_safe(s."P/E (-2FYLTM)")                            AS "P/E (-2FYLTM)",
       text_to_numeric_safe(s."P/E (-3FYLTM)")                            AS "P/E (-3FYLTM)",
       text_to_numeric_safe(s."P/E (3YAVGLTM)")                           AS "P/E (3YAVGLTM)",
       text_to_numeric_safe(s."P/E (-1FQLTM)")                            AS "P/E (-1FQLTM)",
       text_to_numeric_safe(s."P/E (-2FQLTM)")                            AS "P/E (-2FQLTM)",
       text_to_numeric_safe(s."P/E (-3FQLTM)")                            AS "P/E (-3FQLTM)",
       text_to_numeric_safe(s."P/E (5YAVGLTM)")                           AS "P/E (5YAVGLTM)",
       text_to_numeric_safe(s."P/E (-0FQQoQLTM)")                         AS "P/E (-0FQQoQLTM)",
       text_to_numeric_safe(s."P/E (-0FYYoYLTM)")                         AS "P/E (-0FYYoYLTM)",
       text_to_numeric_safe(s."P/E (-1FYYoYLTM)")                         AS "P/E (-1FYYoYLTM)",
       text_to_numeric_safe(s."P/E (-0FQYoYLTM)")                         AS "P/E (-0FQYoYLTM)",
       text_to_numeric_safe(s."Full Time Employees (FQ)")                 AS "Full Time Employees (FQ)",
       text_to_numeric_safe(s."Full Time Employees (FY)")                 AS "Full Time Employees (FY)",
       text_to_numeric_safe(s."Full Time Employees (-1FY)")               AS "Full Time Employees (-1FY)",
       text_to_numeric_safe(s."Full Time Employees (-2FY)")               AS "Full Time Employees (-2FY)",
       text_to_numeric_safe(s."Full Time Employees (-3FY)")               AS "Full Time Employees (-3FY)",
       text_to_numeric_safe(s."Avg Employees (5YAVGFY)")                  AS "Avg Employees (5YAVGFY)",
       text_to_numeric_safe(s."Net EPS - Basic (LTM)")                    AS "Net EPS - Basic (LTM)",
       text_to_numeric_safe(s."Net EPS - Basic (FQ)")                     AS "Net EPS - Basic (FQ)",
       text_to_numeric_safe(s."Net EPS - Basic (FY)")                     AS "Net EPS - Basic (FY)",
       text_to_numeric_safe(s."Net EPS - Basic (-1FQFQ)")                 AS "Net EPS - Basic (-1FQFQ)",
       text_to_numeric_safe(s."Net EPS - Basic (-2FQFQ)")                 AS "Net EPS - Basic (-2FQFQ)",
       text_to_numeric_safe(s."Net EPS - Basic (-3FQFQ)")                 AS "Net EPS - Basic (-3FQFQ)",
       text_to_numeric_safe(s."Net EPS - Basic (-4FQFQ)")                 AS "Net EPS - Basic (-4FQFQ)",
       text_to_numeric_safe(s."Net EPS - Basic (-1FY)")                   AS "Net EPS - Basic (-1FY)",
       text_to_numeric_safe(s."Net EPS - Basic (-2FY)")                   AS "Net EPS - Basic (-2FY)",
       text_to_numeric_safe(s."Net EPS - Basic (-3FY)")                   AS "Net EPS - Basic (-3FY)",
       text_to_numeric_safe(s."Net EPS - Basic (-4FY)")                   AS "Net EPS - Basic (-4FY)",
       text_to_numeric_safe(s."Net EPS - Basic (-5FY)")                   AS "Net EPS - Basic (-5FY)",
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 1W)")            AS "EPS Est Avg Rev % (FY1E - 1W)",
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 1M)")            AS "EPS Est Avg Rev % (FY1E - 1M)",
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 3M)")            AS "EPS Est Avg Rev % (FY1E - 3M)",
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 6M)")            AS "EPS Est Avg Rev % (FY1E - 6M)",
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 1Y)")            AS "EPS Est Avg Rev % (FY1E - 1Y)",
       text_to_numeric_safe(s."Div Yield (-2FYInd)")                      AS "Div Yield (-2FYInd)",
       text_to_numeric_safe(s."Div Yield (-3FYInd)")                      AS "Div Yield (-3FYInd)",
       text_to_numeric_safe(s."Div Yield (-4FYInd)")                      AS "Div Yield (-4FYInd)",
       text_to_numeric_safe(s."Div Yield (-5FYInd)")                      AS "Div Yield (-5FYInd)",
       text_to_numeric_safe(s."EBITDA - Est Avg (NTM)")                   AS "EBITDA - Est Avg (NTM)",
       text_to_numeric_safe(s."EBITDA - Est Avg (FY1E)")                  AS "EBITDA - Est Avg (FY1E)",
       text_to_numeric_safe(s."EPS GAAP - Est Avg (NTM)")                 AS "EPS GAAP - Est Avg (NTM)",
       text_to_numeric_safe(s."EPS GAAP - Est Avg (FY1E)")                AS "EPS GAAP - Est Avg (FY1E)",
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 1M)")       AS "EPS GAAP Est Avg Rev % (FY1E - 1M)",
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 3M)")       AS "EPS GAAP Est Avg Rev % (FY1E - 3M)",
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 6M)")       AS "EPS GAAP Est Avg Rev % (FY1E - 6M)",
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 1Y)")       AS "EPS GAAP Est Avg Rev % (FY1E - 1Y)",
       text_to_numeric_safe(s."EPS Norm - Est # (FY1E)")                  AS "EPS Norm - Est # (FY1E)",
       text_to_numeric_safe(s."Price Target (1W Ago)")                    AS "Price Target (1W Ago)",
       text_to_numeric_safe(s."Price Target (1M Ago)")                    AS "Price Target (1M Ago)",
       text_to_numeric_safe(s."Price Target (3M Ago)")                    AS "Price Target (3M Ago)",
       text_to_numeric_safe(s."Price Target (6M Ago)")                    AS "Price Target (6M Ago)",
       text_to_numeric_safe(s."Price Target (MTD Ago)")                   AS "Price Target (MTD Ago)",
       text_to_numeric_safe(s."Price Target (QTD Ago)")                   AS "Price Target (QTD Ago)",
       text_to_numeric_safe(s."Price Target (1Y Ago)")                    AS "Price Target (1Y Ago)",
       text_to_numeric_safe(s."Price Target - # (3M Ago)")                AS "Price Target - # (3M Ago)",
       text_to_numeric_safe(s."Price Target - # (6M Ago)")                AS "Price Target - # (6M Ago)",
       text_to_numeric_safe(s."Price Target - # (YTD Ago)")               AS "Price Target - # (YTD Ago)",
       text_to_numeric_safe(s."Price Target - # (1Y Ago)")                AS "Price Target - # (1Y Ago)",
       text_to_numeric_safe(s."Price Target - # (1W Ago)")                AS "Price Target - # (1W Ago)",
       text_to_numeric_safe(s."Price Target - # (1M Ago)")                AS "Price Target - # (1M Ago)",
       text_to_numeric_safe(s."Price Target - # (MTD Ago)")               AS "Price Target - # (MTD Ago)",
       text_to_numeric_safe(s."Price Target - # (QTD Ago)")               AS "Price Target - # (QTD Ago)",
       text_to_numeric_safe(s."Price Target - High (1W Ago)")             AS "Price Target - High (1W Ago)",
       text_to_numeric_safe(s."Price Target - High (1M Ago)")             AS "Price Target - High (1M Ago)",
       text_to_numeric_safe(s."Price Target - High (6M Ago)")             AS "Price Target - High (6M Ago)",
       text_to_numeric_safe(s."Price Target - High (MTD Ago)")            AS "Price Target - High (MTD Ago)",
       text_to_numeric_safe(s."Price Target - High (3M Ago)")             AS "Price Target - High (3M Ago)",
       text_to_numeric_safe(s."Price Target - High (QTD Ago)")            AS "Price Target - High (QTD Ago)",
       text_to_numeric_safe(s."Price Target - High (1Y Ago)")             AS "Price Target - High (1Y Ago)",
       text_to_numeric_safe(s."Price Target - High (YTD Ago)")            AS "Price Target - High (YTD Ago)",
       text_to_numeric_safe(s."Price Target - Low (1W Ago)")              AS "Price Target - Low (1W Ago)",
       text_to_numeric_safe(s."Price Target - Low (1M Ago)")              AS "Price Target - Low (1M Ago)",
       text_to_numeric_safe(s."Price Target - Low (3M Ago)")              AS "Price Target - Low (3M Ago)",
       text_to_numeric_safe(s."Price Target - Low (6M Ago)")              AS "Price Target - Low (6M Ago)",
       text_to_numeric_safe(s."Price Target - Low (MTD Ago)")             AS "Price Target - Low (MTD Ago)",
       text_to_numeric_safe(s."Price Target - Low (QTD Ago)")             AS "Price Target - Low (QTD Ago)",
       text_to_numeric_safe(s."Price Target - Low (YTD Ago)")             AS "Price Target - Low (YTD Ago)",
       text_to_numeric_safe(s."Price Target - Low (1Y Ago)")              AS "Price Target - Low (1Y Ago)",
       text_to_numeric_safe(s."Price Target - Median (1W Ago)")           AS "Price Target - Median (1W Ago)",
       text_to_numeric_safe(s."Price Target - Median (1M Ago)")           AS "Price Target - Median (1M Ago)",
       text_to_numeric_safe(s."Price Target - Median (3M Ago)")           AS "Price Target - Median (3M Ago)",
       text_to_numeric_safe(s."Price Target - Median (6M Ago)")           AS "Price Target - Median (6M Ago)",
       text_to_numeric_safe(s."Price Target - Median (MTD Ago)")          AS "Price Target - Median (MTD Ago)",
       text_to_numeric_safe(s."Price Target - Median (QTD Ago)")          AS "Price Target - Median (QTD Ago)",
       text_to_numeric_safe(s."Price Target - Median (YTD Ago)")          AS "Price Target - Median (YTD Ago)",
       text_to_numeric_safe(s."Price Target - Median (1Y Ago)")           AS "Price Target - Median (1Y Ago)",
       text_to_numeric_safe(s."Basic EPS - Cont (LTM)")                   AS "Basic EPS - Cont (LTM)",
       text_to_numeric_safe(s."Basic EPS - Cont (FQ)")                    AS "Basic EPS - Cont (FQ)",
       text_to_numeric_safe(s."Basic EPS - Cont (FY)")                    AS "Basic EPS - Cont (FY)",
       text_to_numeric_safe(s."Basic EPS - Cont (-1FQFQ)")                AS "Basic EPS - Cont (-1FQFQ)",
       text_to_numeric_safe(s."Basic EPS - Cont (-2FQFQ)")                AS "Basic EPS - Cont (-2FQFQ)",
       text_to_numeric_safe(s."Basic EPS - Cont (-4FQFQ)")                AS "Basic EPS - Cont (-4FQFQ)",
       text_to_numeric_safe(s."Basic EPS - Cont (-3FQFQ)")                AS "Basic EPS - Cont (-3FQFQ)",
       text_to_numeric_safe(s."Basic EPS - Cont (-1FY)")                  AS "Basic EPS - Cont (-1FY)",
       text_to_numeric_safe(s."Basic EPS - Cont (-2FY)")                  AS "Basic EPS - Cont (-2FY)",
       text_to_numeric_safe(s."Basic EPS - Cont (-3FY)")                  AS "Basic EPS - Cont (-3FY)",
       text_to_numeric_safe(s."Basic EPS - Cont (-4FY)")                  AS "Basic EPS - Cont (-4FY)",
       text_to_numeric_safe(s."EPS/Adj. (FQ)")                            AS "EPS/Adj. (FQ)",
       text_to_numeric_safe(s."EPS/Adj. (-1FQFQ)")                        AS "EPS/Adj. (-1FQFQ)",
       text_to_numeric_safe(s."EPS/Adj. (-3FQFQ)")                        AS "EPS/Adj. (-3FQFQ)",
       text_to_numeric_safe(s."EPS/Adj. (-4FQFQ)")                        AS "EPS/Adj. (-4FQFQ)",
       text_to_numeric_safe(s."EPS/Adj. (-2FQFQ)")                        AS "EPS/Adj. (-2FQFQ)",
       text_to_numeric_safe(s."EPS/Adj. (-2FY)")                          AS "EPS/Adj. (-2FY)",
       text_to_numeric_safe(s."EPS/Adj. (-3FY)")                          AS "EPS/Adj. (-3FY)",
       text_to_numeric_safe(s."EPS/Adj. (-4FY)")                          AS "EPS/Adj. (-4FY)",
       text_to_numeric_safe(s."Gross Profit (-1FQFQ)")                    AS "Gross Profit (-1FQFQ)",
       text_to_numeric_safe(s."Gross Profit (-3FQFQ)")                    AS "Gross Profit (-3FQFQ)",
       text_to_numeric_safe(s."Gross Profit (-4FQFQ)")                    AS "Gross Profit (-4FQFQ)",
       text_to_numeric_safe(s."Gross Profit (-2FQFQ)")                    AS "Gross Profit (-2FQFQ)",
       text_to_numeric_safe(s."Gross Profit (-1FY)")                      AS "Gross Profit (-1FY)",
       text_to_numeric_safe(s."Gross Profit (-2FY)")                      AS "Gross Profit (-2FY)",
       text_to_numeric_safe(s."Gross Profit (-3FY)")                      AS "Gross Profit (-3FY)",
       text_to_numeric_safe(s."Gross Profit (-4FY)")                      AS "Gross Profit (-4FY)",
       text_to_numeric_safe(s."FCF - Est Avg (FY1E)")                     AS "FCF - Est Avg (FY1E)",
       text_to_numeric_safe(s."FCF - Est Avg (FY2E)")                     AS "FCF - Est Avg (FY2E)",
       text_to_numeric_safe(s."FCF - Est Avg (FY3E)")                     AS "FCF - Est Avg (FY3E)",
       text_to_numeric_safe(s."FCF - Est Avg (FY4E)")                     AS "FCF - Est Avg (FY4E)",
       text_to_numeric_safe(s."FCF - Est Avg (FY5E)")                     AS "FCF - Est Avg (FY5E)",
       text_to_numeric_safe(s."EPS (-0FYEstimate)")                       AS "EPS (-0FYEstimate)",
       text_to_numeric_safe(s."EPS (-0FYActual)")                         AS "EPS (-0FYActual)",
       text_to_numeric_safe(s."EPS (-0FYSurprise %)")                     AS "EPS (-0FYSurprise %)",
       text_to_numeric_safe(s."EPS (-1FYEstimate)")                       AS "EPS (-1FYEstimate)",
       text_to_numeric_safe(s."EPS (-1FYActual)")                         AS "EPS (-1FYActual)",
       text_to_numeric_safe(s."EPS (-1FYSurprise %)")                     AS "EPS (-1FYSurprise %)",
       text_to_numeric_safe(s."EPS (-2FYEstimate)")                       AS "EPS (-2FYEstimate)",
       text_to_numeric_safe(s."EPS (-2FYActual)")                         AS "EPS (-2FYActual)",
       text_to_numeric_safe(s."EPS (-2FYSurprise %)")                     AS "EPS (-2FYSurprise %)",
       text_to_numeric_safe(s."EPS (-3FYEstimate)")                       AS "EPS (-3FYEstimate)",
       text_to_numeric_safe(s."EPS (-3FYActual)")                         AS "EPS (-3FYActual)",
       text_to_numeric_safe(s."EPS (-3FYSurprise %)")                     AS "EPS (-3FYSurprise %)",
       text_to_numeric_safe(s."EPS (-4FYActual)")                         AS "EPS (-4FYActual)",
       text_to_numeric_safe(s."EPS (-4FYEstimate)")                       AS "EPS (-4FYEstimate)",
       text_to_numeric_safe(s."EPS (-4FYSurprise %)")                     AS "EPS (-4FYSurprise %)",
       text_to_numeric_safe(s."EPS (-5FYEstimate)")                       AS "EPS (-5FYEstimate)",
       text_to_numeric_safe(s."EPS (-5FYActual)")                         AS "EPS (-5FYActual)",
       text_to_numeric_safe(s."EPS (-5FYSurprise %)")                     AS "EPS (-5FYSurprise %)",
       text_to_numeric_safe(s."EPS (-0FQEstimate)")                       AS "EPS (-0FQEstimate)",
       text_to_numeric_safe(s."EPS (-0FQActual)")                         AS "EPS (-0FQActual)",
       text_to_numeric_safe(s."EPS (-0FQSurprise %)")                     AS "EPS (-0FQSurprise %)",
       text_to_numeric_safe(s."EPS (-1FQEstimate)")                       AS "EPS (-1FQEstimate)",
       text_to_numeric_safe(s."EPS (-1FQActual)")                         AS "EPS (-1FQActual)",
       text_to_numeric_safe(s."EPS (-1FQSurprise %)")                     AS "EPS (-1FQSurprise %)",
       text_to_numeric_safe(s."EPS (-2FQEstimate)")                       AS "EPS (-2FQEstimate)",
       text_to_numeric_safe(s."EPS (-2FQActual)")                         AS "EPS (-2FQActual)",
       text_to_numeric_safe(s."EPS (-2FQSurprise %)")                     AS "EPS (-2FQSurprise %)",
       text_to_numeric_safe(s."EPS (-3FQEstimate)")                       AS "EPS (-3FQEstimate)",
       text_to_numeric_safe(s."EPS (-3FQActual)")                         AS "EPS (-3FQActual)",
       text_to_numeric_safe(s."EPS (-3FQSurprise %)")                     AS "EPS (-3FQSurprise %)",
       text_to_numeric_safe(s."EPS (-4FQEstimate)")                       AS "EPS (-4FQEstimate)",
       text_to_numeric_safe(s."EPS (-4FQActual)")                         AS "EPS (-4FQActual)",
       text_to_numeric_safe(s."EPS (-4FQSurprise %)")                     AS "EPS (-4FQSurprise %)",
       text_to_numeric_safe(s."FCF (LTM)")                                AS "FCF (LTM)",
       text_to_numeric_safe(s."FCF (FQ)")                                 AS "FCF (FQ)",
       text_to_numeric_safe(s."FCF (-1FQFQ)")                             AS "FCF (-1FQFQ)",
       text_to_numeric_safe(s."FCF (-3FQFQ)")                             AS "FCF (-3FQFQ)",
       text_to_numeric_safe(s."FCF (-4FQFQ)")                             AS "FCF (-4FQFQ)",
       text_to_numeric_safe(s."FCF (-2FQFQ)")                             AS "FCF (-2FQFQ)",
       text_to_numeric_safe(s."FCF (FY)")                                 AS "FCF (FY)",
       text_to_numeric_safe(s."FCF (-1FY)")                               AS "FCF (-1FY)",
       text_to_numeric_safe(s."FCF (-3FY)")                               AS "FCF (-3FY)",
       text_to_numeric_safe(s."FCF (-2FY)")                               AS "FCF (-2FY)",
       text_to_numeric_safe(s."FCF (-4FY)")                               AS "FCF (-4FY)",
       text_to_numeric_safe(s."FCF (-5FY)")                               AS "FCF (-5FY)",
       text_to_numeric_safe(s."Target % (Avg)")                           AS "Target % (Avg)",
       text_to_numeric_safe(s."Target % (Med)")                           AS "Target % (Med)",
       text_to_numeric_safe(s."Target % (Low)")                           AS "Target % (Low)",
       text_to_numeric_safe(s."Target % (High)")                          AS "Target % (High)",
       text_to_numeric_safe(s."Price Target - StdDev")                    AS "Price Target - StdDev",
       text_to_numeric_safe(s."Price Target - StdDev (1W Ago)")           AS "Price Target - StdDev (1W Ago)",
       text_to_numeric_safe(s."Price Target - StdDev (1M Ago)")           AS "Price Target - StdDev (1M Ago)",
       text_to_numeric_safe(s."Price Target - StdDev (3M Ago)")           AS "Price Target - StdDev (3M Ago)",
       text_to_numeric_safe(s."Price Target - StdDev (6M Ago)")           AS "Price Target - StdDev (6M Ago)",
       text_to_numeric_safe(s."Price Target - StdDev (1Y Ago)")           AS "Price Target - StdDev (1Y Ago)",
       text_to_numeric_safe(s."Altman Z-Score (-1FY)")                    AS "Altman Z-Score (-1FY)",
       text_to_numeric_safe(s."Altman Z-Score (-2FY)")                    AS "Altman Z-Score (-2FY)",
       text_to_numeric_safe(s."Altman Z-Score (-3FY)")                    AS "Altman Z-Score (-3FY)",
       text_to_numeric_safe(s."Altman Z-Score (-4FY)")                    AS "Altman Z-Score (-4FY)",
       text_to_numeric_safe(s."Altman Z-Score (-5FY)")                    AS "Altman Z-Score (-5FY)",
       text_to_numeric_safe(s."Altman Z-Score (-1FQFQ)")                  AS "Altman Z-Score (-1FQFQ)",
       text_to_numeric_safe(s."Altman Z-Score (-2FQFQ)")                  AS "Altman Z-Score (-2FQFQ)",
       text_to_numeric_safe(s."Altman Z-Score (-3FQFQ)")                  AS "Altman Z-Score (-3FQFQ)",
       text_to_numeric_safe(s."Altman Z-Score (-4FQFQ)")                  AS "Altman Z-Score (-4FQFQ)",
       text_to_numeric_safe(s."Altman Z-Score (-0FYYoYLTM)")              AS "Altman Z-Score (-0FYYoYLTM)",
       text_to_numeric_safe(s."Altman Z-Score (-1FYYoYLTM)")              AS "Altman Z-Score (-1FYYoYLTM)",
       text_to_numeric_safe(s."Altman Z-Score (-3FYYoYLTM)")              AS "Altman Z-Score (-3FYYoYLTM)",
       text_to_numeric_safe(s."Altman Z-Score (-4FYYoYLTM)")              AS "Altman Z-Score (-4FYYoYLTM)",
       text_to_numeric_safe(s."Altman Z-Score (-5FYYoYLTM)")              AS "Altman Z-Score (-5FYYoYLTM)",
       text_to_numeric_safe(s."Altman Z-Score (-2FYYoYLTM)")              AS "Altman Z-Score (-2FYYoYLTM)",
       text_to_numeric_safe(s."P/E (EST FY2)")                            AS "P/E (EST FY2)",
       text_to_numeric_safe(s."P/E (EST FY3)")                            AS "P/E (EST FY3)",
       text_to_numeric_safe(s."P/E (EST FY4)")                            AS "P/E (EST FY4)",
       text_to_numeric_safe(s."P/E (EST FY5)")                            AS "P/E (EST FY5)",
       text_to_numeric_safe(s."P/E (-4FYLTM)")                            AS "P/E (-4FYLTM)",
       text_to_numeric_safe(s."P/E (-4FQLTM)")                            AS "P/E (-4FQLTM)",
       text_to_numeric_safe(s."P/E (3YAVGNTM)")                           AS "P/E (3YAVGNTM)",
       text_to_numeric_safe(s."P/E (5YAVGNTM)")                           AS "P/E (5YAVGNTM)",
       text_to_numeric_safe(s."EPS Norm - Est Avg (FQ1E)")                AS "EPS Norm - Est Avg (FQ1E)",
       text_to_numeric_safe(s."EPS Norm - Est Avg (FQ2E)")                AS "EPS Norm - Est Avg (FQ2E)",
       text_to_numeric_safe(s."EPS Norm - Est Avg (FQ3E)")                AS "EPS Norm - Est Avg (FQ3E)",
       text_to_numeric_safe(s."EPS Norm - Est Avg (FQ4E)")                AS "EPS Norm - Est Avg (FQ4E)",
       text_to_numeric_safe(s."EPS Norm - Est Avg (FY2E)")                AS "EPS Norm - Est Avg (FY2E)",
       text_to_numeric_safe(s."EPS Norm - Est Avg (FY3E)")                AS "EPS Norm - Est Avg (FY3E)",
       text_to_numeric_safe(s."EPS Norm - Est Avg (FY4E)")                AS "EPS Norm - Est Avg (FY4E)",
       text_to_numeric_safe(s."EPS Norm - Est Avg (FY5E)")                AS "EPS Norm - Est Avg (FY5E)",
       text_to_numeric_safe(s."Capital Expenditure (LTM)")                AS "Capital Expenditure (LTM)",
       text_to_numeric_safe(s."Capital Expenditure (FQ)")                 AS "Capital Expenditure (FQ)",
       text_to_numeric_safe(s."Capital Expenditure (FY)")                 AS "Capital Expenditure (FY)",
       text_to_numeric_safe(s."Capital Expenditure (-1FQFQ)")             AS "Capital Expenditure (-1FQFQ)",
       text_to_numeric_safe(s."Capital Expenditure (-2FQFQ)")             AS "Capital Expenditure (-2FQFQ)",
       text_to_numeric_safe(s."Capital Expenditure (-3FQFQ)")             AS "Capital Expenditure (-3FQFQ)",
       text_to_numeric_safe(s."Capital Expenditure (-4FQFQ)")             AS "Capital Expenditure (-4FQFQ)",
       text_to_numeric_safe(s."Capital Expenditure (-1FY)")               AS "Capital Expenditure (-1FY)",
       text_to_numeric_safe(s."Capital Expenditure (-2FY)")               AS "Capital Expenditure (-2FY)",
       text_to_numeric_safe(s."Capital Expenditure (-4FY)")               AS "Capital Expenditure (-4FY)",
       text_to_numeric_safe(s."Capital Expenditure (-3FY)")               AS "Capital Expenditure (-3FY)",
       text_to_numeric_safe(s."Capital Expenditure (-5FY)")               AS "Capital Expenditure (-5FY)",
       text_to_numeric_safe(s."CFF (LTM)")                                AS "CFF (LTM)",
       text_to_numeric_safe(s."CFF (FQ)")                                 AS "CFF (FQ)",
       text_to_numeric_safe(s."CFF (FY)")                                 AS "CFF (FY)",
       text_to_numeric_safe(s."CFF (-1FQFQ)")                             AS "CFF (-1FQFQ)",
       text_to_numeric_safe(s."CFF (-2FQFQ)")                             AS "CFF (-2FQFQ)",
       text_to_numeric_safe(s."CFF (-3FQFQ)")                             AS "CFF (-3FQFQ)",
       text_to_numeric_safe(s."CFF (-4FQFQ)")                             AS "CFF (-4FQFQ)",
       text_to_numeric_safe(s."CFF (-1FY)")                               AS "CFF (-1FY)",
       text_to_numeric_safe(s."CFF (-2FY)")                               AS "CFF (-2FY)",
       text_to_numeric_safe(s."CFF (-3FY)")                               AS "CFF (-3FY)",
       text_to_numeric_safe(s."CFF (-4FY)")                               AS "CFF (-4FY)",
       text_to_numeric_safe(s."CFI (LTM)")                                AS "CFI (LTM)",
       text_to_numeric_safe(s."CFI (FQ)")                                 AS "CFI (FQ)",
       text_to_numeric_safe(s."CFI (FY)")                                 AS "CFI (FY)",
       text_to_numeric_safe(s."CFI (-1FQFQ)")                             AS "CFI (-1FQFQ)",
       text_to_numeric_safe(s."CFI (-2FQFQ)")                             AS "CFI (-2FQFQ)",
       text_to_numeric_safe(s."CFI (-3FQFQ)")                             AS "CFI (-3FQFQ)",
       text_to_numeric_safe(s."CFI (-4FQFQ)")                             AS "CFI (-4FQFQ)",
       text_to_numeric_safe(s."CFI (-1FY)")                               AS "CFI (-1FY)",
       text_to_numeric_safe(s."CFI (-2FY)")                               AS "CFI (-2FY)",
       text_to_numeric_safe(s."CFI (-3FY)")                               AS "CFI (-3FY)",
       text_to_numeric_safe(s."CFI (-5FY)")                               AS "CFI (-5FY)",
       text_to_numeric_safe(s."CFI (-4FY)")                               AS "CFI (-4FY)",
       text_to_numeric_safe(s."CFO (LTM)")                                AS "CFO (LTM)",
       text_to_numeric_safe(s."CFO (FQ)")                                 AS "CFO (FQ)",
       text_to_numeric_safe(s."CFO (FY)")                                 AS "CFO (FY)",
       text_to_numeric_safe(s."CFO (-1FQFQ)")                             AS "CFO (-1FQFQ)",
       text_to_numeric_safe(s."CFO (-2FQFQ)")                             AS "CFO (-2FQFQ)",
       text_to_numeric_safe(s."CFO (-4FQFQ)")                             AS "CFO (-4FQFQ)",
       text_to_numeric_safe(s."CFO (-3FQFQ)")                             AS "CFO (-3FQFQ)",
       text_to_numeric_safe(s."CFO (-1FY)")                               AS "CFO (-1FY)",
       text_to_numeric_safe(s."CFO (-2FY)")                               AS "CFO (-2FY)",
       text_to_numeric_safe(s."CFO (-3FY)")                               AS "CFO (-3FY)",
       text_to_numeric_safe(s."CFO (-4FY)")                               AS "CFO (-4FY)",
       text_to_numeric_safe(s."CFO (-5FY)")                               AS "CFO (-5FY)",
       text_to_numeric_safe(s."Dividend Per Share (FQ)")                  AS "Dividend Per Share (FQ)",
       text_to_numeric_safe(s."Dividend Per Share (FY)")                  AS "Dividend Per Share (FY)",
       text_to_numeric_safe(s."Dividend Per Share (-1FQFQ)")              AS "Dividend Per Share (-1FQFQ)",
       text_to_numeric_safe(s."Dividend Per Share (-2FQFQ)")              AS "Dividend Per Share (-2FQFQ)",
       text_to_numeric_safe(s."Dividend Per Share (-3FQFQ)")              AS "Dividend Per Share (-3FQFQ)",
       text_to_numeric_safe(s."Dividend Per Share (-4FQFQ)")              AS "Dividend Per Share (-4FQFQ)",
       text_to_numeric_safe(s."Dividend Per Share (-1FY)")                AS "Dividend Per Share (-1FY)",
       text_to_numeric_safe(s."Dividend Per Share (-2FY)")                AS "Dividend Per Share (-2FY)",
       text_to_numeric_safe(s."Dividend Per Share (-3FY)")                AS "Dividend Per Share (-3FY)",
       text_to_numeric_safe(s."Dividend Per Share (-4FY)")                AS "Dividend Per Share (-4FY)",
       text_to_numeric_safe(s."Dividend Per Share (-5FY)")                AS "Dividend Per Share (-5FY)",
       text_to_numeric_safe(s."Enterprise Value (-1FQ)")                  AS "Enterprise Value (-1FQ)",
       text_to_numeric_safe(s."Enterprise Value (-2FQ)")                  AS "Enterprise Value (-2FQ)",
       text_to_numeric_safe(s."Enterprise Value (-3FQ)")                  AS "Enterprise Value (-3FQ)",
       text_to_numeric_safe(s."Enterprise Value (-4FQ)")                  AS "Enterprise Value (-4FQ)",
       text_to_numeric_safe(s."Enterprise Value (-1FY)")                  AS "Enterprise Value (-1FY)",
       text_to_numeric_safe(s."Enterprise Value (-2FY)")                  AS "Enterprise Value (-2FY)",
       text_to_numeric_safe(s."Enterprise Value (-3FY)")                  AS "Enterprise Value (-3FY)",
       text_to_numeric_safe(s."Enterprise Value (-4FY)")                  AS "Enterprise Value (-4FY)",
       text_to_numeric_safe(s."Enterprise Value (-5FY)")                  AS "Enterprise Value (-5FY)",
       text_to_numeric_safe(s."Volatility (1M)")                          AS "Volatility (1M)",
       text_to_numeric_safe(s."Volatility (3M)")                          AS "Volatility (3M)",
       text_to_numeric_safe(s."Volatility (6M)")                          AS "Volatility (6M)",
       text_to_numeric_safe(s."Volatility (1Y)")                          AS "Volatility (1Y)",
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 1W)")       AS "EPS GAAP Est Avg Rev % (FY1E - 1W)",
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - MTD)")      AS "EPS GAAP Est Avg Rev % (FY1E - MTD)",
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - QTD)")      AS "EPS GAAP Est Avg Rev % (FY1E - QTD)",
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - YTD)")      AS "EPS GAAP Est Avg Rev % (FY1E - YTD)",
       text_to_numeric_safe(s."Price Target - StdDev (MTD Ago)")          AS "Price Target - StdDev (MTD Ago)",
       text_to_numeric_safe(s."Price Target - StdDev (QTD Ago)")          AS "Price Target - StdDev (QTD Ago)",
       text_to_numeric_safe(s."Price Target - StdDev (YTD Ago)")          AS "Price Target - StdDev (YTD Ago)"
FROM staging_header_buf s;
-- FINAL VALIDATION
-- ===================================================================
\echo 'Final validation...'
SELECT 'Total rows in pml_df:' AS info, COUNT(*) AS count
FROM pml_df;
SELECT 'Rows by Region:' AS info, "Region", COUNT(*) AS count
FROM pml_df
GROUP BY "Region"
ORDER BY "Region";
SELECT 'Rows by Sector (top 10):' AS info, "Sector", COUNT(*) AS count
FROM pml_df
GROUP BY "Sector"
ORDER BY COUNT(*) DESC
LIMIT 10;

-- ===================================================================
-- CLEANUP
-- ===================================================================
DROP TABLE IF EXISTS staging_header_buf;
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
