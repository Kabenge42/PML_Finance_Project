-- ===================================================================
-- Equities Data Import Script
-- ===================================================================
-- Documentation: See docs/column_mapping_reference.md for column aliases
-- Usage: psql -h localhost -p 5432 -U postgres -d postgres -f import_pml_data.sql
\echo 'Starting pml_df data import...'

-- ===================================================================
-- SESSION-LEVEL TUNING FOR BULK IMPORT
-- ===================================================================
SET search_path = pml; -- Resolve pml_df / pml_us without schema prefix
SET work_mem = '256MB'; -- Increase memory for sorting/hashing operations
SET maintenance_work_mem = '512MB'; -- Increase memory for maintenance operations
SET SYNCHRONOUS_COMMIT = OFF; -- Defer WAL writes (faster, but less durable during import)
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
CREATE OR REPLACE FUNCTION month_abbrev_to_number(month_abbrev TEXT) RETURNS INTEGER AS
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
	       WHEN 'DEC' THEN 12 END
$$
	LANGUAGE sql IMMUTABLE
	             PARALLEL SAFE;

-- ===================================================================
-- HELPER FUNCTION: Get Expected Reporting Lag Days
-- ===================================================================
-- Returns the typical number of days between period end and earnings release
CREATE OR REPLACE FUNCTION get_expected_reporting_lag_days(earnings_report_frequency TEXT) RETURNS INTEGER AS
$$
SELECT CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'QUARTERLY')))
	       WHEN 'QUARTERLY' THEN 45
	       WHEN 'SEMI-ANNUAL' THEN 60
	       WHEN 'SEMI-ANNUALLY' THEN 60
	       WHEN 'ANNUAL' THEN 90
	       WHEN 'ANNUALLY' THEN 90
	       ELSE 45 END
$$
	LANGUAGE sql IMMUTABLE
	             PARALLEL SAFE;

-- Converts TEXT to NUMERIC, treating common non-numeric patterns as NULL
CREATE OR REPLACE FUNCTION text_to_numeric_safe(input_text TEXT) RETURNS NUMERIC AS
$$
SELECT CASE
	       WHEN input_text IS NULL OR
	            TRIM(input_text) IN ('', '-', '--', 'N/A', 'NA', 'NULL', 'NONE', 'n/a', 'na', 'null', 'none') THEN NULL
	       WHEN TRIM(input_text) ~ '^-?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$' THEN TRIM(input_text)::NUMERIC END AS result
$$
	LANGUAGE sql IMMUTABLE
	             PARALLEL SAFE;

-- Converts TEXT to DATE safely, returns NULL for invalid input.
-- Auto-detects common formats found in vendor CSV exports, including:
--   * 'Mon-DD-YYYY'  (e.g. 'Mar-16-2026')   <-- primary PML CSV format
--   * 'YYYY-MM-DD'   (ISO 8601)
--   * 'MM/DD/YYYY'   (US)
--   * 'DD/MM/YYYY'   (EU)
-- If `date_format` is supplied explicitly (and not the default sentinel
-- 'AUTO'), it is honored as-is via TO_DATE().
CREATE OR REPLACE FUNCTION text_to_date_safe(input_text TEXT, date_format TEXT DEFAULT 'AUTO') RETURNS DATE AS
$$
DECLARE
	v TEXT;
BEGIN
	IF input_text IS NULL THEN RETURN NULL; END IF;
	v := TRIM(input_text);
	IF v IN ('', '-', '--', 'N/A', 'NA', 'NULL', 'NONE', 'n/a', 'na', 'null', 'none') THEN RETURN NULL; END IF;

	-- Explicit format requested by caller
	IF date_format IS NOT NULL AND date_format <> 'AUTO' THEN
		BEGIN
			RETURN TO_DATE(v, date_format);
		EXCEPTION
			WHEN OTHERS THEN RETURN NULL;
		END;
	END IF;

	-- Auto-detect by shape
	-- 'Mon-DD-YYYY' e.g. Mar-16-2026
	IF v ~ '^[A-Za-z]{3}-\d{1,2}-\d{4}$' THEN
		BEGIN
			RETURN TO_DATE(v, 'Mon-DD-YYYY');
		EXCEPTION
			WHEN OTHERS THEN RETURN NULL;
		END;
	END IF;

	-- 'Mon DD, YYYY' e.g. 'Mar 16, 2026'
	IF v ~ '^[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}$' THEN
		BEGIN
			RETURN TO_DATE(v, 'Mon DD, YYYY');
		EXCEPTION
			WHEN OTHERS THEN RETURN NULL;
		END;
	END IF;

	-- ISO 'YYYY-MM-DD'
	IF v ~ '^\d{4}-\d{2}-\d{2}$' THEN
		BEGIN
			RETURN TO_DATE(v, 'YYYY-MM-DD');
		EXCEPTION
			WHEN OTHERS THEN RETURN NULL;
		END;
	END IF;

	-- 'MM/DD/YYYY' (US)
	IF v ~ '^\d{1,2}/\d{1,2}/\d{4}$' THEN
		BEGIN
			RETURN TO_DATE(v, 'MM/DD/YYYY');
		EXCEPTION
			WHEN OTHERS THEN RETURN NULL;
		END;
	END IF;

	-- Last-resort: let PostgreSQL's input parser try
	BEGIN
		RETURN v::DATE;
	EXCEPTION
		WHEN OTHERS THEN RETURN NULL;
	END;
END;
$$
	LANGUAGE plpgsql IMMUTABLE
	                 STRICT;

-- ===================================================================
-- HELPER FUNCTION: Parse FY End to Date
-- ===================================================================
CREATE OR REPLACE FUNCTION parse_fiscal_year_end_date(fy_end_text TEXT) RETURNS DATE AS
$$
DECLARE
	parts      TEXT[];
	month_num  INT;
	year_value INT;
BEGIN
	IF fy_end_text IS NULL OR TRIM(fy_end_text) = '' THEN RETURN NULL; END IF;

	parts := regexp_split_to_array(TRIM(fy_end_text), '\s+');
	IF array_length(parts, 1) < 2 OR parts[2] !~ '^\d{4}$' THEN RETURN NULL; END IF;

	year_value := parts[2]::INT;
	month_num := month_abbrev_to_number(parts[1]);

	IF month_num IS NULL OR year_value NOT BETWEEN 1900 AND 2100 THEN RETURN NULL; END IF;

	-- Last day of month via single interval literal date-math idiom
	RETURN (MAKE_DATE(year_value, month_num, 1) + INTERVAL '1 month - 1 day')::DATE;
END;
$$
	LANGUAGE plpgsql IMMUTABLE
	                 STRICT;

-- ===================================================================
-- HELPER FUNCTION: Convert Frequency to Interval Months
-- ===================================================================
CREATE OR REPLACE FUNCTION frequency_to_months(
	earnings_report_frequency TEXT,
	fy_end_date               DATE DEFAULT NULL,
	next_fy_end_date          DATE DEFAULT NULL
) RETURNS INTEGER AS
$$
DECLARE
	fy_range_months INT := 12;
BEGIN
	-- Use AGE() for month arithmetic — correct across year boundaries
	IF fy_end_date IS NOT NULL AND next_fy_end_date IS NOT NULL THEN
		fy_range_months := (DATE_PART('year', AGE(next_fy_end_date, fy_end_date)) * 12 +
		                    DATE_PART('month', AGE(next_fy_end_date, fy_end_date)))::INT;
	END IF;

	RETURN CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'QUARTERLY')))
		       WHEN 'QUARTERLY' THEN fy_range_months / 4
		       WHEN 'SEMI-ANNUAL' THEN fy_range_months / 2
		       WHEN 'SEMI-ANNUALLY' THEN fy_range_months / 2
		       WHEN 'ANNUAL' THEN fy_range_months
		       WHEN 'ANNUALLY' THEN fy_range_months
		       ELSE fy_range_months / 4 END;
END;
$$
	LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Convert Interval Months to Frequency Text
-- ===================================================================
CREATE OR REPLACE FUNCTION months_to_frequency(interval_months INTEGER) RETURNS TEXT AS
$$
SELECT CASE
	       WHEN interval_months IS NULL THEN 'Quarterly'
	       WHEN interval_months <= 3 THEN 'Quarterly'
	       WHEN interval_months <= 6 THEN 'Semi-Annually'
	       ELSE 'Annually' END
$$
	LANGUAGE sql IMMUTABLE
	             PARALLEL SAFE;

-- ===================================================================
-- HELPER FUNCTION: Derive Earnings Report Frequency
-- ===================================================================
CREATE OR REPLACE FUNCTION derive_earnings_report_frequency(
	income_statement_report_date DATE,
	fy_end_date                  DATE
) RETURNS TEXT AS
$$
DECLARE
	months_diff INT;
BEGIN
	IF income_statement_report_date IS NULL OR fy_end_date IS NULL THEN RETURN 'Quarterly'; END IF;

	-- AGE() handles direction & year wrap automatically; it also
	-- respects the DAY component, unlike raw EXTRACT() subtraction.
	months_diff := ABS((DATE_PART('year', AGE(income_statement_report_date, fy_end_date)) * 12 +
	                    DATE_PART('month', AGE(income_statement_report_date, fy_end_date)))::INT);

	-- Normalize within a 12-month window, but treat exact FY-end (0) as Annually
	-- rather than conflating it with Semi-Annually.
	IF months_diff = 0 THEN RETURN 'Annually'; END IF;

	months_diff := months_diff % 12;
	IF months_diff = 0 THEN months_diff := 12; END IF;

	RETURN CASE WHEN months_diff = 12 THEN 'Annually' WHEN months_diff = 6 THEN 'Semi-Annually' ELSE 'Quarterly' END;
END;
$$
	LANGUAGE plpgsql IMMUTABLE;

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
	IF reference_date IS NULL OR fy_end_date IS NULL THEN RETURN; END IF;

	-- Use make_interval + AGE() for date-math-correct boundaries
	next_fy_end_date := (fy_end_date + make_interval(years => 1))::DATE;

	-- Fiscal year range in months via AGE() (respects day component)
	fy_range_months := (DATE_PART('year', AGE(next_fy_end_date, fy_end_date)) * 12 +
	                    DATE_PART('month', AGE(next_fy_end_date, fy_end_date)))::INTEGER;

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
		                   ELSE fy_range_months / 4 END;

	reporting_interval := interval_months;

	-- Calculate periods per fiscal year based on the FY range
	periods_per_year := fy_range_months / interval_months;

	-- Months since fiscal year end via AGE() (correct across year boundaries
	-- AND accounts for the day of month, unlike raw EXTRACT subtraction).
	months_since_fy_end := (DATE_PART('year', AGE(reference_date, fy_end_date)) * 12 +
	                        DATE_PART('month', AGE(reference_date, fy_end_date)))::INTEGER;

	-- Fiscal month (1..fy_range_months); safe for negative months_since_fy_end too
	fiscal_month := ((months_since_fy_end - 1) % fy_range_months + fy_range_months) % fy_range_months + 1;

	-- Fiscal quarter derived from fiscal month relative to FY range
	-- Each quarter represents (fy_range_months / 4) months
	fiscal_quarter := CEIL(fiscal_month / (fy_range_months / 4.0))::INTEGER;

	-- Ensure fiscal_quarter stays within 1-4 range
	IF fiscal_quarter > 4 THEN fiscal_quarter := 4; END IF;

	-- Calculate current reporting period within the fiscal year
	current_period := CEIL(fiscal_month / interval_months::NUMERIC)::INTEGER;
	IF current_period > periods_per_year THEN current_period := periods_per_year; END IF;

	-- Calculate next reporting period
	next_period := current_period + 1;
	IF next_period > periods_per_year THEN next_period := 1; END IF;

	-- Convert next_period back to quarter for output
	-- Next quarter is derived from which reporting period we're moving to
	next_quarter := CASE
		                WHEN periods_per_year = 4 THEN next_period -- Quarterly
		                WHEN periods_per_year = 2 THEN next_period * 2 -- Semi-annual (Q2 or Q4)
		                WHEN periods_per_year = 1 THEN 4 -- Annual (always Q4/full year)
		                ELSE ((fiscal_quarter + (interval_months / (fy_range_months / 4)) - 1) % 4) + 1 END;

	-- Fiscal year calculations based on FY range
	fiscal_year := EXTRACT(YEAR FROM fy_end_date)::INTEGER + 1 + ((months_since_fy_end - 1) / fy_range_months);

	-- Next quarter year
	next_quarter_year :=
			CASE WHEN next_period = 1 AND current_period = periods_per_year THEN fiscal_year + 1 ELSE fiscal_year END;

	-- Report type derived from reporting periods and FY range
	next_earnings_report_type := CASE
		-- Full year if annual reporting OR if next period completes the FY
		                             WHEN interval_months = fy_range_months THEN 'Full Year'
		                             WHEN next_period = periods_per_year AND periods_per_year > 1 THEN 'Full Year'
		-- Half year for semi-annual mid-year report
		                             WHEN interval_months = fy_range_months / 2 AND next_period = 1 THEN 'Half Year'
		                             ELSE 'Interim' END;
END;
$$
	LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next Income Statement Report Date
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_next_income_statement_report_date(
	income_statement_report_date DATE,
	earnings_report_frequency    TEXT
) RETURNS DATE AS
$$
SELECT CASE
	       WHEN income_statement_report_date IS NULL THEN NULL
	       ELSE (income_statement_report_date +
	             make_interval(months => frequency_to_months(earnings_report_frequency)))::DATE END
$$
	LANGUAGE sql IMMUTABLE
	             PARALLEL SAFE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next Fiscal Year End Date
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_next_fy_end_date(fy_end_date DATE) RETURNS DATE AS
$$
SELECT (fy_end_date + make_interval(years => 1))::DATE
$$
	LANGUAGE sql IMMUTABLE
	             STRICT
	             PARALLEL SAFE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next Fiscal Quarter
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_next_fiscal_quarter(
	next_earnings_date           DATE,
	income_statement_report_date DATE,
	fy_end_date                  DATE,
	earnings_report_frequency    TEXT DEFAULT 'Quarterly'
) RETURNS INTEGER AS
$$
DECLARE
	reference_date   DATE;
	interval_months  INT;
	years_ahead      INT;
	current_fy_start DATE;
	months_into_fy   INT;
BEGIN
	IF fy_end_date IS NULL THEN RETURN NULL; END IF;

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
			years_ahead := FLOOR((DATE_PART('year', AGE(reference_date, fy_end_date)) * 12 +
			                      DATE_PART('month', AGE(reference_date, fy_end_date)))::NUMERIC / 12)::INT + 1;
	END IF;

	-- Start of the current fiscal year = (fy_end + (years_ahead - 1) years) + 1 day
	current_fy_start := (fy_end_date + make_interval(years => years_ahead - 1) + INTERVAL '1 day')::DATE;

	-- Months into FY using AGE (handles month-length variations correctly)
	months_into_fy := (DATE_PART('year', AGE(reference_date, current_fy_start)) * 12 +
	                   DATE_PART('month', AGE(reference_date, current_fy_start)))::INT + 1;

	-- Safe 1–12 normalization even for negative values
	months_into_fy := ((months_into_fy - 1) % 12 + 12) % 12 + 1;

	RETURN LEAST(4, GREATEST(1, CEIL(months_into_fy / 3.0)::INT));
END;
$$
	LANGUAGE plpgsql IMMUTABLE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Reporting Lag
-- ===================================================================
-- Returns the ACTUAL lag in days between the next earnings date and the
-- most recent income-statement report date, along with the deviation from
-- the EXPECTED lag for the given frequency. Using date subtraction (Date Math)
-- returns an integer number of days directly.
CREATE OR REPLACE FUNCTION calculate_reporting_lag(
	next_earnings                DATE,
	income_statement_report_date DATE,
	earnings_report_frequency    TEXT DEFAULT 'Quarterly'
) RETURNS INTEGER AS
$$
SELECT CASE
	       WHEN next_earnings IS NULL OR income_statement_report_date IS NULL
		       THEN NULL -- Date - Date returns an INTEGER number of days in PostgreSQL.
	-- We compare against the expected reporting lag for the given frequency
	-- to produce the deviation (positive = late, negative = early).
	       ELSE (next_earnings - income_statement_report_date) -
	            get_expected_reporting_lag_days(earnings_report_frequency) END
$$
	LANGUAGE sql IMMUTABLE
	             PARALLEL SAFE;

-- ===================================================================
-- HELPER FUNCTION: Calculate Expected Report Date
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_expected_report_date(
	period_end_date           DATE,
	earnings_report_frequency TEXT
) RETURNS DATE AS
$$
SELECT CASE
	       WHEN period_end_date IS NULL THEN NULL
	       ELSE (period_end_date +
	             make_interval(days => get_expected_reporting_lag_days(earnings_report_frequency)))::DATE END
$$
	LANGUAGE sql IMMUTABLE
	             PARALLEL SAFE;

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
	IF fy_end_date IS NOT NULL AND
	   fy_end_date <> (DATE_TRUNC('month', fy_end_date) + INTERVAL '1 month - 1 day')::DATE THEN
		RETURN QUERY SELECT 'FY End is not last day of month'::TEXT, 'INFO'::TEXT;
	END IF;
END;
$$
	LANGUAGE plpgsql IMMUTABLE;

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
		v_header TEXT;
		v_cols   TEXT[];
		v_col    TEXT;
		v_ddl    TEXT    := 'CREATE TEMP TABLE staging_header_buf (';
		v_first  BOOLEAN := TRUE;
	BEGIN
		SELECT header_line INTO v_header FROM staging_header_buf LIMIT 1;
		IF v_header IS NULL THEN RAISE EXCEPTION 'Could not read header line from data/pml/pml_us.csv'; END IF;

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
				IF length(v_col) >= 2 AND left(v_col, 1) = '"' AND right(v_col, 1) = '"' THEN
					v_col := substring(v_col FROM 2 FOR length(v_col) - 2);
					-- Un-escape doubled quotes ("" -> ")
					v_col := replace(v_col, '""', '"');
				END IF;

				IF NOT v_first THEN v_ddl := v_ddl || ', '; END IF;
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
INSERT INTO pml_df (ticker, isin, name, description, trading_region, trading_country, exchange, unit, region, country,
                    sector, industry, style_class, size_class, last_updated, income_statement_report_date, fy_end,
                    next_earnings, next_earnings_when, next_earnings_status, fy_end_date,
                    next_income_statement_report_date, next_fy_end_date, days_to_earnings, earnings_report_recency,
                    expected_report_date, dividend_record_currency, dividend_record_amount, dividend_record_frequency,
                    dividend_streak, dividend_record_announce_date, dividend_record_payable_date,
                    dividend_record_record_date, dividend_record_ex_date, market_cap, enterprise_value, last_price,
                    price_target_ytd_ago, total_return_ytd, price_target, price_target_low, price_target_median,
                    price_target_high, price_target_num, p_e_ntm, p_e_ltm, altman_z_score_fy, altman_z_score_fq,
                    altman_z_score_ltm, beta_1y, beta_2y, beta_5y, analyst_rating, num_strong_sell_ratings,
                    num_strong_buys_ratings, num_hold_ratings, num_buys_ratings, num_sell_ratings,
                    num_no_opinion_ratings, market_cap_country_r, tot_return_pct_cagr_3y, tot_return_pct_cagr_10y,
                    total_return_5y, total_return_10y, volume_shrs, dividend_per_share_ltm, div_yield_ind,
                    div_yield_ltm, gross_profit_margin_pct_fy, gross_profit_margin_pct_ltm, eps_norm_est_avg_ntm,
                    eps_adj_neg1fy, eps_adj_fy, eps_adj_ltm, eps_norm_est_avg_fy1e, buyback_yield_ltm,
                    return_on_assets_roa_pct_ltm, return_on_assets_roa_pct_fy, div_yield_neg1fyind, p_b_ltm, p_b_neg1fy,
                    p_b_5yavg, div_yield_ttm, div_yield_ntm, div_yield_5yavgltm, price_chg_pct_3m, one_day_pct,
                    price_5d_ago, price_1w_ago, price_1m_ago, price_3m_ago, price_6m_ago, price_1y_ago, price_3y_ago,
                    price_5y_ago, price_qtd_ago, rel_volume, shrs_out, shrs_out_neg1fy, common_dividends_paid_ltm,
                    common_dividends_paid_fy, ev_sales_ltm, ev_sales_ntm, ev_sales_neg1fyltm, ev_sales_neg2fyltm,
                    ev_sales_neg3fyltm, ev_sales_3yavgltm, ev_sales_neg1fqltm, ev_sales_neg2fqltm, ev_sales_neg3fqltm,
                    ev_sales_neg4fqltm, w_52high_adj, w_52low_adj, ema_20d, ema_50d, ema_100d, ema_250d, ev_ebitda_ltm,
                    ev_ebitda_ntm, ev_ebitda_neg1fyltm, ev_ebitda_neg1fqltm, ev_ebitda_3yavgltm, ev_ebitda_est_fy1,
                    p_e_est_fy1, p_e_neg1fyltm, p_e_neg2fyltm, p_e_neg3fyltm, p_e_3yavgltm, p_e_neg1fqltm,
                    p_e_neg2fqltm, p_e_neg3fqltm, p_e_5yavgltm, p_e_neg0fqqoqltm, p_e_neg0fyyoyltm, p_e_neg1fyyoyltm,
                    p_e_neg0fqyoyltm, full_time_employees_fq, full_time_employees_fy, full_time_employees_neg1fy,
                    full_time_employees_neg2fy, full_time_employees_neg3fy, avg_employees_5yavgfy, net_eps_basic_ltm,
                    net_eps_basic_fq, net_eps_basic_fy, net_eps_basic_neg1fqfq, net_eps_basic_neg2fqfq,
                    net_eps_basic_neg3fqfq, net_eps_basic_neg4fqfq, net_eps_basic_neg1fy, net_eps_basic_neg2fy,
                    net_eps_basic_neg3fy, net_eps_basic_neg4fy, net_eps_basic_neg5fy, eps_est_avg_rev_pct_fy1e_1w,
                    eps_est_avg_rev_pct_fy1e_1m, eps_est_avg_rev_pct_fy1e_3m, eps_est_avg_rev_pct_fy1e_6m,
                    eps_est_avg_rev_pct_fy1e_1y, div_yield_neg2fyind, div_yield_neg3fyind, div_yield_neg4fyind,
                    div_yield_neg5fyind, ebitda_est_avg_ntm, ebitda_est_avg_fy1e, eps_gaap_est_avg_ntm,
                    eps_gaap_est_avg_fy1e, eps_gaap_est_avg_rev_pct_fy1e_1m, eps_gaap_est_avg_rev_pct_fy1e_3m,
                    eps_gaap_est_avg_rev_pct_fy1e_6m, eps_gaap_est_avg_rev_pct_fy1e_1y, eps_norm_est_num_fy1e,
                    price_target_1w_ago, price_target_1m_ago, price_target_3m_ago, price_target_6m_ago,
                    price_target_mtd_ago, price_target_qtd_ago, price_target_1y_ago, price_target_num_3m_ago,
                    price_target_num_6m_ago, price_target_num_ytd_ago, price_target_num_1y_ago, price_target_num_1w_ago,
                    price_target_num_1m_ago, price_target_num_mtd_ago, price_target_num_qtd_ago,
                    price_target_high_1w_ago, price_target_high_1m_ago, price_target_high_6m_ago,
                    price_target_high_mtd_ago, price_target_high_3m_ago, price_target_high_qtd_ago,
                    price_target_high_1y_ago, price_target_high_ytd_ago, price_target_low_1w_ago,
                    price_target_low_1m_ago, price_target_low_3m_ago, price_target_low_6m_ago, price_target_low_mtd_ago,
                    price_target_low_qtd_ago, price_target_low_ytd_ago, price_target_low_1y_ago,
                    price_target_median_1w_ago, price_target_median_1m_ago, price_target_median_3m_ago,
                    price_target_median_6m_ago, price_target_median_mtd_ago, price_target_median_qtd_ago,
                    price_target_median_ytd_ago, price_target_median_1y_ago, basic_eps_cont_ltm, basic_eps_cont_fq,
                    basic_eps_cont_fy, basic_eps_cont_neg1fqfq, basic_eps_cont_neg2fqfq, basic_eps_cont_neg4fqfq,
                    basic_eps_cont_neg3fqfq, basic_eps_cont_neg1fy, basic_eps_cont_neg2fy, basic_eps_cont_neg3fy,
                    basic_eps_cont_neg4fy, eps_adj_fq, eps_adj_neg1fqfq, eps_adj_neg3fqfq, eps_adj_neg4fqfq,
                    eps_adj_neg2fqfq, eps_adj_neg2fy, eps_adj_neg3fy, eps_adj_neg4fy, gross_profit_neg1fqfq,
                    gross_profit_neg3fqfq, gross_profit_neg4fqfq, gross_profit_neg2fqfq, gross_profit_neg1fy,
                    gross_profit_neg2fy, gross_profit_neg3fy, gross_profit_neg4fy, fcf_est_avg_fy1e, fcf_est_avg_fy2e,
                    fcf_est_avg_fy3e, fcf_est_avg_fy4e, fcf_est_avg_fy5e, eps_neg0fyestimate, eps_neg0fyactual,
                    eps_neg0fysurprise_pct, eps_neg1fyestimate, eps_neg1fyactual, eps_neg1fysurprise_pct,
                    eps_neg2fyestimate, eps_neg2fyactual, eps_neg2fysurprise_pct, eps_neg3fyestimate, eps_neg3fyactual,
                    eps_neg3fysurprise_pct, eps_neg4fyactual, eps_neg4fyestimate, eps_neg4fysurprise_pct,
                    eps_neg5fyestimate, eps_neg5fyactual, eps_neg5fysurprise_pct, eps_neg0fqestimate, eps_neg0fqactual,
                    eps_neg0fqsurprise_pct, eps_neg1fqestimate, eps_neg1fqactual, eps_neg1fqsurprise_pct,
                    eps_neg2fqestimate, eps_neg2fqactual, eps_neg2fqsurprise_pct, eps_neg3fqestimate, eps_neg3fqactual,
                    eps_neg3fqsurprise_pct, eps_neg4fqestimate, eps_neg4fqactual, eps_neg4fqsurprise_pct, fcf_ltm,
                    fcf_fq, fcf_neg1fqfq, fcf_neg3fqfq, fcf_neg4fqfq, fcf_neg2fqfq, fcf_fy, fcf_neg1fy, fcf_neg3fy,
                    fcf_neg2fy, fcf_neg4fy, fcf_neg5fy, target_pct_avg, target_pct_med, target_pct_low, target_pct_high,
                    price_target_stddev, price_target_stddev_1w_ago, price_target_stddev_1m_ago,
                    price_target_stddev_3m_ago, price_target_stddev_6m_ago, price_target_stddev_1y_ago,
                    altman_z_score_neg1fy, altman_z_score_neg2fy, altman_z_score_neg3fy, altman_z_score_neg4fy,
                    altman_z_score_neg5fy, altman_z_score_neg1fqfq, altman_z_score_neg2fqfq, altman_z_score_neg3fqfq,
                    altman_z_score_neg4fqfq, altman_z_score_neg0fyyoyltm, altman_z_score_neg1fyyoyltm,
                    altman_z_score_neg3fyyoyltm, altman_z_score_neg4fyyoyltm, altman_z_score_neg5fyyoyltm,
                    altman_z_score_neg2fyyoyltm, p_e_est_fy2, p_e_est_fy3, p_e_est_fy4, p_e_est_fy5, p_e_neg4fyltm,
                    p_e_neg4fqltm, p_e_3yavgntm, p_e_5yavgntm, eps_norm_est_avg_fq1e, eps_norm_est_avg_fq2e,
                    eps_norm_est_avg_fq3e, eps_norm_est_avg_fq4e, eps_norm_est_avg_fy2e, eps_norm_est_avg_fy3e,
                    eps_norm_est_avg_fy4e, eps_norm_est_avg_fy5e, capital_expenditure_ltm, capital_expenditure_fq,
                    capital_expenditure_fy, capital_expenditure_neg1fqfq, capital_expenditure_neg2fqfq,
                    capital_expenditure_neg3fqfq, capital_expenditure_neg4fqfq, capital_expenditure_neg1fy,
                    capital_expenditure_neg2fy, capital_expenditure_neg4fy, capital_expenditure_neg3fy,
                    capital_expenditure_neg5fy, cff_ltm, cff_fq, cff_fy, cff_neg1fqfq, cff_neg2fqfq, cff_neg3fqfq,
                    cff_neg4fqfq, cff_neg1fy, cff_neg2fy, cff_neg3fy, cff_neg4fy, cfi_ltm, cfi_fq, cfi_fy, cfi_neg1fqfq,
                    cfi_neg2fqfq, cfi_neg3fqfq, cfi_neg4fqfq, cfi_neg1fy, cfi_neg2fy, cfi_neg3fy, cfi_neg5fy,
                    cfi_neg4fy, cfo_ltm, cfo_fq, cfo_fy, cfo_neg1fqfq, cfo_neg2fqfq, cfo_neg4fqfq, cfo_neg3fqfq,
                    cfo_neg1fy, cfo_neg2fy, cfo_neg3fy, cfo_neg4fy, cfo_neg5fy, dividend_per_share_fq,
                    dividend_per_share_fy, dividend_per_share_neg1fqfq, dividend_per_share_neg2fqfq,
                    dividend_per_share_neg3fqfq, dividend_per_share_neg4fqfq, dividend_per_share_neg1fy,
                    dividend_per_share_neg2fy, dividend_per_share_neg3fy, dividend_per_share_neg4fy,
                    dividend_per_share_neg5fy, enterprise_value_neg1fq, enterprise_value_neg2fq,
                    enterprise_value_neg3fq, enterprise_value_neg4fq, enterprise_value_neg1fy, enterprise_value_neg2fy,
                    enterprise_value_neg3fy, enterprise_value_neg4fy, enterprise_value_neg5fy, volatility_1m,
                    volatility_3m, volatility_6m, volatility_1y, eps_gaap_est_avg_rev_pct_fy1e_1w,
                    eps_gaap_est_avg_rev_pct_fy1e_mtd, eps_gaap_est_avg_rev_pct_fy1e_qtd,
                    eps_gaap_est_avg_rev_pct_fy1e_ytd, price_target_stddev_mtd_ago, price_target_stddev_qtd_ago,
                    price_target_stddev_ytd_ago, ebit_neg0fqestimate, ebit_neg1fqestimate, ebit_neg2fqestimate,
                    ebit_neg3fqestimate, ebit_neg4fqestimate, ebit_neg0fqactual, ebit_neg1fqactual, ebit_neg2fqactual,
                    ebit_neg3fqactual, ebit_neg4fqactual, ebit_neg0fyactual, ebit_neg1fyactual, ebit_neg2fyactual,
                    ebit_neg3fyactual, ebit_neg4fyactual, ebit_neg5fyactual, ebit_neg0fyestimate, ebit_neg1fyestimate,
                    ebit_neg2fyestimate, ebit_neg3fyestimate, ebit_neg4fyestimate, ebit_neg5fyestimate,
                    ebit_neg0fqsurprise_pct, ebit_neg1fqsurprise_pct, ebit_neg2fqsurprise_pct, ebit_neg3fqsurprise_pct,
                    ebit_neg4fqsurprise_pct, ebit_neg0fysurprise_pct, ebit_neg1fysurprise_pct, ebit_neg2fysurprise_pct,
                    ebit_neg3fysurprise_pct, ebit_neg4fysurprise_pct, ebit_neg5fysurprise_pct, ebitda_neg0fqestimate,
                    ebitda_neg1fqestimate, ebitda_neg2fqestimate, ebitda_neg3fqestimate, ebitda_neg4fqestimate,
                    ebitda_neg0fyestimate, ebitda_neg1fyestimate, ebitda_neg2fyestimate, ebitda_neg3fyestimate,
                    ebitda_neg4fyestimate, ebitda_neg5fyestimate, ebitda_neg0fqactual, ebitda_neg1fqactual,
                    ebitda_neg2fqactual, ebitda_neg3fqactual, ebitda_neg4fqactual, ebitda_neg0fyactual,
                    ebitda_neg1fyactual, ebitda_neg2fyactual, ebitda_neg3fyactual, ebitda_neg4fyactual,
                    ebitda_neg5fyactual, ebitda_neg0fqsurprise_pct, ebitda_neg1fqsurprise_pct,
                    ebitda_neg2fqsurprise_pct, ebitda_neg3fqsurprise_pct, ebitda_neg4fqsurprise_pct,
                    ebitda_neg0fysurprise_pct, ebitda_neg1fysurprise_pct, ebitda_neg2fysurprise_pct,
                    ebitda_neg3fysurprise_pct, ebitda_neg4fysurprise_pct, ebitda_neg5fysurprise_pct, price_1d_ago,
                    price_ytd_ago, price_mtd_ago, sales_neg0fqestimate, sales_neg0fqactual, sales_neg0fqsurprise_pct,
                    sales_neg1fqestimate, sales_neg1fqactual, sales_neg1fqsurprise_pct, sales_neg2fqestimate,
                    sales_neg2fqactual, sales_neg2fqsurprise_pct, sales_neg3fqestimate, sales_neg3fqactual,
                    sales_neg3fqsurprise_pct, sales_neg4fqestimate, sales_neg4fqactual, sales_neg4fqsurprise_pct,
                    sales_neg0fyestimate, sales_neg0fyactual, sales_neg0fysurprise_pct, sales_neg1fyestimate,
                    sales_neg1fyactual, sales_neg1fysurprise_pct, sales_neg2fyestimate, sales_neg2fyactual,
                    sales_neg2fysurprise_pct, sales_neg3fyestimate, sales_neg3fyactual, sales_neg3fysurprise_pct,
                    sales_neg4fyestimate, sales_neg4fyactual, sales_neg4fysurprise_pct, sales_neg5fyestimate,
                    sales_neg5fyactual, sales_neg5fysurprise_pct, fcf_per_share_ltm, fcf_per_share_fq, fcf_per_share_fy,
                    fcf_per_share_neg1fqfq, fcf_per_share_neg2fqfq, fcf_per_share_neg3fqfq, fcf_per_share_neg4fqfq,
                    fcf_per_share_neg1fy, fcf_per_share_neg2fy, fcf_per_share_neg3fy, fcf_per_share_neg4fy,
                    fcf_per_share_neg5fy, issuance_common_stock_ltm, issuance_common_stock_fq, issuance_common_stock_fy,
                    issuance_common_stock_neg1fqfq, issuance_common_stock_neg2fqfq, issuance_common_stock_neg3fqfq,
                    issuance_common_stock_neg4fqfq, issuance_common_stock_neg1fy, issuance_common_stock_neg2fy,
                    issuance_common_stock_neg3fy, issuance_common_stock_neg4fy, issuance_common_stock_neg5fy,
                    shrs_out_neg1fq, shrs_out_neg2fq, shrs_out_neg3fq, shrs_out_neg4fq, shrs_out_neg2fy,
                    shrs_out_neg3fy, shrs_out_neg4fy, shrs_out_neg5fy, shrs_out_3yavg, shrs_out_5yavg,
                    repurchase_common_stock_ltm, repurchase_common_stock_fq, repurchase_common_stock_fy,
                    repurchase_common_stock_neg1fqfq, repurchase_common_stock_neg2fqfq,
                    repurchase_common_stock_neg3fqfq, repurchase_common_stock_neg4fqfq, repurchase_common_stock_neg1fy,
                    repurchase_common_stock_neg2fy, repurchase_common_stock_neg3fy, repurchase_common_stock_neg4fy,
                    repurchase_common_stock_neg5fy, repurchase_common_stock_3yavgfq, repurchase_common_stock_5yavgfq,
                    peg_ntm, peg_neg1fy, peg_neg2fy, peg_neg3fy, peg_neg4fy, peg_neg5fy, peg_3yavg, peg_5yavg,
                    gross_profit_margin_pct_fq, gross_profit_margin_pct_neg1fqfq, gross_profit_margin_pct_neg2fqfq,
                    gross_profit_margin_pct_neg3fqfq, gross_profit_margin_pct_neg4fqfq, gross_profit_margin_pct_neg1fy,
                    gross_profit_margin_pct_neg2fy, gross_profit_margin_pct_neg3fy, gross_profit_margin_pct_neg4fy,
                    gross_profit_margin_pct_neg5fy, gross_profit_margin_pct_3yavgfq, gross_profit_margin_pct_5yavgfq,
                    market_cap_neg1fq, market_cap_neg2fq, market_cap_neg3fq, market_cap_neg4fq, market_cap_neg1fy,
                    market_cap_neg2fy, market_cap_neg3fy, market_cap_neg4fy, market_cap_3yavg, market_cap_5yavg,
                    enterprise_value_3yavg, enterprise_value_5yavg, tot_return_pct_cagr_5y, tot_return_pct_cagr_1y,
                    total_return_1d, total_return_5d, total_return_1w, total_return_1m, total_return_3m,
                    total_return_6m, total_return_1y, total_return_3y, total_return_mtd, total_return_qtd,
                    total_return_2025, total_return_2024, total_return_2023, total_return_2022, total_return_2021)
SELECT NULLIF(TRIM(s."Ticker"), '')                                                                 AS ticker,
       NULLIF(TRIM(s."ISIN"), '')                                                                   AS isin,
       NULLIF(TRIM(s."Name"), '')                                                                   AS name,
       NULLIF(TRIM(s."Description"), '')                                                            AS description,
       COALESCE(NULLIF(TRIM(s."Trading Region"), ''), 'n/a')                AS trading_region,
       COALESCE(NULLIF(TRIM(s."Trading Country"), ''), 'n/a')                                       AS trading_country,
       COALESCE(NULLIF(TRIM(s."Exchange"), ''), 'n/a')                                              AS exchange,
       COALESCE(NULLIF(TRIM(s."Unit"), ''), 'n/a')                                                  AS unit,
       COALESCE(NULLIF(TRIM(s."Region"), ''), 'n/a')                        AS region,
       COALESCE(NULLIF(TRIM(s."Country"), ''), 'n/a')                       AS country,
       COALESCE(NULLIF(TRIM(s."Sector"), ''), 'n/a')                                                AS sector,
       COALESCE(NULLIF(TRIM(s."Industry"), ''), 'n/a')                                              AS industry,
       COALESCE(NULLIF(TRIM(s."Style Class"), ''), 'n/a')                                           AS style_class,
       COALESCE(NULLIF(TRIM(s."Size Class"), ''), 'n/a')                                            AS size_class,
       text_to_date_safe(s."Last Updated")                                                          AS last_updated,
       text_to_date_safe(s."Income Statement Report Date")                                          AS income_statement_report_date,
       COALESCE(NULLIF(TRIM(s."FY End"), ''), 'n/a')                                                AS fy_end,
       text_to_date_safe(s."Next Earnings")                                                         AS next_earnings,
       COALESCE(NULLIF(TRIM(s."Next Earnings (When)"), ''), 'n/a')                                  AS next_earnings_when,
       COALESCE(NULLIF(TRIM(s."Next Earnings (Status)"), ''), 'n/a')                                AS next_earnings_status,
       parse_fiscal_year_end_date(s."FY End")                                                       AS fy_end_date,
       calculate_next_income_statement_report_date(text_to_date_safe(s."Income Statement Report Date"),
                                                   derive_earnings_report_frequency(
		                                                   text_to_date_safe(s."Income Statement Report Date"),
		                                                   parse_fiscal_year_end_date(s."FY End"))) AS next_income_statement_report_date,
       calculate_next_fy_end_date(parse_fiscal_year_end_date(s."FY End"))                           AS next_fy_end_date,
       (text_to_date_safe(s."Next Earnings") - CURRENT_DATE)                AS days_to_earnings,
       (CURRENT_DATE - text_to_date_safe(s."Income Statement Report Date")) AS earnings_report_recency,
       calculate_expected_report_date(text_to_date_safe(s."Income Statement Report Date"),
                                      derive_earnings_report_frequency(
		                                      text_to_date_safe(s."Income Statement Report Date"),
		                                      parse_fiscal_year_end_date(s."FY End")))              AS expected_report_date,
       COALESCE(NULLIF(TRIM(s."Dividend Record (Currency)"), ''), 'n/a')                            AS dividend_record_currency,
       text_to_numeric_safe(s."Dividend Record (Amount)")                                           AS dividend_record_amount,
       COALESCE(NULLIF(TRIM(s."Dividend Record (Frequency)"), ''),
                'n/a')                                                                              AS dividend_record_frequency,
       text_to_numeric_safe(s."Dividend Streak")                                                    AS dividend_streak,
       text_to_date_safe(s."Dividend Record (Announce Date)")                                       AS dividend_record_announce_date,
       text_to_date_safe(s."Dividend Record (Payable Date)")                                        AS dividend_record_payable_date,
       text_to_date_safe(s."Dividend Record (Record Date)")                                         AS dividend_record_record_date,
       text_to_date_safe(s."Dividend Record (Ex Date)")                                             AS dividend_record_ex_date,
       text_to_numeric_safe(s."Market Cap")                                                         AS market_cap,
       text_to_numeric_safe(s."Enterprise Value")                                                   AS enterprise_value,
       text_to_numeric_safe(s."Last Price")                                                         AS last_price,
       text_to_numeric_safe(s."Price Target (YTD Ago)")                                             AS price_target_ytd_ago,
       text_to_numeric_safe(s."Total Return (YTD)")                                                 AS total_return_ytd,
       text_to_numeric_safe(s."Price Target")                                                       AS price_target,
       text_to_numeric_safe(s."Price Target - Low")                                                 AS price_target_low,
       text_to_numeric_safe(s."Price Target - Median")                                              AS price_target_median,
       text_to_numeric_safe(s."Price Target - High")                                                AS price_target_high,
       text_to_numeric_safe(s."Price Target - #")                                                   AS price_target_num,
       text_to_numeric_safe(s."P/E (NTM)")                                                          AS p_e_ntm,
       text_to_numeric_safe(s."P/E (LTM)")                                                          AS p_e_ltm,
       text_to_numeric_safe(s."Altman Z-Score (FY)")                                                AS altman_z_score_fy,
       text_to_numeric_safe(s."Altman Z-Score (FQ)")                                                AS altman_z_score_fq,
       text_to_numeric_safe(s."Altman Z-Score (LTM)")                                               AS altman_z_score_ltm,
       text_to_numeric_safe(s."Beta (1Y)")                                                          AS beta_1y,
       text_to_numeric_safe(s."Beta (2Y)")                                                          AS beta_2y,
       text_to_numeric_safe(s."Beta (5Y)")                                                          AS beta_5y,
       text_to_numeric_safe(s."Analyst Rating")                                                     AS analyst_rating,
       COALESCE(text_to_numeric_safe(s."# Strong Sell Ratings"), 0)::INT                            AS num_strong_sell_ratings,
       COALESCE(text_to_numeric_safe(s."# Strong Buys Ratings"), 0)::INT                            AS num_strong_buys_ratings,
       COALESCE(text_to_numeric_safe(s."# Hold Ratings"), 0)::INT                                   AS num_hold_ratings,
       COALESCE(text_to_numeric_safe(s."# Buys Ratings"), 0)::INT                                   AS num_buys_ratings,
       COALESCE(text_to_numeric_safe(s."# Sell Ratings"), 0)::INT                                   AS num_sell_ratings,
       COALESCE(text_to_numeric_safe(s."# No Opinion Ratings"), 0)::INT                             AS num_no_opinion_ratings,
       COALESCE(text_to_numeric_safe(s."Market Cap (Country R)"), 0)::INT                           AS market_cap_country_r,
       text_to_numeric_safe(s."Tot. Return %/CAGR (3Y)")                                            AS tot_return_pct_cagr_3y,
       text_to_numeric_safe(s."Tot. Return %/CAGR (10Y)")                                           AS tot_return_pct_cagr_10y,
       text_to_numeric_safe(s."Total Return (5Y)")                                                  AS total_return_5y,
       text_to_numeric_safe(s."Total Return (10Y)")                                                 AS total_return_10y,
       text_to_numeric_safe(s."Volume (Shrs)")                                                      AS volume_shrs,
       text_to_numeric_safe(s."Dividend Per Share (LTM)")                                           AS dividend_per_share_ltm,
       text_to_numeric_safe(s."Div Yield (Ind)")                                                    AS div_yield_ind,
       text_to_numeric_safe(s."Div Yield (LTM)")                                                    AS div_yield_ltm,
       text_to_numeric_safe(s."Gross Profit Margin % (FY)")                                         AS gross_profit_margin_pct_fy,
       text_to_numeric_safe(s."Gross Profit Margin % (LTM)")                                        AS gross_profit_margin_pct_ltm,
       text_to_numeric_safe(s."EPS Norm - Est Avg (NTM)")                                           AS eps_norm_est_avg_ntm,
       text_to_numeric_safe(s."EPS/Adj. (-1FY)")                                                    AS eps_adj_neg1fy,
       text_to_numeric_safe(s."EPS/Adj. (FY)")                                                      AS eps_adj_fy,
       text_to_numeric_safe(s."EPS/Adj. (LTM)")                                                     AS eps_adj_ltm,
       text_to_numeric_safe(s."EPS Norm - Est Avg (FY1E)")                                          AS eps_norm_est_avg_fy1e,
       text_to_numeric_safe(s."Buyback Yield (LTM)")                                                AS buyback_yield_ltm,
       text_to_numeric_safe(s."Return on Assets (ROA) % (LTM)")                                     AS return_on_assets_roa_pct_ltm,
       text_to_numeric_safe(s."Return on Assets (ROA) % (FY)")                                      AS return_on_assets_roa_pct_fy,
       text_to_numeric_safe(s."Div Yield (-1FYInd)")                                                AS div_yield_neg1fyind,
       text_to_numeric_safe(s."P/B (LTM)")                                                          AS p_b_ltm,
       text_to_numeric_safe(s."P/B (-1FY)")                                                         AS p_b_neg1fy,
       text_to_numeric_safe(s."P/B (5YAVG)")                                                        AS p_b_5yavg,
       text_to_numeric_safe(s."Div Yield (TTM)")                                                    AS div_yield_ttm,
       text_to_numeric_safe(s."Div Yield (NTM)")                                                    AS div_yield_ntm,
       text_to_numeric_safe(s."Div Yield (5YAVGLTM)")                                               AS div_yield_5yavgltm,
       text_to_numeric_safe(s."Price Chg. % (3M)")                                                  AS price_chg_pct_3m,
       text_to_numeric_safe(s."1-Day %")                                                            AS one_day_pct,
       text_to_numeric_safe(s."Price (5D Ago)")                                                     AS price_5d_ago,
       text_to_numeric_safe(s."Price (1W Ago)")                                                     AS price_1w_ago,
       text_to_numeric_safe(s."Price (1M Ago)")                                                     AS price_1m_ago,
       text_to_numeric_safe(s."Price (3M Ago)")                                                     AS price_3m_ago,
       text_to_numeric_safe(s."Price (6M Ago)")                                                     AS price_6m_ago,
       text_to_numeric_safe(s."Price (1Y Ago)")                                                     AS price_1y_ago,
       text_to_numeric_safe(s."Price (3Y Ago)")                                                     AS price_3y_ago,
       text_to_numeric_safe(s."Price (5Y Ago)")                                                     AS price_5y_ago,
       text_to_numeric_safe(s."Price (QTD Ago)")                                                    AS price_qtd_ago,
       text_to_numeric_safe(s."Rel. Volume")                                                        AS rel_volume,
       ROUND(text_to_numeric_safe(s."Shrs Out") / 1000000.0, 2)             AS shrs_out,
       ROUND(text_to_numeric_safe(s."Shrs Out (-1FY)") / 1000000.0, 2)      AS shrs_out_neg1fy,
       text_to_numeric_safe(s."Common Dividends Paid (LTM)")                                        AS common_dividends_paid_ltm,
       text_to_numeric_safe(s."Common Dividends Paid (FY)")                                         AS common_dividends_paid_fy,
       text_to_numeric_safe(s."EV/Sales (LTM)")                                                     AS ev_sales_ltm,
       text_to_numeric_safe(s."EV/Sales (NTM)")                                                     AS ev_sales_ntm,
       text_to_numeric_safe(s."EV/Sales (-1FYLTM)")                                                 AS ev_sales_neg1fyltm,
       text_to_numeric_safe(s."EV/Sales (-2FYLTM)")                                                 AS ev_sales_neg2fyltm,
       text_to_numeric_safe(s."EV/Sales (-3FYLTM)")                                                 AS ev_sales_neg3fyltm,
       text_to_numeric_safe(s."EV/Sales (3YAVGLTM)")                                                AS ev_sales_3yavgltm,
       text_to_numeric_safe(s."EV/Sales (-1FQLTM)")                                                 AS ev_sales_neg1fqltm,
       text_to_numeric_safe(s."EV/Sales (-2FQLTM)")                                                 AS ev_sales_neg2fqltm,
       text_to_numeric_safe(s."EV/Sales (-3FQLTM)")                                                 AS ev_sales_neg3fqltm,
       text_to_numeric_safe(s."EV/Sales (-4FQLTM)")                                                 AS ev_sales_neg4fqltm,
       text_to_numeric_safe(s."52W High/Adj")                                                       AS w_52high_adj,
       text_to_numeric_safe(s."52W Low/Adj")                                                        AS w_52low_adj,
       text_to_numeric_safe(s."EMA (20D)")                                                          AS ema_20d,
       text_to_numeric_safe(s."EMA (50D)")                                                          AS ema_50d,
       text_to_numeric_safe(s."EMA (100D)")                                                         AS ema_100d,
       text_to_numeric_safe(s."EMA (250D)")                                                         AS ema_250d,
       text_to_numeric_safe(s."EV/EBITDA (LTM)")                                                    AS ev_ebitda_ltm,
       text_to_numeric_safe(s."EV/EBITDA (NTM)")                                                    AS ev_ebitda_ntm,
       text_to_numeric_safe(s."EV/EBITDA (-1FYLTM)")                                                AS ev_ebitda_neg1fyltm,
       text_to_numeric_safe(s."EV/EBITDA (-1FQLTM)")                                                AS ev_ebitda_neg1fqltm,
       text_to_numeric_safe(s."EV/EBITDA (3YAVGLTM)")                                               AS ev_ebitda_3yavgltm,
       text_to_numeric_safe(s."EV/EBITDA (EST FY1)")                                                AS ev_ebitda_est_fy1,
       text_to_numeric_safe(s."P/E (EST FY1)")                                                      AS p_e_est_fy1,
       text_to_numeric_safe(s."P/E (-1FYLTM)")                                                      AS p_e_neg1fyltm,
       text_to_numeric_safe(s."P/E (-2FYLTM)")                                                      AS p_e_neg2fyltm,
       text_to_numeric_safe(s."P/E (-3FYLTM)")                                                      AS p_e_neg3fyltm,
       text_to_numeric_safe(s."P/E (3YAVGLTM)")                                                     AS p_e_3yavgltm,
       text_to_numeric_safe(s."P/E (-1FQLTM)")                                                      AS p_e_neg1fqltm,
       text_to_numeric_safe(s."P/E (-2FQLTM)")                                                      AS p_e_neg2fqltm,
       text_to_numeric_safe(s."P/E (-3FQLTM)")                                                      AS p_e_neg3fqltm,
       text_to_numeric_safe(s."P/E (5YAVGLTM)")                                                     AS p_e_5yavgltm,
       text_to_numeric_safe(s."P/E (-0FQQoQLTM)")                                                   AS p_e_neg0fqqoqltm,
       text_to_numeric_safe(s."P/E (-0FYYoYLTM)")                                                   AS p_e_neg0fyyoyltm,
       text_to_numeric_safe(s."P/E (-1FYYoYLTM)")                                                   AS p_e_neg1fyyoyltm,
       text_to_numeric_safe(s."P/E (-0FQYoYLTM)")                                                   AS p_e_neg0fqyoyltm,
       text_to_numeric_safe(s."Full Time Employees (FQ)")                                           AS full_time_employees_fq,
       text_to_numeric_safe(s."Full Time Employees (FY)")                                           AS full_time_employees_fy,
       text_to_numeric_safe(s."Full Time Employees (-1FY)")                                         AS full_time_employees_neg1fy,
       text_to_numeric_safe(s."Full Time Employees (-2FY)")                                         AS full_time_employees_neg2fy,
       text_to_numeric_safe(s."Full Time Employees (-3FY)")                                         AS full_time_employees_neg3fy,
       text_to_numeric_safe(s."Avg Employees (5YAVGFY)")                                            AS avg_employees_5yavgfy,
       text_to_numeric_safe(s."Net EPS - Basic (LTM)")                                              AS net_eps_basic_ltm,
       text_to_numeric_safe(s."Net EPS - Basic (FQ)")                                               AS net_eps_basic_fq,
       text_to_numeric_safe(s."Net EPS - Basic (FY)")                                               AS net_eps_basic_fy,
       text_to_numeric_safe(s."Net EPS - Basic (-1FQFQ)")                                           AS net_eps_basic_neg1fqfq,
       text_to_numeric_safe(s."Net EPS - Basic (-2FQFQ)")                                           AS net_eps_basic_neg2fqfq,
       text_to_numeric_safe(s."Net EPS - Basic (-3FQFQ)")                                           AS net_eps_basic_neg3fqfq,
       text_to_numeric_safe(s."Net EPS - Basic (-4FQFQ)")                                           AS net_eps_basic_neg4fqfq,
       text_to_numeric_safe(s."Net EPS - Basic (-1FY)")                                             AS net_eps_basic_neg1fy,
       text_to_numeric_safe(s."Net EPS - Basic (-2FY)")                                             AS net_eps_basic_neg2fy,
       text_to_numeric_safe(s."Net EPS - Basic (-3FY)")                                             AS net_eps_basic_neg3fy,
       text_to_numeric_safe(s."Net EPS - Basic (-4FY)")                                             AS net_eps_basic_neg4fy,
       text_to_numeric_safe(s."Net EPS - Basic (-5FY)")                                             AS net_eps_basic_neg5fy,
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 1W)")                                      AS eps_est_avg_rev_pct_fy1e_1w,
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 1M)")                                      AS eps_est_avg_rev_pct_fy1e_1m,
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 3M)")                                      AS eps_est_avg_rev_pct_fy1e_3m,
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 6M)")                                      AS eps_est_avg_rev_pct_fy1e_6m,
       text_to_numeric_safe(s."EPS Est Avg Rev % (FY1E - 1Y)")                                      AS eps_est_avg_rev_pct_fy1e_1y,
       text_to_numeric_safe(s."Div Yield (-2FYInd)")                                                AS div_yield_neg2fyind,
       text_to_numeric_safe(s."Div Yield (-3FYInd)")                                                AS div_yield_neg3fyind,
       text_to_numeric_safe(s."Div Yield (-4FYInd)")                                                AS div_yield_neg4fyind,
       text_to_numeric_safe(s."Div Yield (-5FYInd)")                                                AS div_yield_neg5fyind,
       text_to_numeric_safe(s."EBITDA - Est Avg (NTM)")                                             AS ebitda_est_avg_ntm,
       text_to_numeric_safe(s."EBITDA - Est Avg (FY1E)")                                            AS ebitda_est_avg_fy1e,
       text_to_numeric_safe(s."EPS GAAP - Est Avg (NTM)")                                           AS eps_gaap_est_avg_ntm,
       text_to_numeric_safe(s."EPS GAAP - Est Avg (FY1E)")                                          AS eps_gaap_est_avg_fy1e,
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 1M)")                                 AS eps_gaap_est_avg_rev_pct_fy1e_1m,
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 3M)")                                 AS eps_gaap_est_avg_rev_pct_fy1e_3m,
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 6M)")                                 AS eps_gaap_est_avg_rev_pct_fy1e_6m,
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 1Y)")                                 AS eps_gaap_est_avg_rev_pct_fy1e_1y,
       text_to_numeric_safe(s."EPS Norm - Est # (FY1E)")                                            AS eps_norm_est_num_fy1e,
       text_to_numeric_safe(s."Price Target (1W Ago)")                                              AS price_target_1w_ago,
       text_to_numeric_safe(s."Price Target (1M Ago)")                                              AS price_target_1m_ago,
       text_to_numeric_safe(s."Price Target (3M Ago)")                                              AS price_target_3m_ago,
       text_to_numeric_safe(s."Price Target (6M Ago)")                                              AS price_target_6m_ago,
       text_to_numeric_safe(s."Price Target (MTD Ago)")                                             AS price_target_mtd_ago,
       text_to_numeric_safe(s."Price Target (QTD Ago)")                                             AS price_target_qtd_ago,
       text_to_numeric_safe(s."Price Target (1Y Ago)")                                              AS price_target_1y_ago,
       text_to_numeric_safe(s."Price Target - # (3M Ago)")                                          AS price_target_num_3m_ago,
       text_to_numeric_safe(s."Price Target - # (6M Ago)")                                          AS price_target_num_6m_ago,
       text_to_numeric_safe(s."Price Target - # (YTD Ago)")                                         AS price_target_num_ytd_ago,
       text_to_numeric_safe(s."Price Target - # (1Y Ago)")                                          AS price_target_num_1y_ago,
       text_to_numeric_safe(s."Price Target - # (1W Ago)")                                          AS price_target_num_1w_ago,
       text_to_numeric_safe(s."Price Target - # (1M Ago)")                                          AS price_target_num_1m_ago,
       text_to_numeric_safe(s."Price Target - # (MTD Ago)")                                         AS price_target_num_mtd_ago,
       text_to_numeric_safe(s."Price Target - # (QTD Ago)")                                         AS price_target_num_qtd_ago,
       text_to_numeric_safe(s."Price Target - High (1W Ago)")                                       AS price_target_high_1w_ago,
       text_to_numeric_safe(s."Price Target - High (1M Ago)")                                       AS price_target_high_1m_ago,
       text_to_numeric_safe(s."Price Target - High (6M Ago)")                                       AS price_target_high_6m_ago,
       text_to_numeric_safe(s."Price Target - High (MTD Ago)")                                      AS price_target_high_mtd_ago,
       text_to_numeric_safe(s."Price Target - High (3M Ago)")                                       AS price_target_high_3m_ago,
       text_to_numeric_safe(s."Price Target - High (QTD Ago)")                                      AS price_target_high_qtd_ago,
       text_to_numeric_safe(s."Price Target - High (1Y Ago)")                                       AS price_target_high_1y_ago,
       text_to_numeric_safe(s."Price Target - High (YTD Ago)")                                      AS price_target_high_ytd_ago,
       text_to_numeric_safe(s."Price Target - Low (1W Ago)")                                        AS price_target_low_1w_ago,
       text_to_numeric_safe(s."Price Target - Low (1M Ago)")                                        AS price_target_low_1m_ago,
       text_to_numeric_safe(s."Price Target - Low (3M Ago)")                                        AS price_target_low_3m_ago,
       text_to_numeric_safe(s."Price Target - Low (6M Ago)")                                        AS price_target_low_6m_ago,
       text_to_numeric_safe(s."Price Target - Low (MTD Ago)")                                       AS price_target_low_mtd_ago,
       text_to_numeric_safe(s."Price Target - Low (QTD Ago)")                                       AS price_target_low_qtd_ago,
       text_to_numeric_safe(s."Price Target - Low (YTD Ago)")                                       AS price_target_low_ytd_ago,
       text_to_numeric_safe(s."Price Target - Low (1Y Ago)")                                        AS price_target_low_1y_ago,
       text_to_numeric_safe(s."Price Target - Median (1W Ago)")                                     AS price_target_median_1w_ago,
       text_to_numeric_safe(s."Price Target - Median (1M Ago)")                                     AS price_target_median_1m_ago,
       text_to_numeric_safe(s."Price Target - Median (3M Ago)")                                     AS price_target_median_3m_ago,
       text_to_numeric_safe(s."Price Target - Median (6M Ago)")                                     AS price_target_median_6m_ago,
       text_to_numeric_safe(s."Price Target - Median (MTD Ago)")                                    AS price_target_median_mtd_ago,
       text_to_numeric_safe(s."Price Target - Median (QTD Ago)")                                    AS price_target_median_qtd_ago,
       text_to_numeric_safe(s."Price Target - Median (YTD Ago)")                                    AS price_target_median_ytd_ago,
       text_to_numeric_safe(s."Price Target - Median (1Y Ago)")                                     AS price_target_median_1y_ago,
       text_to_numeric_safe(s."Basic EPS - Cont (LTM)")                                             AS basic_eps_cont_ltm,
       text_to_numeric_safe(s."Basic EPS - Cont (FQ)")                                              AS basic_eps_cont_fq,
       text_to_numeric_safe(s."Basic EPS - Cont (FY)")                                              AS basic_eps_cont_fy,
       text_to_numeric_safe(s."Basic EPS - Cont (-1FQFQ)")                                          AS basic_eps_cont_neg1fqfq,
       text_to_numeric_safe(s."Basic EPS - Cont (-2FQFQ)")                                          AS basic_eps_cont_neg2fqfq,
       text_to_numeric_safe(s."Basic EPS - Cont (-4FQFQ)")                                          AS basic_eps_cont_neg4fqfq,
       text_to_numeric_safe(s."Basic EPS - Cont (-3FQFQ)")                                          AS basic_eps_cont_neg3fqfq,
       text_to_numeric_safe(s."Basic EPS - Cont (-1FY)")                                            AS basic_eps_cont_neg1fy,
       text_to_numeric_safe(s."Basic EPS - Cont (-2FY)")                                            AS basic_eps_cont_neg2fy,
       text_to_numeric_safe(s."Basic EPS - Cont (-3FY)")                                            AS basic_eps_cont_neg3fy,
       text_to_numeric_safe(s."Basic EPS - Cont (-4FY)")                                            AS basic_eps_cont_neg4fy,
       text_to_numeric_safe(s."EPS/Adj. (FQ)")                                                      AS eps_adj_fq,
       text_to_numeric_safe(s."EPS/Adj. (-1FQFQ)")                                                  AS eps_adj_neg1fqfq,
       text_to_numeric_safe(s."EPS/Adj. (-3FQFQ)")                                                  AS eps_adj_neg3fqfq,
       text_to_numeric_safe(s."EPS/Adj. (-4FQFQ)")                                                  AS eps_adj_neg4fqfq,
       text_to_numeric_safe(s."EPS/Adj. (-2FQFQ)")                                                  AS eps_adj_neg2fqfq,
       text_to_numeric_safe(s."EPS/Adj. (-2FY)")                                                    AS eps_adj_neg2fy,
       text_to_numeric_safe(s."EPS/Adj. (-3FY)")                                                    AS eps_adj_neg3fy,
       text_to_numeric_safe(s."EPS/Adj. (-4FY)")                                                    AS eps_adj_neg4fy,
       text_to_numeric_safe(s."Gross Profit (-1FQFQ)")                                              AS gross_profit_neg1fqfq,
       text_to_numeric_safe(s."Gross Profit (-3FQFQ)")                                              AS gross_profit_neg3fqfq,
       text_to_numeric_safe(s."Gross Profit (-4FQFQ)")                                              AS gross_profit_neg4fqfq,
       text_to_numeric_safe(s."Gross Profit (-2FQFQ)")                                              AS gross_profit_neg2fqfq,
       text_to_numeric_safe(s."Gross Profit (-1FY)")                                                AS gross_profit_neg1fy,
       text_to_numeric_safe(s."Gross Profit (-2FY)")                                                AS gross_profit_neg2fy,
       text_to_numeric_safe(s."Gross Profit (-3FY)")                                                AS gross_profit_neg3fy,
       text_to_numeric_safe(s."Gross Profit (-4FY)")                                                AS gross_profit_neg4fy,
       text_to_numeric_safe(s."FCF - Est Avg (FY1E)")                                               AS fcf_est_avg_fy1e,
       text_to_numeric_safe(s."FCF - Est Avg (FY2E)")                                               AS fcf_est_avg_fy2e,
       text_to_numeric_safe(s."FCF - Est Avg (FY3E)")                                               AS fcf_est_avg_fy3e,
       text_to_numeric_safe(s."FCF - Est Avg (FY4E)")                                               AS fcf_est_avg_fy4e,
       text_to_numeric_safe(s."FCF - Est Avg (FY5E)")                                               AS fcf_est_avg_fy5e,
       text_to_numeric_safe(s."EPS (-0FYEstimate)")                                                 AS eps_neg0fyestimate,
       text_to_numeric_safe(s."EPS (-0FYActual)")                                                   AS eps_neg0fyactual,
       text_to_numeric_safe(s."EPS (-0FYSurprise %)")                                               AS eps_neg0fysurprise_pct,
       text_to_numeric_safe(s."EPS (-1FYEstimate)")                                                 AS eps_neg1fyestimate,
       text_to_numeric_safe(s."EPS (-1FYActual)")                                                   AS eps_neg1fyactual,
       text_to_numeric_safe(s."EPS (-1FYSurprise %)")                                               AS eps_neg1fysurprise_pct,
       text_to_numeric_safe(s."EPS (-2FYEstimate)")                                                 AS eps_neg2fyestimate,
       text_to_numeric_safe(s."EPS (-2FYActual)")                                                   AS eps_neg2fyactual,
       text_to_numeric_safe(s."EPS (-2FYSurprise %)")                                               AS eps_neg2fysurprise_pct,
       text_to_numeric_safe(s."EPS (-3FYEstimate)")                                                 AS eps_neg3fyestimate,
       text_to_numeric_safe(s."EPS (-3FYActual)")                                                   AS eps_neg3fyactual,
       text_to_numeric_safe(s."EPS (-3FYSurprise %)")                                               AS eps_neg3fysurprise_pct,
       text_to_numeric_safe(s."EPS (-4FYActual)")                                                   AS eps_neg4fyactual,
       text_to_numeric_safe(s."EPS (-4FYEstimate)")                                                 AS eps_neg4fyestimate,
       text_to_numeric_safe(s."EPS (-4FYSurprise %)")                                               AS eps_neg4fysurprise_pct,
       text_to_numeric_safe(s."EPS (-5FYEstimate)")                                                 AS eps_neg5fyestimate,
       text_to_numeric_safe(s."EPS (-5FYActual)")                                                   AS eps_neg5fyactual,
       text_to_numeric_safe(s."EPS (-5FYSurprise %)")                                               AS eps_neg5fysurprise_pct,
       text_to_numeric_safe(s."EPS (-0FQEstimate)")                                                 AS eps_neg0fqestimate,
       text_to_numeric_safe(s."EPS (-0FQActual)")                                                   AS eps_neg0fqactual,
       text_to_numeric_safe(s."EPS (-0FQSurprise %)")                                               AS eps_neg0fqsurprise_pct,
       text_to_numeric_safe(s."EPS (-1FQEstimate)")                                                 AS eps_neg1fqestimate,
       text_to_numeric_safe(s."EPS (-1FQActual)")                                                   AS eps_neg1fqactual,
       text_to_numeric_safe(s."EPS (-1FQSurprise %)")                                               AS eps_neg1fqsurprise_pct,
       text_to_numeric_safe(s."EPS (-2FQEstimate)")                                                 AS eps_neg2fqestimate,
       text_to_numeric_safe(s."EPS (-2FQActual)")                                                   AS eps_neg2fqactual,
       text_to_numeric_safe(s."EPS (-2FQSurprise %)")                                               AS eps_neg2fqsurprise_pct,
       text_to_numeric_safe(s."EPS (-3FQEstimate)")                                                 AS eps_neg3fqestimate,
       text_to_numeric_safe(s."EPS (-3FQActual)")                                                   AS eps_neg3fqactual,
       text_to_numeric_safe(s."EPS (-3FQSurprise %)")                                               AS eps_neg3fqsurprise_pct,
       text_to_numeric_safe(s."EPS (-4FQEstimate)")                                                 AS eps_neg4fqestimate,
       text_to_numeric_safe(s."EPS (-4FQActual)")                                                   AS eps_neg4fqactual,
       text_to_numeric_safe(s."EPS (-4FQSurprise %)")                                               AS eps_neg4fqsurprise_pct,
       text_to_numeric_safe(s."FCF (LTM)")                                                          AS fcf_ltm,
       text_to_numeric_safe(s."FCF (FQ)")                                                           AS fcf_fq,
       text_to_numeric_safe(s."FCF (-1FQFQ)")                                                       AS fcf_neg1fqfq,
       text_to_numeric_safe(s."FCF (-3FQFQ)")                                                       AS fcf_neg3fqfq,
       text_to_numeric_safe(s."FCF (-4FQFQ)")                                                       AS fcf_neg4fqfq,
       text_to_numeric_safe(s."FCF (-2FQFQ)")                                                       AS fcf_neg2fqfq,
       text_to_numeric_safe(s."FCF (FY)")                                                           AS fcf_fy,
       text_to_numeric_safe(s."FCF (-1FY)")                                                         AS fcf_neg1fy,
       text_to_numeric_safe(s."FCF (-3FY)")                                                         AS fcf_neg3fy,
       text_to_numeric_safe(s."FCF (-2FY)")                                                         AS fcf_neg2fy,
       text_to_numeric_safe(s."FCF (-4FY)")                                                         AS fcf_neg4fy,
       text_to_numeric_safe(s."FCF (-5FY)")                                                         AS fcf_neg5fy,
       text_to_numeric_safe(s."Target % (Avg)")                                                     AS target_pct_avg,
       text_to_numeric_safe(s."Target % (Med)")                                                     AS target_pct_med,
       text_to_numeric_safe(s."Target % (Low)")                                                     AS target_pct_low,
       text_to_numeric_safe(s."Target % (High)")                                                    AS target_pct_high,
       text_to_numeric_safe(s."Price Target - StdDev")                                              AS price_target_stddev,
       text_to_numeric_safe(s."Price Target - StdDev (1W Ago)")                                     AS price_target_stddev_1w_ago,
       text_to_numeric_safe(s."Price Target - StdDev (1M Ago)")                                     AS price_target_stddev_1m_ago,
       text_to_numeric_safe(s."Price Target - StdDev (3M Ago)")                                     AS price_target_stddev_3m_ago,
       text_to_numeric_safe(s."Price Target - StdDev (6M Ago)")                                     AS price_target_stddev_6m_ago,
       text_to_numeric_safe(s."Price Target - StdDev (1Y Ago)")                                     AS price_target_stddev_1y_ago,
       text_to_numeric_safe(s."Altman Z-Score (-1FY)")                                              AS altman_z_score_neg1fy,
       text_to_numeric_safe(s."Altman Z-Score (-2FY)")                                              AS altman_z_score_neg2fy,
       text_to_numeric_safe(s."Altman Z-Score (-3FY)")                                              AS altman_z_score_neg3fy,
       text_to_numeric_safe(s."Altman Z-Score (-4FY)")                                              AS altman_z_score_neg4fy,
       text_to_numeric_safe(s."Altman Z-Score (-5FY)")                                              AS altman_z_score_neg5fy,
       text_to_numeric_safe(s."Altman Z-Score (-1FQFQ)")                                            AS altman_z_score_neg1fqfq,
       text_to_numeric_safe(s."Altman Z-Score (-2FQFQ)")                                            AS altman_z_score_neg2fqfq,
       text_to_numeric_safe(s."Altman Z-Score (-3FQFQ)")                                            AS altman_z_score_neg3fqfq,
       text_to_numeric_safe(s."Altman Z-Score (-4FQFQ)")                                            AS altman_z_score_neg4fqfq,
       text_to_numeric_safe(s."Altman Z-Score (-0FYYoYLTM)")                                        AS altman_z_score_neg0fyyoyltm,
       text_to_numeric_safe(s."Altman Z-Score (-1FYYoYLTM)")                                        AS altman_z_score_neg1fyyoyltm,
       text_to_numeric_safe(s."Altman Z-Score (-3FYYoYLTM)")                                        AS altman_z_score_neg3fyyoyltm,
       text_to_numeric_safe(s."Altman Z-Score (-4FYYoYLTM)")                                        AS altman_z_score_neg4fyyoyltm,
       text_to_numeric_safe(s."Altman Z-Score (-5FYYoYLTM)")                                        AS altman_z_score_neg5fyyoyltm,
       text_to_numeric_safe(s."Altman Z-Score (-2FYYoYLTM)")                                        AS altman_z_score_neg2fyyoyltm,
       text_to_numeric_safe(s."P/E (EST FY2)")                                                      AS p_e_est_fy2,
       text_to_numeric_safe(s."P/E (EST FY3)")                                                      AS p_e_est_fy3,
       text_to_numeric_safe(s."P/E (EST FY4)")                                                      AS p_e_est_fy4,
       text_to_numeric_safe(s."P/E (EST FY5)")                                                      AS p_e_est_fy5,
       text_to_numeric_safe(s."P/E (-4FYLTM)")                                                      AS p_e_neg4fyltm,
       text_to_numeric_safe(s."P/E (-4FQLTM)")                                                      AS p_e_neg4fqltm,
       text_to_numeric_safe(s."P/E (3YAVGNTM)")                                                     AS p_e_3yavgntm,
       text_to_numeric_safe(s."P/E (5YAVGNTM)")                                                     AS p_e_5yavgntm,
       text_to_numeric_safe(s."EPS Norm - Est Avg (FQ1E)")                                          AS eps_norm_est_avg_fq1e,
       text_to_numeric_safe(s."EPS Norm - Est Avg (FQ2E)")                                          AS eps_norm_est_avg_fq2e,
       text_to_numeric_safe(s."EPS Norm - Est Avg (FQ3E)")                                          AS eps_norm_est_avg_fq3e,
       text_to_numeric_safe(s."EPS Norm - Est Avg (FQ4E)")                                          AS eps_norm_est_avg_fq4e,
       text_to_numeric_safe(s."EPS Norm - Est Avg (FY2E)")                                          AS eps_norm_est_avg_fy2e,
       text_to_numeric_safe(s."EPS Norm - Est Avg (FY3E)")                                          AS eps_norm_est_avg_fy3e,
       text_to_numeric_safe(s."EPS Norm - Est Avg (FY4E)")                                          AS eps_norm_est_avg_fy4e,
       text_to_numeric_safe(s."EPS Norm - Est Avg (FY5E)")                                          AS eps_norm_est_avg_fy5e,
       text_to_numeric_safe(s."Capital Expenditure (LTM)")                                          AS capital_expenditure_ltm,
       text_to_numeric_safe(s."Capital Expenditure (FQ)")                                           AS capital_expenditure_fq,
       text_to_numeric_safe(s."Capital Expenditure (FY)")                                           AS capital_expenditure_fy,
       text_to_numeric_safe(s."Capital Expenditure (-1FQFQ)")                                       AS capital_expenditure_neg1fqfq,
       text_to_numeric_safe(s."Capital Expenditure (-2FQFQ)")                                       AS capital_expenditure_neg2fqfq,
       text_to_numeric_safe(s."Capital Expenditure (-3FQFQ)")                                       AS capital_expenditure_neg3fqfq,
       text_to_numeric_safe(s."Capital Expenditure (-4FQFQ)")                                       AS capital_expenditure_neg4fqfq,
       text_to_numeric_safe(s."Capital Expenditure (-1FY)")                                         AS capital_expenditure_neg1fy,
       text_to_numeric_safe(s."Capital Expenditure (-2FY)")                                         AS capital_expenditure_neg2fy,
       text_to_numeric_safe(s."Capital Expenditure (-4FY)")                                         AS capital_expenditure_neg4fy,
       text_to_numeric_safe(s."Capital Expenditure (-3FY)")                                         AS capital_expenditure_neg3fy,
       text_to_numeric_safe(s."Capital Expenditure (-5FY)")                                         AS capital_expenditure_neg5fy,
       text_to_numeric_safe(s."CFF (LTM)")                                                          AS cff_ltm,
       text_to_numeric_safe(s."CFF (FQ)")                                                           AS cff_fq,
       text_to_numeric_safe(s."CFF (FY)")                                                           AS cff_fy,
       text_to_numeric_safe(s."CFF (-1FQFQ)")                                                       AS cff_neg1fqfq,
       text_to_numeric_safe(s."CFF (-2FQFQ)")                                                       AS cff_neg2fqfq,
       text_to_numeric_safe(s."CFF (-3FQFQ)")                                                       AS cff_neg3fqfq,
       text_to_numeric_safe(s."CFF (-4FQFQ)")                                                       AS cff_neg4fqfq,
       text_to_numeric_safe(s."CFF (-1FY)")                                                         AS cff_neg1fy,
       text_to_numeric_safe(s."CFF (-2FY)")                                                         AS cff_neg2fy,
       text_to_numeric_safe(s."CFF (-3FY)")                                                         AS cff_neg3fy,
       text_to_numeric_safe(s."CFF (-4FY)")                                                         AS cff_neg4fy,
       text_to_numeric_safe(s."CFI (LTM)")                                                          AS cfi_ltm,
       text_to_numeric_safe(s."CFI (FQ)")                                                           AS cfi_fq,
       text_to_numeric_safe(s."CFI (FY)")                                                           AS cfi_fy,
       text_to_numeric_safe(s."CFI (-1FQFQ)")                                                       AS cfi_neg1fqfq,
       text_to_numeric_safe(s."CFI (-2FQFQ)")                                                       AS cfi_neg2fqfq,
       text_to_numeric_safe(s."CFI (-3FQFQ)")                                                       AS cfi_neg3fqfq,
       text_to_numeric_safe(s."CFI (-4FQFQ)")                                                       AS cfi_neg4fqfq,
       text_to_numeric_safe(s."CFI (-1FY)")                                                         AS cfi_neg1fy,
       text_to_numeric_safe(s."CFI (-2FY)")                                                         AS cfi_neg2fy,
       text_to_numeric_safe(s."CFI (-3FY)")                                                         AS cfi_neg3fy,
       text_to_numeric_safe(s."CFI (-5FY)")                                                         AS cfi_neg5fy,
       text_to_numeric_safe(s."CFI (-4FY)")                                                         AS cfi_neg4fy,
       text_to_numeric_safe(s."CFO (LTM)")                                                          AS cfo_ltm,
       text_to_numeric_safe(s."CFO (FQ)")                                                           AS cfo_fq,
       text_to_numeric_safe(s."CFO (FY)")                                                           AS cfo_fy,
       text_to_numeric_safe(s."CFO (-1FQFQ)")                                                       AS cfo_neg1fqfq,
       text_to_numeric_safe(s."CFO (-2FQFQ)")                                                       AS cfo_neg2fqfq,
       text_to_numeric_safe(s."CFO (-4FQFQ)")                                                       AS cfo_neg4fqfq,
       text_to_numeric_safe(s."CFO (-3FQFQ)")                                                       AS cfo_neg3fqfq,
       text_to_numeric_safe(s."CFO (-1FY)")                                                         AS cfo_neg1fy,
       text_to_numeric_safe(s."CFO (-2FY)")                                                         AS cfo_neg2fy,
       text_to_numeric_safe(s."CFO (-3FY)")                                                         AS cfo_neg3fy,
       text_to_numeric_safe(s."CFO (-4FY)")                                                         AS cfo_neg4fy,
       text_to_numeric_safe(s."CFO (-5FY)")                                                         AS cfo_neg5fy,
       text_to_numeric_safe(s."Dividend Per Share (FQ)")                                            AS dividend_per_share_fq,
       text_to_numeric_safe(s."Dividend Per Share (FY)")                                            AS dividend_per_share_fy,
       text_to_numeric_safe(s."Dividend Per Share (-1FQFQ)")                                        AS dividend_per_share_neg1fqfq,
       text_to_numeric_safe(s."Dividend Per Share (-2FQFQ)")                                        AS dividend_per_share_neg2fqfq,
       text_to_numeric_safe(s."Dividend Per Share (-3FQFQ)")                                        AS dividend_per_share_neg3fqfq,
       text_to_numeric_safe(s."Dividend Per Share (-4FQFQ)")                                        AS dividend_per_share_neg4fqfq,
       text_to_numeric_safe(s."Dividend Per Share (-1FY)")                                          AS dividend_per_share_neg1fy,
       text_to_numeric_safe(s."Dividend Per Share (-2FY)")                                          AS dividend_per_share_neg2fy,
       text_to_numeric_safe(s."Dividend Per Share (-3FY)")                                          AS dividend_per_share_neg3fy,
       text_to_numeric_safe(s."Dividend Per Share (-4FY)")                                          AS dividend_per_share_neg4fy,
       text_to_numeric_safe(s."Dividend Per Share (-5FY)")                                          AS dividend_per_share_neg5fy,
       text_to_numeric_safe(s."Enterprise Value (-1FQ)")                                            AS enterprise_value_neg1fq,
       text_to_numeric_safe(s."Enterprise Value (-2FQ)")                                            AS enterprise_value_neg2fq,
       text_to_numeric_safe(s."Enterprise Value (-3FQ)")                                            AS enterprise_value_neg3fq,
       text_to_numeric_safe(s."Enterprise Value (-4FQ)")                                            AS enterprise_value_neg4fq,
       text_to_numeric_safe(s."Enterprise Value (-1FY)")                                            AS enterprise_value_neg1fy,
       text_to_numeric_safe(s."Enterprise Value (-2FY)")                                            AS enterprise_value_neg2fy,
       text_to_numeric_safe(s."Enterprise Value (-3FY)")                                            AS enterprise_value_neg3fy,
       text_to_numeric_safe(s."Enterprise Value (-4FY)")                                            AS enterprise_value_neg4fy,
       text_to_numeric_safe(s."Enterprise Value (-5FY)")                                            AS enterprise_value_neg5fy,
       text_to_numeric_safe(s."Volatility (1M)")                                                    AS volatility_1m,
       text_to_numeric_safe(s."Volatility (3M)")                                                    AS volatility_3m,
       text_to_numeric_safe(s."Volatility (6M)")                                                    AS volatility_6m,
       text_to_numeric_safe(s."Volatility (1Y)")                                                    AS volatility_1y,
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - 1W)")                                 AS eps_gaap_est_avg_rev_pct_fy1e_1w,
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - MTD)")                                AS eps_gaap_est_avg_rev_pct_fy1e_mtd,
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - QTD)")                                AS eps_gaap_est_avg_rev_pct_fy1e_qtd,
       text_to_numeric_safe(s."EPS GAAP Est Avg Rev % (FY1E - YTD)")                                AS eps_gaap_est_avg_rev_pct_fy1e_ytd,
       text_to_numeric_safe(s."Price Target - StdDev (MTD Ago)")                                    AS price_target_stddev_mtd_ago,
       text_to_numeric_safe(s."Price Target - StdDev (QTD Ago)")                                    AS price_target_stddev_qtd_ago,
       text_to_numeric_safe(s."Price Target - StdDev (YTD Ago)")            AS price_target_stddev_ytd_ago,
       text_to_numeric_safe(s."EBIT (-0FQEstimate)")                        AS ebit_neg0fqestimate,
       text_to_numeric_safe(s."EBIT (-1FQEstimate)")                        AS ebit_neg1fqestimate,
       text_to_numeric_safe(s."EBIT (-2FQEstimate)")                        AS ebit_neg2fqestimate,
       text_to_numeric_safe(s."EBIT (-3FQEstimate)")                        AS ebit_neg3fqestimate,
       text_to_numeric_safe(s."EBIT (-4FQEstimate)")                        AS ebit_neg4fqestimate,
       text_to_numeric_safe(s."EBIT (-0FQActual)")                          AS ebit_neg0fqactual,
       text_to_numeric_safe(s."EBIT (-1FQActual)")                          AS ebit_neg1fqactual,
       text_to_numeric_safe(s."EBIT (-2FQActual)")                          AS ebit_neg2fqactual,
       text_to_numeric_safe(s."EBIT (-3FQActual)")                          AS ebit_neg3fqactual,
       text_to_numeric_safe(s."EBIT (-4FQActual)")                          AS ebit_neg4fqactual,
       text_to_numeric_safe(s."EBIT (-0FYActual)")                          AS ebit_neg0fyactual,
       text_to_numeric_safe(s."EBIT (-1FYActual)")                          AS ebit_neg1fyactual,
       text_to_numeric_safe(s."EBIT (-2FYActual)")                          AS ebit_neg2fyactual,
       text_to_numeric_safe(s."EBIT (-3FYActual)")                          AS ebit_neg3fyactual,
       text_to_numeric_safe(s."EBIT (-4FYActual)")                          AS ebit_neg4fyactual,
       text_to_numeric_safe(s."EBIT (-5FYActual)")                          AS ebit_neg5fyactual,
       text_to_numeric_safe(s."EBIT (-0FYEstimate)")                        AS ebit_neg0fyestimate,
       text_to_numeric_safe(s."EBIT (-1FYEstimate)")                        AS ebit_neg1fyestimate,
       text_to_numeric_safe(s."EBIT (-2FYEstimate)")                        AS ebit_neg2fyestimate,
       text_to_numeric_safe(s."EBIT (-3FYEstimate)")                        AS ebit_neg3fyestimate,
       text_to_numeric_safe(s."EBIT (-4FYEstimate)")                        AS ebit_neg4fyestimate,
       text_to_numeric_safe(s."EBIT (-5FYEstimate)")                        AS ebit_neg5fyestimate,
       text_to_numeric_safe(s."EBIT (-0FQSurprise %)")                      AS ebit_neg0fqsurprise_pct,
       text_to_numeric_safe(s."EBIT (-1FQSurprise %)")                      AS ebit_neg1fqsurprise_pct,
       text_to_numeric_safe(s."EBIT (-2FQSurprise %)")                      AS ebit_neg2fqsurprise_pct,
       text_to_numeric_safe(s."EBIT (-3FQSurprise %)")                      AS ebit_neg3fqsurprise_pct,
       text_to_numeric_safe(s."EBIT (-4FQSurprise %)")                      AS ebit_neg4fqsurprise_pct,
       text_to_numeric_safe(s."EBIT (-0FYSurprise %)")                      AS ebit_neg0fysurprise_pct,
       text_to_numeric_safe(s."EBIT (-1FYSurprise %)")                      AS ebit_neg1fysurprise_pct,
       text_to_numeric_safe(s."EBIT (-2FYSurprise %)")                      AS ebit_neg2fysurprise_pct,
       text_to_numeric_safe(s."EBIT (-3FYSurprise %)")                      AS ebit_neg3fysurprise_pct,
       text_to_numeric_safe(s."EBIT (-4FYSurprise %)")                      AS ebit_neg4fysurprise_pct,
       text_to_numeric_safe(s."EBIT (-5FYSurprise %)")                      AS ebit_neg5fysurprise_pct,
       text_to_numeric_safe(s."EBITDA (-0FQEstimate)")                      AS ebitda_neg0fqestimate,
       text_to_numeric_safe(s."EBITDA (-1FQEstimate)")                      AS ebitda_neg1fqestimate,
       text_to_numeric_safe(s."EBITDA (-2FQEstimate)")                      AS ebitda_neg2fqestimate,
       text_to_numeric_safe(s."EBITDA (-3FQEstimate)")                      AS ebitda_neg3fqestimate,
       text_to_numeric_safe(s."EBITDA (-4FQEstimate)")                      AS ebitda_neg4fqestimate,
       text_to_numeric_safe(s."EBITDA (-0FYEstimate)")                      AS ebitda_neg0fyestimate,
       text_to_numeric_safe(s."EBITDA (-1FYEstimate)")                      AS ebitda_neg1fyestimate,
       text_to_numeric_safe(s."EBITDA (-2FYEstimate)")                      AS ebitda_neg2fyestimate,
       text_to_numeric_safe(s."EBITDA (-3FYEstimate)")                      AS ebitda_neg3fyestimate,
       text_to_numeric_safe(s."EBITDA (-4FYEstimate)")                      AS ebitda_neg4fyestimate,
       text_to_numeric_safe(s."EBITDA (-5FYEstimate)")                      AS ebitda_neg5fyestimate,
       text_to_numeric_safe(s."EBITDA (-0FQActual)")                        AS ebitda_neg0fqactual,
       text_to_numeric_safe(s."EBITDA (-1FQActual)")                        AS ebitda_neg1fqactual,
       text_to_numeric_safe(s."EBITDA (-2FQActual)")                        AS ebitda_neg2fqactual,
       text_to_numeric_safe(s."EBITDA (-3FQActual)")                        AS ebitda_neg3fqactual,
       text_to_numeric_safe(s."EBITDA (-4FQActual)")                        AS ebitda_neg4fqactual,
       text_to_numeric_safe(s."EBITDA (-0FYActual)")                        AS ebitda_neg0fyactual,
       text_to_numeric_safe(s."EBITDA (-1FYActual)")                        AS ebitda_neg1fyactual,
       text_to_numeric_safe(s."EBITDA (-2FYActual)")                        AS ebitda_neg2fyactual,
       text_to_numeric_safe(s."EBITDA (-3FYActual)")                        AS ebitda_neg3fyactual,
       text_to_numeric_safe(s."EBITDA (-4FYActual)")                        AS ebitda_neg4fyactual,
       text_to_numeric_safe(s."EBITDA (-5FYActual)")                        AS ebitda_neg5fyactual,
       text_to_numeric_safe(s."EBITDA (-0FQSurprise %)")                    AS ebitda_neg0fqsurprise_pct,
       text_to_numeric_safe(s."EBITDA (-1FQSurprise %)")                    AS ebitda_neg1fqsurprise_pct,
       text_to_numeric_safe(s."EBITDA (-2FQSurprise %)")                    AS ebitda_neg2fqsurprise_pct,
       text_to_numeric_safe(s."EBITDA (-3FQSurprise %)")                    AS ebitda_neg3fqsurprise_pct,
       text_to_numeric_safe(s."EBITDA (-4FQSurprise %)")                    AS ebitda_neg4fqsurprise_pct,
       text_to_numeric_safe(s."EBITDA (-0FYSurprise %)")                    AS ebitda_neg0fysurprise_pct,
       text_to_numeric_safe(s."EBITDA (-1FYSurprise %)")                    AS ebitda_neg1fysurprise_pct,
       text_to_numeric_safe(s."EBITDA (-2FYSurprise %)")                    AS ebitda_neg2fysurprise_pct,
       text_to_numeric_safe(s."EBITDA (-3FYSurprise %)")                    AS ebitda_neg3fysurprise_pct,
       text_to_numeric_safe(s."EBITDA (-4FYSurprise %)")                    AS ebitda_neg4fysurprise_pct,
       text_to_numeric_safe(s."EBITDA (-5FYSurprise %)")                    AS ebitda_neg5fysurprise_pct,
       text_to_numeric_safe(s."Price (1D Ago)")                             AS price_1d_ago,
       text_to_numeric_safe(s."Price (YTD Ago)")                            AS price_ytd_ago,
       text_to_numeric_safe(s."Price (MTD Ago)")                            AS price_mtd_ago,
       text_to_numeric_safe(s."Sales (-0FQEstimate)")                       AS sales_neg0fqestimate,
       text_to_numeric_safe(s."Sales (-0FQActual)")                         AS sales_neg0fqactual,
       text_to_numeric_safe(s."Sales (-0FQSurprise %)")                     AS sales_neg0fqsurprise_pct,
       text_to_numeric_safe(s."Sales (-1FQEstimate)")                       AS sales_neg1fqestimate,
       text_to_numeric_safe(s."Sales (-1FQActual)")                         AS sales_neg1fqactual,
       text_to_numeric_safe(s."Sales (-1FQSurprise %)")                     AS sales_neg1fqsurprise_pct,
       text_to_numeric_safe(s."Sales (-2FQEstimate)")                       AS sales_neg2fqestimate,
       text_to_numeric_safe(s."Sales (-2FQActual)")                         AS sales_neg2fqactual,
       text_to_numeric_safe(s."Sales (-2FQSurprise %)")                     AS sales_neg2fqsurprise_pct,
       text_to_numeric_safe(s."Sales (-3FQEstimate)")                       AS sales_neg3fqestimate,
       text_to_numeric_safe(s."Sales (-3FQActual)")                         AS sales_neg3fqactual,
       text_to_numeric_safe(s."Sales (-3FQSurprise %)")                     AS sales_neg3fqsurprise_pct,
       text_to_numeric_safe(s."Sales (-4FQEstimate)")                       AS sales_neg4fqestimate,
       text_to_numeric_safe(s."Sales (-4FQActual)")                         AS sales_neg4fqactual,
       text_to_numeric_safe(s."Sales (-4FQSurprise %)")                     AS sales_neg4fqsurprise_pct,
       text_to_numeric_safe(s."Sales (-0FYEstimate)")                       AS sales_neg0fyestimate,
       text_to_numeric_safe(s."Sales (-0FYActual)")                         AS sales_neg0fyactual,
       text_to_numeric_safe(s."Sales (-0FYSurprise %)")                     AS sales_neg0fysurprise_pct,
       text_to_numeric_safe(s."Sales (-1FYEstimate)")                       AS sales_neg1fyestimate,
       text_to_numeric_safe(s."Sales (-1FYActual)")                         AS sales_neg1fyactual,
       text_to_numeric_safe(s."Sales (-1FYSurprise %)")                     AS sales_neg1fysurprise_pct,
       text_to_numeric_safe(s."Sales (-2FYEstimate)")                       AS sales_neg2fyestimate,
       text_to_numeric_safe(s."Sales (-2FYActual)")                         AS sales_neg2fyactual,
       text_to_numeric_safe(s."Sales (-2FYSurprise %)")                     AS sales_neg2fysurprise_pct,
       text_to_numeric_safe(s."Sales (-3FYEstimate)")                       AS sales_neg3fyestimate,
       text_to_numeric_safe(s."Sales (-3FYActual)")                         AS sales_neg3fyactual,
       text_to_numeric_safe(s."Sales (-3FYSurprise %)")                     AS sales_neg3fysurprise_pct,
       text_to_numeric_safe(s."Sales (-4FYEstimate)")                       AS sales_neg4fyestimate,
       text_to_numeric_safe(s."Sales (-4FYActual)")                         AS sales_neg4fyactual,
       text_to_numeric_safe(s."Sales (-4FYSurprise %)")                     AS sales_neg4fysurprise_pct,
       text_to_numeric_safe(s."Sales (-5FYEstimate)")                       AS sales_neg5fyestimate,
       text_to_numeric_safe(s."Sales (-5FYActual)")                         AS sales_neg5fyactual,
       text_to_numeric_safe(s."Sales (-5FYSurprise %)")                     AS sales_neg5fysurprise_pct,
       text_to_numeric_safe(s."FCF / Share (LTM)")                          AS fcf_per_share_ltm,
       text_to_numeric_safe(s."FCF / Share (FQ)")                           AS fcf_per_share_fq,
       text_to_numeric_safe(s."FCF / Share (FY)")                           AS fcf_per_share_fy,
       text_to_numeric_safe(s."FCF / Share (-1FQFQ)")                       AS fcf_per_share_neg1fqfq,
       text_to_numeric_safe(s."FCF / Share (-2FQFQ)")                       AS fcf_per_share_neg2fqfq,
       text_to_numeric_safe(s."FCF / Share (-3FQFQ)")                       AS fcf_per_share_neg3fqfq,
       text_to_numeric_safe(s."FCF / Share (-4FQFQ)")                       AS fcf_per_share_neg4fqfq,
       text_to_numeric_safe(s."FCF / Share (-1FY)")                         AS fcf_per_share_neg1fy,
       text_to_numeric_safe(s."FCF / Share (-2FY)")                         AS fcf_per_share_neg2fy,
       text_to_numeric_safe(s."FCF / Share (-3FY)")                         AS fcf_per_share_neg3fy,
       text_to_numeric_safe(s."FCF / Share (-4FY)")                         AS fcf_per_share_neg4fy,
       text_to_numeric_safe(s."FCF / Share (-5FY)")                         AS fcf_per_share_neg5fy,
       text_to_numeric_safe(s."Issuance of Common Stock (LTM)")             AS issuance_common_stock_ltm,
       text_to_numeric_safe(s."Issuance of Common Stock (FQ)")              AS issuance_common_stock_fq,
       text_to_numeric_safe(s."Issuance of Common Stock (FY)")              AS issuance_common_stock_fy,
       text_to_numeric_safe(s."Issuance of Common Stock (-1FQFQ)")          AS issuance_common_stock_neg1fqfq,
       text_to_numeric_safe(s."Issuance of Common Stock (-2FQFQ)")          AS issuance_common_stock_neg2fqfq,
       text_to_numeric_safe(s."Issuance of Common Stock (-3FQFQ)")          AS issuance_common_stock_neg3fqfq,
       text_to_numeric_safe(s."Issuance of Common Stock (-4FQFQ)")          AS issuance_common_stock_neg4fqfq,
       text_to_numeric_safe(s."Issuance of Common Stock (-1FY)")            AS issuance_common_stock_neg1fy,
       text_to_numeric_safe(s."Issuance of Common Stock (-2FY)")            AS issuance_common_stock_neg2fy,
       text_to_numeric_safe(s."Issuance of Common Stock (-3FY)")            AS issuance_common_stock_neg3fy,
       text_to_numeric_safe(s."Issuance of Common Stock (-4FY)")            AS issuance_common_stock_neg4fy,
       text_to_numeric_safe(s."Issuance of Common Stock (-5FY)")            AS issuance_common_stock_neg5fy,
       ROUND(text_to_numeric_safe(s."Shrs Out (-1FQ)") / 1000000.0, 2)      AS shrs_out_neg1fq,
       ROUND(text_to_numeric_safe(s."Shrs Out (-2FQ)") / 1000000.0, 2)      AS shrs_out_neg2fq,
       ROUND(text_to_numeric_safe(s."Shrs Out (-3FQ)") / 1000000.0, 2)      AS shrs_out_neg3fq,
       ROUND(text_to_numeric_safe(s."Shrs Out (-4FQ)") / 1000000.0, 2)      AS shrs_out_neg4fq,
       ROUND(text_to_numeric_safe(s."Shrs Out (-2FY)") / 1000000.0, 2)      AS shrs_out_neg2fy,
       ROUND(text_to_numeric_safe(s."Shrs Out (-3FY)") / 1000000.0, 2)      AS shrs_out_neg3fy,
       ROUND(text_to_numeric_safe(s."Shrs Out (-4FY)") / 1000000.0, 2)      AS shrs_out_neg4fy,
       ROUND(text_to_numeric_safe(s."Shrs Out (-5FY)") / 1000000.0, 2)      AS shrs_out_neg5fy,
       ROUND(text_to_numeric_safe(s."Shrs Out (3YAVG)") / 1000000.0, 2)     AS shrs_out_3yavg,
       ROUND(text_to_numeric_safe(s."Shrs Out (5YAVG)") / 1000000.0, 2)     AS shrs_out_5yavg,
       text_to_numeric_safe(s."Repurchase of Common Stock (LTM)")           AS repurchase_common_stock_ltm,
       text_to_numeric_safe(s."Repurchase of Common Stock (FQ)")            AS repurchase_common_stock_fq,
       text_to_numeric_safe(s."Repurchase of Common Stock (FY)")            AS repurchase_common_stock_fy,
       text_to_numeric_safe(s."Repurchase of Common Stock (-1FQFQ)")        AS repurchase_common_stock_neg1fqfq,
       text_to_numeric_safe(s."Repurchase of Common Stock (-2FQFQ)")        AS repurchase_common_stock_neg2fqfq,
       text_to_numeric_safe(s."Repurchase of Common Stock (-3FQFQ)")        AS repurchase_common_stock_neg3fqfq,
       text_to_numeric_safe(s."Repurchase of Common Stock (-4FQFQ)")        AS repurchase_common_stock_neg4fqfq,
       text_to_numeric_safe(s."Repurchase of Common Stock (-1FY)")          AS repurchase_common_stock_neg1fy,
       text_to_numeric_safe(s."Repurchase of Common Stock (-2FY)")          AS repurchase_common_stock_neg2fy,
       text_to_numeric_safe(s."Repurchase of Common Stock (-3FY)")          AS repurchase_common_stock_neg3fy,
       text_to_numeric_safe(s."Repurchase of Common Stock (-4FY)")          AS repurchase_common_stock_neg4fy,
       text_to_numeric_safe(s."Repurchase of Common Stock (-5FY)")          AS repurchase_common_stock_neg5fy,
       text_to_numeric_safe(s."Repurchase of Common Stock (3YAVGFQ)")       AS repurchase_common_stock_3yavgfq,
       text_to_numeric_safe(s."Repurchase of Common Stock (5YAVGFQ)")       AS repurchase_common_stock_5yavgfq,
       text_to_numeric_safe(s."PEG (NTM)")                                  AS peg_ntm,
       text_to_numeric_safe(s."PEG (-1FY)")                                 AS peg_neg1fy,
       text_to_numeric_safe(s."PEG (-2FY)")                                 AS peg_neg2fy,
       text_to_numeric_safe(s."PEG (-3FY)")                                 AS peg_neg3fy,
       text_to_numeric_safe(s."PEG (-4FY)")                                 AS peg_neg4fy,
       text_to_numeric_safe(s."PEG (-5FY)")                                 AS peg_neg5fy,
       text_to_numeric_safe(s."PEG (3YAVG)")                                AS peg_3yavg,
       text_to_numeric_safe(s."PEG (5YAVG)")                                AS peg_5yavg,
       text_to_numeric_safe(s."Gross Profit Margin % (FQ)")                 AS gross_profit_margin_pct_fq,
       text_to_numeric_safe(s."Gross Profit Margin % (-1FQFQ)")             AS gross_profit_margin_pct_neg1fqfq,
       text_to_numeric_safe(s."Gross Profit Margin % (-2FQFQ)")             AS gross_profit_margin_pct_neg2fqfq,
       text_to_numeric_safe(s."Gross Profit Margin % (-3FQFQ)")             AS gross_profit_margin_pct_neg3fqfq,
       text_to_numeric_safe(s."Gross Profit Margin % (-4FQFQ)")             AS gross_profit_margin_pct_neg4fqfq,
       text_to_numeric_safe(s."Gross Profit Margin % (-1FY)")               AS gross_profit_margin_pct_neg1fy,
       text_to_numeric_safe(s."Gross Profit Margin % (-2FY)")               AS gross_profit_margin_pct_neg2fy,
       text_to_numeric_safe(s."Gross Profit Margin % (-3FY)")               AS gross_profit_margin_pct_neg3fy,
       text_to_numeric_safe(s."Gross Profit Margin % (-4FY)")               AS gross_profit_margin_pct_neg4fy,
       text_to_numeric_safe(s."Gross Profit Margin % (-5FY)")               AS gross_profit_margin_pct_neg5fy,
       text_to_numeric_safe(s."Gross Profit Margin % (3YAVGFQ)")            AS gross_profit_margin_pct_3yavgfq,
       text_to_numeric_safe(s."Gross Profit Margin % (5YAVGFQ)")            AS gross_profit_margin_pct_5yavgfq,
       text_to_numeric_safe(s."Market Cap (-1FQ)")                          AS market_cap_neg1fq,
       text_to_numeric_safe(s."Market Cap (-2FQ)")                          AS market_cap_neg2fq,
       text_to_numeric_safe(s."Market Cap (-3FQ)")                          AS market_cap_neg3fq,
       text_to_numeric_safe(s."Market Cap (-4FQ)")                          AS market_cap_neg4fq,
       text_to_numeric_safe(s."Market Cap (-1FY)")                          AS market_cap_neg1fy,
       text_to_numeric_safe(s."Market Cap (-2FY)")                          AS market_cap_neg2fy,
       text_to_numeric_safe(s."Market Cap (-3FY)")                          AS market_cap_neg3fy,
       text_to_numeric_safe(s."Market Cap (-4FY)")                          AS market_cap_neg4fy,
       text_to_numeric_safe(s."Market Cap (3YAVG)")                         AS market_cap_3yavg,
       text_to_numeric_safe(s."Market Cap (5YAVG)")                         AS market_cap_5yavg,
       text_to_numeric_safe(s."Enterprise Value (3YAVG)")                   AS enterprise_value_3yavg,
       text_to_numeric_safe(s."Enterprise Value (5YAVG)")                   AS enterprise_value_5yavg,
       text_to_numeric_safe(s."Tot. Return %/CAGR (5Y)")                    AS tot_return_pct_cagr_5y,
       text_to_numeric_safe(s."Tot. Return %/CAGR (1Y)")                    AS tot_return_pct_cagr_1y,
       text_to_numeric_safe(s."Total Return (1D)")                          AS total_return_1d,
       text_to_numeric_safe(s."Total Return (5D)")                          AS total_return_5d,
       text_to_numeric_safe(s."Total Return (1W)")                          AS total_return_1w,
       text_to_numeric_safe(s."Total Return (1M)")                          AS total_return_1m,
       text_to_numeric_safe(s."Total Return (3M)")                          AS total_return_3m,
       text_to_numeric_safe(s."Total Return (6M)")                          AS total_return_6m,
       text_to_numeric_safe(s."Total Return (1Y)")                          AS total_return_1y,
       text_to_numeric_safe(s."Total Return (3Y)")                          AS total_return_3y,
       text_to_numeric_safe(s."Total Return (MTD)")                         AS total_return_mtd,
       text_to_numeric_safe(s."Total Return (QTD)")                         AS total_return_qtd,
       text_to_numeric_safe(s."Total Return (2025)")                        AS total_return_2025,
       text_to_numeric_safe(s."Total Return (2024)")                        AS total_return_2024,
       text_to_numeric_safe(s."Total Return (2023)")                        AS total_return_2023,
       text_to_numeric_safe(s."Total Return (2022)")                        AS total_return_2022,
       text_to_numeric_safe(s."Total Return (2021)")                        AS total_return_2021


FROM staging_header_buf s;
-- FINAL VALIDATION
-- ===================================================================
\echo 'Final validation...'
SELECT 'Total rows in pml_df:' AS info, COUNT(*) AS count
FROM pml_df;
SELECT 'Rows by Trading Region:' AS info, trading_region, COUNT(*) AS count
FROM pml_df
GROUP BY trading_region
ORDER BY trading_region;
SELECT 'Rows by Sector (top 10):' AS info, sector, COUNT(*) AS count
FROM pml_df
GROUP BY sector
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