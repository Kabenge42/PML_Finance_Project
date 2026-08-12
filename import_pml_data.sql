-- ===================================================================
-- Equities Data Import Script
-- ===================================================================
-- Documentation: See docs/column_mapping_reference.md for column aliases
-- Usage: psql -h localhost -p 5432 -U postgres -d postgres -f import_pml_data.sql
\echo 'Starting pml_df data import...'

-- ===================================================================
-- SESSION-LEVEL TUNING FOR BULK IMPORT
-- ===================================================================
SET search_path = pml
; -- Resolve pml_df / pml_us without schema prefix
SET work_mem = '256MB'
; -- Increase memory for sorting/hashing operations
SET maintenance_work_mem = '512MB'
; -- Increase memory for maintenance operations
SET SYNCHRONOUS_COMMIT = OFF
; -- Defer WAL writes (faster, but less durable during import)
SET checkpoint_completion_target = 0.9
; -- Spread checkpoint I/O over longer period

\echo 'Session tuning applied for bulk import optimization.'

DO
$$
	BEGIN
		RAISE NOTICE 'Import started at %', NOW();
	END
$$
;

-- Show current table status
SELECT 'Current pml_df table row count:' AS status, COUNT(*) AS row_count
FROM pml_df
;

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
	             PARALLEL SAFE
;

-- ===================================================================
-- HELPER FUNCTION: Get Expected Reporting Lag Days
-- ===================================================================
-- Returns the typical number of days between period end and earnings release
CREATE OR REPLACE FUNCTION get_expected_reporting_lag_days(earnings_report_frequency TEXT) RETURNS INTEGER AS
$$
SELECT CASE UPPER(TRIM(COALESCE(earnings_report_frequency, 'QUARTERLY')))
	       WHEN 'QUARTERLY'     THEN 45
	       WHEN 'SEMI-ANNUALLY' THEN 60
	       WHEN 'ANNUALLY'      THEN 90
	                            ELSE 45 END
$$
	LANGUAGE sql IMMUTABLE
	             PARALLEL SAFE
;

-- Converts TEXT to NUMERIC, treating common non-numeric patterns as NULL
CREATE OR REPLACE FUNCTION text_to_numeric_safe(input_text TEXT) RETURNS NUMERIC AS
$$
SELECT CASE
	       WHEN input_text IS NULL OR
	            TRIM(input_text) IN ('', '-', '--', 'N/A', 'NA', 'NULL', 'NONE', 'n/a', 'na', 'null', 'none') THEN NULL
	       WHEN TRIM(input_text) ~ '^-?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
	                                                                                                          THEN TRIM(input_text)::NUMERIC END AS result
$$
	LANGUAGE sql IMMUTABLE
	             PARALLEL SAFE
;

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
	                 STRICT
;

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
	                 STRICT
;

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
		       WHEN 'QUARTERLY'     THEN fy_range_months / 4
		       WHEN 'SEMI-ANNUALLY' THEN fy_range_months / 2
		       WHEN 'ANNUALLY'      THEN fy_range_months
		                            ELSE fy_range_months / 4 END;
END;
$$
	LANGUAGE plpgsql IMMUTABLE
;

-- ===================================================================
-- HELPER FUNCTION: Convert Interval Months to Frequency Text
-- ===================================================================
CREATE OR REPLACE FUNCTION months_to_frequency(interval_months INTEGER) RETURNS TEXT AS
$$
SELECT CASE
	       WHEN interval_months IS NULL THEN 'Quarterly'
	       WHEN interval_months <= 3    THEN 'Quarterly'
	       WHEN interval_months <= 6    THEN 'Semi-Annually'
	                                    ELSE 'Annually' END
$$
	LANGUAGE sql IMMUTABLE
	             PARALLEL SAFE
;

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
	LANGUAGE plpgsql IMMUTABLE
;

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
		                   WHEN 'QUARTERLY'     THEN fy_range_months / 4
		                   WHEN 'SEMI-ANNUALLY' THEN fy_range_months / 2
		                   WHEN 'ANNUALY'       THEN fy_range_months
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
		                             WHEN interval_months = fy_range_months                         THEN 'Full Year'
		                             WHEN next_period = periods_per_year AND periods_per_year > 1   THEN 'Full Year'
		-- Half year for semi-annual mid-year report
		                             WHEN interval_months = fy_range_months / 2 AND next_period = 1 THEN 'Half Year'
		                                                                                            ELSE 'Interim' END;
END;
$$
	LANGUAGE plpgsql IMMUTABLE
;

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
	             PARALLEL SAFE
;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next Fiscal Year End Date
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_next_fy_end_date(fy_end_date DATE) RETURNS DATE AS
$$
SELECT (fy_end_date + make_interval(years => 1))::DATE
$$
	LANGUAGE sql IMMUTABLE
	             STRICT
	             PARALLEL SAFE
;
-- ===================================================================
-- HELPER FUNCTION: Calculate Next Fiscal Quarter Date
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_next_fiscal_quarter_date(income_statement_report_date DATE) RETURNS DATE AS
$$
SELECT (income_statement_report_date + make_interval(months := AGE(current_date, fy_end_date)))::DATE
$$
	LANGUAGE sql IMMUTABLE
	             STRICT
	             PARALLEL SAFE
;

-- ===================================================================
-- HELPER FUNCTION: Calculate Next Fiscal Quarter
-- ===================================================================
CREATE OR REPLACE FUNCTION calculate_next_fiscal_quarter(
	next_earnings                DATE,
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
		ELSIF next_earnings IS NOT NULL THEN
			reference_date := next_earnings;
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
	LANGUAGE plpgsql IMMUTABLE
;

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
	             PARALLEL SAFE
;

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
	             PARALLEL SAFE
;

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
	LANGUAGE plpgsql IMMUTABLE
;

-- ===================================================================
-- COUNTRY CODE -> COUNTRY NAME REFERENCE (ISO 3166-1 alpha-2)
-- ===================================================================
-- Value-to-value mapping. NOTE: a COLLATION cannot translate codes to
-- names; it only affects sort/compare rules. We use a lookup table.
-- Shared by both "Country" and "Trading Country" (the trading-country
-- code set is a strict subset of the country code set).
DROP TABLE IF EXISTS country_ref
;
CREATE TEMP TABLE country_ref
(
	code TEXT PRIMARY KEY,
	name TEXT NOT NULL
)
;

INSERT INTO country_ref (code, name)
VALUES ('AE', 'United Arab Emirates'),
       ('AR', 'Argentina'),
       ('AT', 'Austria'),
       ('AU', 'Australia'),
       ('AZ', 'Azerbaijan'),
       ('BB', 'Barbados'),
       ('BE', 'Belgium'),
       ('BG', 'Bulgaria'),
       ('BH', 'Bahrain'),
       ('BM', 'Bermuda'),
       ('BR', 'Brazil'),
       ('BS', 'Bahamas'),
       ('CA', 'Canada'),
       ('CH', 'Switzerland'),
       ('CL', 'Chile'),
       ('CN', 'China'),
       ('CO', 'Colombia'),
       ('CY', 'Cyprus'),
       ('CZ', 'Czech Republic'),
       ('DE', 'Germany'),
       ('DK', 'Denmark'),
       ('EE', 'Estonia'),
       ('EG', 'Egypt'),
       ('ES', 'Spain'),
       ('FI', 'Finland'),
       ('FR', 'France'),
       ('GB', 'United Kingdom'),
       ('GE', 'Georgia'),
       ('GG', 'Guernsey'),
       ('GH', 'Ghana'),
       ('GI', 'Gibraltar'),
       ('GR', 'Greece'),
       ('HK', 'Hong Kong'),
       ('HR', 'Croatia'),
       ('HU', 'Hungary'),
       ('ID', 'Indonesia'),
       ('IE', 'Ireland'),
       ('IL', 'Israel'),
       ('IM', 'Isle of Man'),
       ('IN', 'India'),
       ('IS', 'Iceland'),
       ('IT', 'Italy'),
       ('JE', 'Jersey'),
       ('JP', 'Japan'),
       ('KE', 'Kenya'),
       ('KH', 'Cambodia'),
       ('KR', 'South Korea'),
       ('KW', 'Kuwait'),
       ('KY', 'Cayman Islands'),
       ('KZ', 'Kazakhstan'),
       ('LI', 'Liechtenstein'),
       ('LT', 'Lithuania'),
       ('LU', 'Luxembourg'),
       ('MA', 'Morocco'),
       ('MC', 'Monaco'),
       ('MD', 'Moldova'),
       ('MO', 'Macau'),
       ('MT', 'Malta'),
       ('MX', 'Mexico'),
       ('MY', 'Malaysia'),
       ('NG', 'Nigeria'),
       ('NL', 'Netherlands'),
       ('NO', 'Norway'),
       ('NZ', 'New Zealand'),
       ('OM', 'Oman'),
       ('PA', 'Panama'),
       ('PE', 'Peru'),
       ('PH', 'Philippines'),
       ('PK', 'Pakistan'),
       ('PL', 'Poland'),
       ('PT', 'Portugal'),
       ('QA', 'Qatar'),
       ('RO', 'Romania'),
       ('RS', 'Serbia'),
       ('SA', 'Saudi Arabia'),
       ('SE', 'Sweden'),
       ('SG', 'Singapore'),
       ('SI', 'Slovenia'),
       ('SK', 'Slovakia'),
       ('TH', 'Thailand'),
       ('TR', 'Turkey'),
       ('TW', 'Taiwan'),
       ('US', 'United States'),
       ('UY', 'Uruguay'),
       ('VG', 'British Virgin Islands'),
       ('VN', 'Vietnam'),
       ('ZA', 'South Africa')
;
-- Extend with the full set of codes present in your CSVs.

-- Resolve a country code to its full name. Falls back to the original
-- input when no mapping exists so unknown codes aren't lost.
CREATE OR REPLACE FUNCTION country_name(code_text TEXT) RETURNS TEXT AS
$$
SELECT COALESCE((SELECT r.name FROM country_ref r WHERE r.code = UPPER(TRIM(code_text))), NULLIF(TRIM(code_text), ''))
$$
	LANGUAGE sql STABLE
;

-- ===================================================================
-- EXCHANGE CODE -> EXCHANGE NAME REFERENCE (vendor / Capital IQ codes)
-- ===================================================================
-- Codes are mixed case in the CSVs (e.g. NasdaqGS), so keys are stored
-- uppercase and lookups normalise with UPPER(TRIM(...)).
DROP TABLE IF EXISTS exchange_ref
;
CREATE TEMP TABLE exchange_ref
(
	code TEXT PRIMARY KEY,
	name TEXT NOT NULL
)
;

INSERT INTO exchange_ref (code, name)
VALUES ('ADX', 'Abu Dhabi Securities Exchange'),
       ('AIM', 'London Stock Exchange AIM'),
       ('ASX', 'Australian Securities Exchange'),
       ('ATSE', 'Athens Stock Exchange'),
       ('BASE', 'Buenos Aires Stock Exchange'),
       ('BATS', 'Cboe BZX Exchange (BATS)'),
       ('BAX', 'Bahrain Bourse'),
       ('BDL', 'Bourse de Luxembourg'),
       ('BELEX', 'Belgrade Stock Exchange'),
       ('BIT', 'Borsa Italiana'),
       ('BME', 'Bolsas y Mercados Espanoles'),
       ('BMV', 'Bolsa Mexicana de Valores'),
       ('BOVESPA', 'B3 - Brasil Bolsa Balcao'),
       ('BSE', 'Bombay Stock Exchange'),
       ('BUL', 'Bulgarian Stock Exchange'),
       ('BUSE', 'Budapest Stock Exchange'),
       ('BVB', 'Bucharest Stock Exchange'),
       ('BVC', 'Bolsa de Valores de Colombia'),
       ('BVL', 'Bolsa de Valores de Lima'),
       ('CASE', 'Egyptian Exchange (Cairo)'),
       ('CBSE', 'Casablanca Stock Exchange'),
       ('CNSX', 'Canadian Securities Exchange'),
       ('CPSE', 'Nasdaq Copenhagen'),
       ('CSE', 'Cyprus Stock Exchange'),
       ('DB', 'Deutsche Boerse (Frankfurt)'),
       ('DFM', 'Dubai Financial Market'),
       ('DSM', 'Qatar Stock Exchange (Doha)'),
       ('ENXTAM', 'Euronext Amsterdam'),
       ('ENXTBR', 'Euronext Brussels'),
       ('ENXTLS', 'Euronext Lisbon'),
       ('ENXTPA', 'Euronext Paris'),
       ('GHSE', 'Ghana Stock Exchange'),
       ('HLSE', 'Nasdaq Helsinki'),
       ('HMSE', 'Hamburg Stock Exchange'),
       ('HOSE', 'Ho Chi Minh Stock Exchange'),
       ('IBSE', 'Borsa Istanbul'),
       ('ICSE', 'Nasdaq Iceland'),
       ('IDX', 'Indonesia Stock Exchange'),
       ('ISE', 'Euronext Dublin (Irish SE)'),
       ('JSE', 'Johannesburg Stock Exchange'),
       ('KASE', 'Pakistan Stock Exchange (Karachi)'),
       ('KLSE', 'Bursa Malaysia'),
       ('KOSDAQ', 'KOSDAQ (Korea)'),
       ('KOSE', 'Korea Stock Exchange (KRX)'),
       ('KWSE', 'Boursa Kuwait'),
       ('LJSE', 'Ljubljana Stock Exchange'),
       ('LSE', 'London Stock Exchange'),
       ('MSM', 'Muscat Securities Market'),
       ('MUN', 'Boerse Muenchen'),
       ('NASDAQCM', 'Nasdaq Capital Market'),
       ('NASDAQGM', 'Nasdaq Global Market'),
       ('NASDAQGS', 'Nasdaq Global Select Market'),
       ('NASE', 'Nairobi Securities Exchange'),
       ('NGM', 'Nordic Growth Market'),
       ('NGSE', 'Nigerian Exchange'),
       ('NSEI', 'National Stock Exchange of India'),
       ('NSEL', 'Nasdaq Vilnius'),
       ('NYSE', 'New York Stock Exchange'),
       ('NYSEAM', 'NYSE American'),
       ('NZSE', 'New Zealand Stock Exchange'),
       ('OB', 'Oslo Bors'),
       ('OFEX', 'Aquis Stock Exchange (OFEX)'),
       ('OM', 'Nasdaq Stockholm (OMX)'),
       ('OTCPK', 'OTC Pink Markets'),
       ('PSE', 'Philippine Stock Exchange'),
       ('SASE', 'Saudi Exchange (Tadawul)'),
       ('SEHK', 'Hong Kong Stock Exchange'),
       ('SEP', 'Prague Stock Exchange'),
       ('SET', 'Stock Exchange of Thailand'),
       ('SGX', 'Singapore Exchange'),
       ('SHSE', 'Shanghai Stock Exchange'),
       ('SNSE', 'Santiago Stock Exchange'),
       ('SWX', 'SIX Swiss Exchange'),
       ('SZSE', 'Shenzhen Stock Exchange'),
       ('TASE', 'Tel Aviv Stock Exchange'),
       ('TLSE', 'Nasdaq Tallinn'),
       ('TPEX', 'Taipei Exchange'),
       ('TSE', 'Tokyo Stock Exchange'),
       ('TSX', 'Toronto Stock Exchange'),
       ('TSXV', 'TSX Venture Exchange'),
       ('TWSE', 'Taiwan Stock Exchange'),
       ('WBAG', 'Wiener Boerse'),
       ('WSE', 'Warsaw Stock Exchange'),
       ('XSAT', 'Spotlight Stock Market'),
       ('XTRA', 'Deutsche Boerse Xetra'),
       ('ZGSE', 'Zagreb Stock Exchange')
;

-- Resolve an exchange code to its full name. Falls back to the original
-- input when no mapping exists so unknown codes aren't lost.
CREATE OR REPLACE FUNCTION exchange_name(code_text TEXT) RETURNS TEXT AS
$$
SELECT COALESCE((SELECT r.name FROM exchange_ref r WHERE r.code = UPPER(TRIM(code_text))), NULLIF(TRIM(code_text), ''))
$$
	LANGUAGE sql STABLE
;

-- ===================================================================
-- CURRENCY CODE -> CURRENCY NAME REFERENCE (ISO 4217)
-- ===================================================================
DROP TABLE IF EXISTS currency_ref
;
CREATE TEMP TABLE currency_ref
(
	code TEXT PRIMARY KEY,
	name TEXT NOT NULL
)
;

INSERT INTO currency_ref (code, name)
VALUES ('AED', 'UAE Dirham'),
       ('ARS', 'Argentine Peso'),
       ('AUD', 'Australian Dollar'),
       ('AZN', 'Azerbaijani Manat'),
       ('BHD', 'Bahraini Dinar'),
       ('BRL', 'Brazilian Real'),
       ('CAD', 'Canadian Dollar'),
       ('CHF', 'Swiss Franc'),
       ('CLP', 'Chilean Peso'),
       ('CNY', 'Chinese Yuan'),
       ('COP', 'Colombian Peso'),
       ('CZK', 'Czech Koruna'),
       ('DKK', 'Danish Krone'),
       ('EGP', 'Egyptian Pound'),
       ('EUR', 'Euro'),
       ('GBP', 'British Pound'),
       ('GEL', 'Georgian Lari'),
       ('GHS', 'Ghanaian Cedi'),
       ('HKD', 'Hong Kong Dollar'),
       ('HUF', 'Hungarian Forint'),
       ('IDR', 'Indonesian Rupiah'),
       ('ILS', 'Israeli New Shekel'),
       ('INR', 'Indian Rupee'),
       ('ISK', 'Icelandic Krona'),
       ('JPY', 'Japanese Yen'),
       ('KES', 'Kenyan Shilling'),
       ('KRW', 'South Korean Won'),
       ('KWD', 'Kuwaiti Dinar'),
       ('MAD', 'Moroccan Dirham'),
       ('MXN', 'Mexican Peso'),
       ('MYR', 'Malaysian Ringgit'),
       ('NGN', 'Nigerian Naira'),
       ('NOK', 'Norwegian Krone'),
       ('NZD', 'New Zealand Dollar'),
       ('OMR', 'Omani Rial'),
       ('PEN', 'Peruvian Sol'),
       ('PHP', 'Philippine Peso'),
       ('PKR', 'Pakistani Rupee'),
       ('PLN', 'Polish Zloty'),
       ('QAR', 'Qatari Riyal'),
       ('RON', 'Romanian Leu'),
       ('SAR', 'Saudi Riyal'),
       ('SEK', 'Swedish Krona'),
       ('SGD', 'Singapore Dollar'),
       ('THB', 'Thai Baht'),
       ('TRY', 'Turkish Lira'),
       ('TWD', 'New Taiwan Dollar'),
       ('USD', 'US Dollar'),
       ('VND', 'Vietnamese Dong'),
       ('ZAR', 'South African Rand')
;

-- Resolve a currency code to its full name. Falls back to the original
-- input when no mapping exists so unknown codes aren't lost.
CREATE OR REPLACE FUNCTION currency_name(code_text TEXT) RETURNS TEXT AS
$$
SELECT COALESCE((SELECT r.name FROM currency_ref r WHERE r.code = UPPER(TRIM(code_text))), NULLIF(TRIM(code_text), ''))
$$
	LANGUAGE sql STABLE
;

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

DROP TABLE IF EXISTS staging_header_buf
;

CREATE TEMP TABLE staging_header_buf
(
	header_line TEXT
)
;

-- Load ONLY the first line of the US file into the buffer.
-- We disable HEADER so the line is captured verbatim, and use a delimiter
-- that won't appear in a CSV header line (\b = backspace) so the entire
-- line lands in the single column.
\copy staging_header_buf FROM PROGRAM 'powershell -NoProfile -Command "Get-Content -Path data/pml/pml_us.csv -TotalCount 1"' WITH (FORMAT text)

-- Build the staging table from the discovered header.
DO
$$
	DECLARE
		v_header   TEXT;
		v_cols     TEXT[];
		v_col      TEXT;
		v_ddl      TEXT    := 'CREATE TEMP TABLE staging_header_buf (';
		v_first    BOOLEAN := TRUE;
		v_has_isin BOOLEAN := FALSE;
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

				IF v_col = 'ISIN' THEN v_has_isin := TRUE; END IF;
			END LOOP;

		v_ddl := v_ddl || ')';

		-- The \copy WHERE filters below reference "ISIN" by name; fail fast
		-- here if the vendor ever renames or drops that header column so we
		-- get a clear error instead of a broken filter at import time.
		IF NOT v_has_isin THEN
			RAISE EXCEPTION 'CSV header has no ISIN column. Found columns: %', v_cols;
		END IF;

		EXECUTE 'DROP TABLE IF EXISTS staging_header_buf';
		EXECUTE v_ddl;

		RAISE NOTICE 'Staging table created with % columns.', array_length(v_cols, 1);
	END
$$
;

-- ===================================================================
-- DATA IMPORT EXECUTION
-- ===================================================================
-- All four regional files MUST share the same header (same column set,
-- same order) as data/pml/pml_us.csv. If a vendor ever ships a file
-- with a divergent header, we want a hard, immediate failure rather
-- than silent column misalignment -- which is exactly what \copy
-- gives us, since it validates field count per row.
--
-- ROW FILTER: each \copy carries a WHERE clause (PostgreSQL 12+) that
-- drops rows whose ISIN is missing -- e.g. `688825,,CXMT Corporation,`
-- in pml_apac.csv. ISIN is the primary identifier / PyMC coord for
-- pml.pml_df, so ISIN-less rows are unusable downstream. The predicate
-- covers all three empty shapes: unquoted empty (NULL via the NULL ''
-- option), quoted empty (""), and whitespace-only values.

\echo 'Importing US data...'
\copy staging_header_buf FROM 'data/pml/pml_us.csv'   WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description")) WHERE NULLIF(BTRIM("ISIN"), '') IS NOT NULL

\echo 'Importing EU data...'
\copy staging_header_buf FROM 'data/pml/pml_eu.csv'   WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description")) WHERE NULLIF(BTRIM("ISIN"), '') IS NOT NULL

\echo 'Importing APAC data...'
\copy staging_header_buf FROM 'data/pml/pml_apac.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description")) WHERE NULLIF(BTRIM("ISIN"), '') IS NOT NULL

\echo 'Importing ROTW data...'
\copy staging_header_buf FROM 'data/pml/pml_rotw.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8', QUOTE '"', ESCAPE '"', DELIMITER ',', FORCE_NULL("Description")) WHERE NULLIF(BTRIM("ISIN"), '') IS NOT NULL

-- ===================================================================
-- DATA VALIDATION (PRE-INSERT)
-- ===================================================================
\echo 'Validating imported data...'
SELECT 'Total rows in staging:' AS info, COUNT(*) AS count
FROM staging_header_buf
;

-- Assert the \copy WHERE filters did their job: no ISIN-less rows may
-- reach pml_df. Runs BEFORE the TRUNCATE so a regression (e.g. a \copy
-- line rewritten without its WHERE clause) aborts with the current
-- pml_df contents intact.
DO
$$
	DECLARE
		v_missing_isin BIGINT;
	BEGIN
		SELECT COUNT(*) INTO v_missing_isin FROM staging_header_buf WHERE NULLIF(BTRIM("isin"), '') IS NULL;
		IF v_missing_isin > 0 THEN
			RAISE EXCEPTION '% staged row(s) have an empty ISIN; the \copy WHERE filters should have dropped them.',
				v_missing_isin;
		END IF;
		RAISE NOTICE 'ISIN validation passed: no empty-ISIN rows in staging.';
	END
$$
;

TRUNCATE TABLE pml_df CASCADE
;
INSERT INTO pml_df (ticker, isin, name, description, trading_region, trading_country, trading_country_name, exchange,
                    exchange_name, unit, unit_name, region, country, country_name, sector, industry, style_class,
                    size_class, last_updated, income_statement_report_date, fy_end, next_earnings, next_earnings_when,
                    next_earnings_status, fy_end_date, next_fiscal_quarter, next_income_statement_report_date,
                    next_fy_end_date, days_to_earnings, earnings_report_recency, expected_report_date,
                    earnings_report_frequency, dividend_record_currency, dividend_record_currency_name,
                    dividend_record_amount, dividend_record_frequency, dividend_streak, dividend_record_announce_date,
                    dividend_record_payable_date, dividend_record_record_date, dividend_record_ex_date, market_cap,
                    enterprise_value, last_price, price_target_ytd_ago, total_return_ytd, price_target,
                    price_target_low, price_target_median, price_target_high, price_target_num, p_e_ntm, p_e_ltm,
                    altman_z_score_fy, altman_z_score_fq, altman_z_score_ltm, beta_1y, beta_2y, beta_5y, analyst_rating,
                    num_strong_sell_ratings, num_strong_buys_ratings, num_hold_ratings, num_buys_ratings,
                    num_sell_ratings, num_no_opinion_ratings, market_cap_country_r, tot_return_pct_cagr_3y,
                    tot_return_pct_cagr_10y, total_return_5y, total_return_10y, volume_shrs, dividend_per_share_ltm,
                    div_yield_ind, div_yield_ltm, gross_profit_margin_pct_fy, gross_profit_margin_pct_ltm,
                    eps_norm_est_avg_ntm, eps_adj_neg1fy, eps_adj_fy, eps_adj_ltm, eps_norm_est_avg_fy1e,
                    buyback_yield_ltm, return_on_assets_roa_pct_ltm, return_on_assets_roa_pct_fy, div_yield_neg1fyind,
                    p_b_ltm, p_b_neg1fy, p_b_5yavg, div_yield_ttm, div_yield_ntm, div_yield_5yavgltm, price_chg_pct_3m,
                    one_day_pct, price_5d_ago, price_1w_ago, price_1m_ago, price_3m_ago, price_6m_ago, price_1y_ago,
                    price_3y_ago, price_5y_ago, price_qtd_ago, rel_volume, shrs_out, shrs_out_neg1fy,
                    common_dividends_paid_ltm, common_dividends_paid_fy, ev_sales_ltm, ev_sales_ntm, ev_sales_neg1fyltm,
                    ev_sales_neg2fyltm, ev_sales_neg3fyltm, ev_sales_3yavgltm, ev_sales_neg1fqltm, ev_sales_neg2fqltm,
                    ev_sales_neg3fqltm, ev_sales_neg4fqltm, w_52high_adj, w_52low_adj, ema_20d, ema_50d, ema_100d,
                    ema_250d, ev_ebitda_ltm, ev_ebitda_ntm, ev_ebitda_neg1fyltm, ev_ebitda_neg1fqltm,
                    ev_ebitda_3yavgltm, ev_ebitda_est_fy1, p_e_est_fy1, p_e_neg1fyltm, p_e_neg2fyltm, p_e_neg3fyltm,
                    p_e_3yavgltm, p_e_neg1fqltm, p_e_neg2fqltm, p_e_neg3fqltm, p_e_5yavgltm, p_e_neg0fqqoqltm,
                    p_e_neg0fyyoyltm, p_e_neg1fyyoyltm, p_e_neg0fqyoyltm, full_time_employees_fq,
                    full_time_employees_fy, full_time_employees_neg1fy, full_time_employees_neg2fy,
                    full_time_employees_neg3fy, avg_employees_5yavgfy, net_eps_basic_ltm, net_eps_basic_fq,
                    net_eps_basic_fy, net_eps_basic_neg1fqfq, net_eps_basic_neg2fqfq, net_eps_basic_neg3fqfq,
                    net_eps_basic_neg4fqfq, net_eps_basic_neg1fy, net_eps_basic_neg2fy, net_eps_basic_neg3fy,
                    net_eps_basic_neg4fy, net_eps_basic_neg5fy, eps_est_avg_rev_pct_fy1e_1w,
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
                    total_return_2025, total_return_2024, total_return_2023, total_return_2022, total_return_2021,
                    return_on_assets_roa_pct_fq, return_on_assets_roa_pct_neg1fqfq, return_on_assets_roa_pct_neg2fqfq,
                    return_on_assets_roa_pct_neg3fqfq, return_on_assets_roa_pct_neg4fqfq,
                    return_on_assets_roa_pct_neg1fy, return_on_assets_roa_pct_neg2fy, return_on_assets_roa_pct_neg3fy,
                    return_on_assets_roa_pct_neg4fy, asset_turnover_fq, asset_turnover_fy, asset_turnover_neg1fqfq,
                    asset_turnover_neg2fqfq, asset_turnover_neg3fqfq, asset_turnover_neg4fqfq, asset_turnover_neg1fy,
                    asset_turnover_neg2fy, asset_turnover_neg3fy, asset_turnover_neg4fy, quick_ratio_ltm,
                    quick_ratio_fq, quick_ratio_fy, quick_ratio_neg1fqfq, quick_ratio_neg2fqfq, quick_ratio_neg3fqfq,
                    quick_ratio_neg4fqfq, quick_ratio_neg1fy, quick_ratio_neg2fy, quick_ratio_neg3fy,
                    quick_ratio_neg4fy, current_ratio_ltm, current_ratio_fq, current_ratio_fy, current_ratio_neg1fqfq,
                    current_ratio_neg2fqfq, current_ratio_neg3fqfq, current_ratio_neg4fqfq, current_ratio_neg1fy,
                    current_ratio_neg2fy, current_ratio_neg3fy, current_ratio_neg4fy, long_term_debt_equity_ltm,
                    long_term_debt_equity_fq, long_term_debt_equity_fy, long_term_debt_equity_neg1fqfq,
                    long_term_debt_equity_neg2fqfq, long_term_debt_equity_neg3fqfq, long_term_debt_equity_neg4fqfq,
                    long_term_debt_equity_neg1fy, long_term_debt_equity_neg2fy, long_term_debt_equity_neg3fy,
                    long_term_debt_equity_neg4fy, net_income_ltm, net_income_fq, net_income_fy, net_income_neg1fqfq,
                    net_income_neg2fqfq, net_income_neg3fqfq, net_income_neg4fqfq, net_income_neg1fy, net_income_neg2fy,
                    net_income_neg3fy, net_income_neg4fy)
SELECT NULLIF(TRIM(s."ticker"), '')                                                                 AS ticker,
       NULLIF(TRIM(s."isin"), '')                                                                   AS isin,
       NULLIF(TRIM(s."name"), '')                                                                   AS name,
       NULLIF(TRIM(s."description"), '')                                                            AS description,
       COALESCE(NULLIF(TRIM(s."trading region"), ''), 'n/a')                                        AS trading_region,
       COALESCE(NULLIF(TRIM(s."trading country"), ''), 'n/a')                                       AS trading_country,
       COALESCE(country_name(s."trading country"), 'n/a')                                           AS trading_country_name,
       COALESCE(NULLIF(TRIM(s."exchange"), ''), 'n/a')                                              AS exchange,
       COALESCE(exchange_name(s."exchange"), 'n/a')                                                 AS exchange_name,
       COALESCE(NULLIF(TRIM(s."unit"), ''), 'n/a')                                                  AS unit,
       COALESCE(currency_name(s."unit"), 'n/a')                                                     AS unit_name,
       COALESCE(NULLIF(TRIM(s."region"), ''), 'n/a')                                                AS region,
       COALESCE(NULLIF(TRIM(s."country"), ''), 'n/a')                                               AS country,
       COALESCE(country_name(s."country"), 'n/a')                                                   AS country_name,
       COALESCE(NULLIF(TRIM(s."sector"), ''), 'n/a')                                                AS sector,
       COALESCE(NULLIF(TRIM(s."industry"), ''), 'n/a')                                              AS industry,
       COALESCE(NULLIF(TRIM(s."style class"), ''), 'n/a')                                           AS style_class,
       COALESCE(NULLIF(TRIM(s."size class"), ''), 'n/a')                                            AS size_class,
       text_to_date_safe(s."last updated")                                                          AS last_updated,
       COALESCE(text_to_date_safe(s."income statement report date"),
                '2026-03-31')                                                                       AS income_statement_report_date,
       COALESCE(NULLIF(TRIM(s."fy end"), ''), 'Dec 2025')                                           AS fy_end,
       text_to_date_safe(s."next earnings")                                                         AS next_earnings,
       COALESCE(NULLIF(TRIM(s."next earnings (when)"), ''), 'n/a')                                  AS next_earnings_when,
       COALESCE(NULLIF(TRIM(s."next earnings (status)"), ''), 'n/a')                                AS next_earnings_status,
       COALESCE(parse_fiscal_year_end_date(s."fy end"), '2025-12-31')                               AS fy_end_date,
       calculate_next_fiscal_quarter_date(text_to_date_safe(s."income statement report date"))      AS next_fiscal_quarter,
       calculate_next_income_statement_report_date(text_to_date_safe(s."income statement report date"),
                                                   derive_earnings_report_frequency(
		                                                   text_to_date_safe(s."income statement report date"),
		                                                   parse_fiscal_year_end_date(s."fy end"))) AS next_income_statement_report_date,
       calculate_next_fy_end_date(parse_fiscal_year_end_date(s."fy end"))                           AS next_fy_end_date,
       (text_to_date_safe(s."next earnings") - CURRENT_DATE)                                        AS days_to_earnings,
       (CURRENT_DATE - text_to_date_safe(s."income statement report date"))                         AS earnings_report_recency,
       calculate_expected_report_date(text_to_date_safe(s."income statement report date"),
                                      derive_earnings_report_frequency(
		                                      text_to_date_safe(s."income statement report date"),
		                                      parse_fiscal_year_end_date(s."fy end")))              AS expected_report_date,
       derive_earnings_report_frequency(text_to_date_safe(s."income statement report date"),
                                        parse_fiscal_year_end_date(s."fy end"))                     AS earnings_report_frequency,
       COALESCE(NULLIF(TRIM(s."dividend record (currency)"), ''), 'n/a')                            AS dividend_record_currency,
       COALESCE(currency_name(s."dividend record (currency)"), 'n/a')                               AS dividend_record_currency_name,
       text_to_numeric_safe(s."dividend record (amount)")                                           AS dividend_record_amount,
       COALESCE(NULLIF(TRIM(s."dividend record (frequency)"), ''),
                'n/a')                                                                              AS dividend_record_frequency,
       text_to_numeric_safe(s."dividend streak")                                                    AS dividend_streak,
       text_to_date_safe(s."dividend record (announce date)")                                       AS dividend_record_announce_date,
       text_to_date_safe(s."dividend record (payable date)")                                        AS dividend_record_payable_date,
       text_to_date_safe(s."dividend record (record date)")                                         AS dividend_record_record_date,
       text_to_date_safe(s."dividend record (ex date)")                                             AS dividend_record_ex_date,
       text_to_numeric_safe(s."market cap")                                                         AS market_cap,
       text_to_numeric_safe(s."enterprise value")                                                   AS enterprise_value,
       text_to_numeric_safe(s."last price")                                                         AS last_price,
       text_to_numeric_safe(s."price target (ytd ago)")                                             AS price_target_ytd_ago,
       text_to_numeric_safe(s."total return (ytd)")                                                 AS total_return_ytd,
       text_to_numeric_safe(s."price target")                                                       AS price_target,
       text_to_numeric_safe(s."price target - low")                                                 AS price_target_low,
       text_to_numeric_safe(s."price target - median")                                              AS price_target_median,
       text_to_numeric_safe(s."price target - high")                                                AS price_target_high,
       text_to_numeric_safe(s."price target - #")                                                   AS price_target_num,
       text_to_numeric_safe(s."p/e (ntm)")                                                          AS p_e_ntm,
       text_to_numeric_safe(s."p/e (ltm)")                                                          AS p_e_ltm,
       text_to_numeric_safe(s."altman z-score (fy)")                                                AS altman_z_score_fy,
       text_to_numeric_safe(s."altman z-score (fq)")                                                AS altman_z_score_fq,
       text_to_numeric_safe(s."altman z-score (ltm)")                                               AS altman_z_score_ltm,
       text_to_numeric_safe(s."beta (1y)")                                                          AS beta_1y,
       text_to_numeric_safe(s."beta (2y)")                                                          AS beta_2y,
       text_to_numeric_safe(s."beta (5y)")                                                          AS beta_5y,
       text_to_numeric_safe(s."analyst rating")                                                     AS analyst_rating,
       COALESCE(text_to_numeric_safe(s."# strong sell ratings"), 0)::INT                            AS num_strong_sell_ratings,
       COALESCE(text_to_numeric_safe(s."# strong buys ratings"), 0)::INT                            AS num_strong_buys_ratings,
       COALESCE(text_to_numeric_safe(s."# hold ratings"), 0)::INT                                   AS num_hold_ratings,
       COALESCE(text_to_numeric_safe(s."# buys ratings"), 0)::INT                                   AS num_buys_ratings,
       COALESCE(text_to_numeric_safe(s."# sell ratings"), 0)::INT                                   AS num_sell_ratings,
       COALESCE(text_to_numeric_safe(s."# no opinion ratings"), 0)::INT                             AS num_no_opinion_ratings,
       COALESCE(text_to_numeric_safe(s."market cap (country r)"), 0)::INT                           AS market_cap_country_r,
       text_to_numeric_safe(s."tot. return %/cagr (3y)")                                            AS tot_return_pct_cagr_3y,
       text_to_numeric_safe(s."tot. return %/cagr (10y)")                                           AS tot_return_pct_cagr_10y,
       text_to_numeric_safe(s."total return (5y)")                                                  AS total_return_5y,
       text_to_numeric_safe(s."total return (10y)")                                                 AS total_return_10y,
       text_to_numeric_safe(s."volume (shrs)")                                                      AS volume_shrs,
       text_to_numeric_safe(s."dividend per share (ltm)")                                           AS dividend_per_share_ltm,
       text_to_numeric_safe(s."div yield (ind)")                                                    AS div_yield_ind,
       text_to_numeric_safe(s."div yield (ltm)")                                                    AS div_yield_ltm,
       text_to_numeric_safe(s."gross profit margin % (fy)")                                         AS gross_profit_margin_pct_fy,
       text_to_numeric_safe(s."gross profit margin % (ltm)")                                        AS gross_profit_margin_pct_ltm,
       text_to_numeric_safe(s."eps norm - est avg (ntm)")                                           AS eps_norm_est_avg_ntm,
       text_to_numeric_safe(s."eps/adj. (-1fy)")                                                    AS eps_adj_neg1fy,
       text_to_numeric_safe(s."eps/adj. (fy)")                                                      AS eps_adj_fy,
       text_to_numeric_safe(s."eps/adj. (ltm)")                                                     AS eps_adj_ltm,
       text_to_numeric_safe(s."eps norm - est avg (fy1e)")                                          AS eps_norm_est_avg_fy1e,
       text_to_numeric_safe(s."buyback yield (ltm)")                                                AS buyback_yield_ltm,
       text_to_numeric_safe(s."return on assets (roa) % (ltm)")                                     AS return_on_assets_roa_pct_ltm,
       text_to_numeric_safe(s."return on assets (roa) % (fy)")                                      AS return_on_assets_roa_pct_fy,
       text_to_numeric_safe(s."div yield (-1fyind)")                                                AS div_yield_neg1fyind,
       text_to_numeric_safe(s."p/b (ltm)")                                                          AS p_b_ltm,
       text_to_numeric_safe(s."p/b (-1fy)")                                                         AS p_b_neg1fy,
       text_to_numeric_safe(s."p/b (5yavg)")                                                        AS p_b_5yavg,
       text_to_numeric_safe(s."div yield (ttm)")                                                    AS div_yield_ttm,
       text_to_numeric_safe(s."div yield (ntm)")                                                    AS div_yield_ntm,
       text_to_numeric_safe(s."div yield (5yavgltm)")                                               AS div_yield_5yavgltm,
       text_to_numeric_safe(s."price chg. % (3m)")                                                  AS price_chg_pct_3m,
       text_to_numeric_safe(s."1-day %")                                                            AS one_day_pct,
       text_to_numeric_safe(s."price (5d ago)")                                                     AS price_5d_ago,
       text_to_numeric_safe(s."price (1w ago)")                                                     AS price_1w_ago,
       text_to_numeric_safe(s."price (1m ago)")                                                     AS price_1m_ago,
       text_to_numeric_safe(s."price (3m ago)")                                                     AS price_3m_ago,
       text_to_numeric_safe(s."price (6m ago)")                                                     AS price_6m_ago,
       text_to_numeric_safe(s."price (1y ago)")                                                     AS price_1y_ago,
       text_to_numeric_safe(s."price (3y ago)")                                                     AS price_3y_ago,
       text_to_numeric_safe(s."price (5y ago)")                                                     AS price_5y_ago,
       text_to_numeric_safe(s."price (qtd ago)")                                                    AS price_qtd_ago,
       text_to_numeric_safe(s."rel. volume")                                                        AS rel_volume,
       ROUND(text_to_numeric_safe(s."shrs out") / 1000000.0, 2)                                     AS shrs_out,
       ROUND(text_to_numeric_safe(s."shrs out (-1fy)") / 1000000.0, 2)                              AS shrs_out_neg1fy,
       text_to_numeric_safe(s."common dividends paid (ltm)")                                        AS common_dividends_paid_ltm,
       text_to_numeric_safe(s."common dividends paid (fy)")                                         AS common_dividends_paid_fy,
       text_to_numeric_safe(s."ev/sales (ltm)")                                                     AS ev_sales_ltm,
       text_to_numeric_safe(s."ev/sales (ntm)")                                                     AS ev_sales_ntm,
       text_to_numeric_safe(s."ev/sales (-1fyltm)")                                                 AS ev_sales_neg1fyltm,
       text_to_numeric_safe(s."ev/sales (-2fyltm)")                                                 AS ev_sales_neg2fyltm,
       text_to_numeric_safe(s."ev/sales (-3fyltm)")                                                 AS ev_sales_neg3fyltm,
       text_to_numeric_safe(s."ev/sales (3yavgltm)")                                                AS ev_sales_3yavgltm,
       text_to_numeric_safe(s."ev/sales (-1fqltm)")                                                 AS ev_sales_neg1fqltm,
       text_to_numeric_safe(s."ev/sales (-2fqltm)")                                                 AS ev_sales_neg2fqltm,
       text_to_numeric_safe(s."ev/sales (-3fqltm)")                                                 AS ev_sales_neg3fqltm,
       text_to_numeric_safe(s."ev/sales (-4fqltm)")                                                 AS ev_sales_neg4fqltm,
       text_to_numeric_safe(s."52w high/adj")                                                       AS w_52high_adj,
       text_to_numeric_safe(s."52w low/adj")                                                        AS w_52low_adj,
       text_to_numeric_safe(s."ema (20d)")                                                          AS ema_20d,
       text_to_numeric_safe(s."ema (50d)")                                                          AS ema_50d,
       text_to_numeric_safe(s."ema (100d)")                                                         AS ema_100d,
       text_to_numeric_safe(s."ema (250d)")                                                         AS ema_250d,
       text_to_numeric_safe(s."ev/ebitda (ltm)")                                                    AS ev_ebitda_ltm,
       text_to_numeric_safe(s."ev/ebitda (ntm)")                                                    AS ev_ebitda_ntm,
       text_to_numeric_safe(s."ev/ebitda (-1fyltm)")                                                AS ev_ebitda_neg1fyltm,
       text_to_numeric_safe(s."ev/ebitda (-1fqltm)")                                                AS ev_ebitda_neg1fqltm,
       text_to_numeric_safe(s."ev/ebitda (3yavgltm)")                                               AS ev_ebitda_3yavgltm,
       text_to_numeric_safe(s."ev/ebitda (est fy1)")                                                AS ev_ebitda_est_fy1,
       text_to_numeric_safe(s."p/e (est fy1)")                                                      AS p_e_est_fy1,
       text_to_numeric_safe(s."p/e (-1fyltm)")                                                      AS p_e_neg1fyltm,
       text_to_numeric_safe(s."p/e (-2fyltm)")                                                      AS p_e_neg2fyltm,
       text_to_numeric_safe(s."p/e (-3fyltm)")                                                      AS p_e_neg3fyltm,
       text_to_numeric_safe(s."p/e (3yavgltm)")                                                     AS p_e_3yavgltm,
       text_to_numeric_safe(s."p/e (-1fqltm)")                                                      AS p_e_neg1fqltm,
       text_to_numeric_safe(s."p/e (-2fqltm)")                                                      AS p_e_neg2fqltm,
       text_to_numeric_safe(s."p/e (-3fqltm)")                                                      AS p_e_neg3fqltm,
       text_to_numeric_safe(s."p/e (5yavgltm)")                                                     AS p_e_5yavgltm,
       text_to_numeric_safe(s."p/e (-0fqqoqltm)")                                                   AS p_e_neg0fqqoqltm,
       text_to_numeric_safe(s."p/e (-0fyyoyltm)")                                                   AS p_e_neg0fyyoyltm,
       text_to_numeric_safe(s."p/e (-1fyyoyltm)")                                                   AS p_e_neg1fyyoyltm,
       text_to_numeric_safe(s."p/e (-0fqyoyltm)")                                                   AS p_e_neg0fqyoyltm,
       text_to_numeric_safe(s."full time employees (fq)")                                           AS full_time_employees_fq,
       text_to_numeric_safe(s."full time employees (fy)")                                           AS full_time_employees_fy,
       text_to_numeric_safe(s."full time employees (-1fy)")                                         AS full_time_employees_neg1fy,
       text_to_numeric_safe(s."full time employees (-2fy)")                                         AS full_time_employees_neg2fy,
       text_to_numeric_safe(s."full time employees (-3fy)")                                         AS full_time_employees_neg3fy,
       text_to_numeric_safe(s."avg employees (5yavgfy)")                                            AS avg_employees_5yavgfy,
       text_to_numeric_safe(s."net eps - basic (ltm)")                                              AS net_eps_basic_ltm,
       text_to_numeric_safe(s."net eps - basic (fq)")                                               AS net_eps_basic_fq,
       text_to_numeric_safe(s."net eps - basic (fy)")                                               AS net_eps_basic_fy,
       text_to_numeric_safe(s."net eps - basic (-1fqfq)")                                           AS net_eps_basic_neg1fqfq,
       text_to_numeric_safe(s."net eps - basic (-2fqfq)")                                           AS net_eps_basic_neg2fqfq,
       text_to_numeric_safe(s."net eps - basic (-3fqfq)")                                           AS net_eps_basic_neg3fqfq,
       text_to_numeric_safe(s."net eps - basic (-4fqfq)")                                           AS net_eps_basic_neg4fqfq,
       text_to_numeric_safe(s."net eps - basic (-1fy)")                                             AS net_eps_basic_neg1fy,
       text_to_numeric_safe(s."net eps - basic (-2fy)")                                             AS net_eps_basic_neg2fy,
       text_to_numeric_safe(s."net eps - basic (-3fy)")                                             AS net_eps_basic_neg3fy,
       text_to_numeric_safe(s."net eps - basic (-4fy)")                                             AS net_eps_basic_neg4fy,
       text_to_numeric_safe(s."net eps - basic (-5fy)")                                             AS net_eps_basic_neg5fy,
       text_to_numeric_safe(s."eps est avg rev % (fy1e - 1w)")                                      AS eps_est_avg_rev_pct_fy1e_1w,
       text_to_numeric_safe(s."eps est avg rev % (fy1e - 1m)")                                      AS eps_est_avg_rev_pct_fy1e_1m,
       text_to_numeric_safe(s."eps est avg rev % (fy1e - 3m)")                                      AS eps_est_avg_rev_pct_fy1e_3m,
       text_to_numeric_safe(s."eps est avg rev % (fy1e - 6m)")                                      AS eps_est_avg_rev_pct_fy1e_6m,
       text_to_numeric_safe(s."eps est avg rev % (fy1e - 1y)")                                      AS eps_est_avg_rev_pct_fy1e_1y,
       text_to_numeric_safe(s."div yield (-2fyind)")                                                AS div_yield_neg2fyind,
       text_to_numeric_safe(s."div yield (-3fyind)")                                                AS div_yield_neg3fyind,
       text_to_numeric_safe(s."div yield (-4fyind)")                                                AS div_yield_neg4fyind,
       text_to_numeric_safe(s."div yield (-5fyind)")                                                AS div_yield_neg5fyind,
       text_to_numeric_safe(s."ebitda - est avg (ntm)")                                             AS ebitda_est_avg_ntm,
       text_to_numeric_safe(s."ebitda - est avg (fy1e)")                                            AS ebitda_est_avg_fy1e,
       text_to_numeric_safe(s."eps gaap - est avg (ntm)")                                           AS eps_gaap_est_avg_ntm,
       text_to_numeric_safe(s."eps gaap - est avg (fy1e)")                                          AS eps_gaap_est_avg_fy1e,
       text_to_numeric_safe(s."eps gaap est avg rev % (fy1e - 1m)")                                 AS eps_gaap_est_avg_rev_pct_fy1e_1m,
       text_to_numeric_safe(s."eps gaap est avg rev % (fy1e - 3m)")                                 AS eps_gaap_est_avg_rev_pct_fy1e_3m,
       text_to_numeric_safe(s."eps gaap est avg rev % (fy1e - 6m)")                                 AS eps_gaap_est_avg_rev_pct_fy1e_6m,
       text_to_numeric_safe(s."eps gaap est avg rev % (fy1e - 1y)")                                 AS eps_gaap_est_avg_rev_pct_fy1e_1y,
       text_to_numeric_safe(s."eps norm - est # (fy1e)")                                            AS eps_norm_est_num_fy1e,
       text_to_numeric_safe(s."price target (1w ago)")                                              AS price_target_1w_ago,
       text_to_numeric_safe(s."price target (1m ago)")                                              AS price_target_1m_ago,
       text_to_numeric_safe(s."price target (3m ago)")                                              AS price_target_3m_ago,
       text_to_numeric_safe(s."price target (6m ago)")                                              AS price_target_6m_ago,
       text_to_numeric_safe(s."price target (mtd ago)")                                             AS price_target_mtd_ago,
       text_to_numeric_safe(s."price target (qtd ago)")                                             AS price_target_qtd_ago,
       text_to_numeric_safe(s."price target (1y ago)")                                              AS price_target_1y_ago,
       text_to_numeric_safe(s."price target - # (3m ago)")                                          AS price_target_num_3m_ago,
       text_to_numeric_safe(s."price target - # (6m ago)")                                          AS price_target_num_6m_ago,
       text_to_numeric_safe(s."price target - # (ytd ago)")                                         AS price_target_num_ytd_ago,
       text_to_numeric_safe(s."price target - # (1y ago)")                                          AS price_target_num_1y_ago,
       text_to_numeric_safe(s."price target - # (1w ago)")                                          AS price_target_num_1w_ago,
       text_to_numeric_safe(s."price target - # (1m ago)")                                          AS price_target_num_1m_ago,
       text_to_numeric_safe(s."price target - # (mtd ago)")                                         AS price_target_num_mtd_ago,
       text_to_numeric_safe(s."price target - # (qtd ago)")                                         AS price_target_num_qtd_ago,
       text_to_numeric_safe(s."price target - high (1w ago)")                                       AS price_target_high_1w_ago,
       text_to_numeric_safe(s."price target - high (1m ago)")                                       AS price_target_high_1m_ago,
       text_to_numeric_safe(s."price target - high (6m ago)")                                       AS price_target_high_6m_ago,
       text_to_numeric_safe(s."price target - high (mtd ago)")                                      AS price_target_high_mtd_ago,
       text_to_numeric_safe(s."price target - high (3m ago)")                                       AS price_target_high_3m_ago,
       text_to_numeric_safe(s."price target - high (qtd ago)")                                      AS price_target_high_qtd_ago,
       text_to_numeric_safe(s."price target - high (1y ago)")                                       AS price_target_high_1y_ago,
       text_to_numeric_safe(s."price target - high (ytd ago)")                                      AS price_target_high_ytd_ago,
       text_to_numeric_safe(s."price target - low (1w ago)")                                        AS price_target_low_1w_ago,
       text_to_numeric_safe(s."price target - low (1m ago)")                                        AS price_target_low_1m_ago,
       text_to_numeric_safe(s."price target - low (3m ago)")                                        AS price_target_low_3m_ago,
       text_to_numeric_safe(s."price target - low (6m ago)")                                        AS price_target_low_6m_ago,
       text_to_numeric_safe(s."price target - low (mtd ago)")                                       AS price_target_low_mtd_ago,
       text_to_numeric_safe(s."price target - low (qtd ago)")                                       AS price_target_low_qtd_ago,
       text_to_numeric_safe(s."price target - low (ytd ago)")                                       AS price_target_low_ytd_ago,
       text_to_numeric_safe(s."price target - low (1y ago)")                                        AS price_target_low_1y_ago,
       text_to_numeric_safe(s."price target - median (1w ago)")                                     AS price_target_median_1w_ago,
       text_to_numeric_safe(s."price target - median (1m ago)")                                     AS price_target_median_1m_ago,
       text_to_numeric_safe(s."price target - median (3m ago)")                                     AS price_target_median_3m_ago,
       text_to_numeric_safe(s."price target - median (6m ago)")                                     AS price_target_median_6m_ago,
       text_to_numeric_safe(s."price target - median (mtd ago)")                                    AS price_target_median_mtd_ago,
       text_to_numeric_safe(s."price target - median (qtd ago)")                                    AS price_target_median_qtd_ago,
       text_to_numeric_safe(s."price target - median (ytd ago)")                                    AS price_target_median_ytd_ago,
       text_to_numeric_safe(s."price target - median (1y ago)")                                     AS price_target_median_1y_ago,
       text_to_numeric_safe(s."basic eps - cont (ltm)")                                             AS basic_eps_cont_ltm,
       text_to_numeric_safe(s."basic eps - cont (fq)")                                              AS basic_eps_cont_fq,
       text_to_numeric_safe(s."basic eps - cont (fy)")                                              AS basic_eps_cont_fy,
       text_to_numeric_safe(s."basic eps - cont (-1fqfq)")                                          AS basic_eps_cont_neg1fqfq,
       text_to_numeric_safe(s."basic eps - cont (-2fqfq)")                                          AS basic_eps_cont_neg2fqfq,
       text_to_numeric_safe(s."basic eps - cont (-4fqfq)")                                          AS basic_eps_cont_neg4fqfq,
       text_to_numeric_safe(s."basic eps - cont (-3fqfq)")                                          AS basic_eps_cont_neg3fqfq,
       text_to_numeric_safe(s."basic eps - cont (-1fy)")                                            AS basic_eps_cont_neg1fy,
       text_to_numeric_safe(s."basic eps - cont (-2fy)")                                            AS basic_eps_cont_neg2fy,
       text_to_numeric_safe(s."basic eps - cont (-3fy)")                                            AS basic_eps_cont_neg3fy,
       text_to_numeric_safe(s."basic eps - cont (-4fy)")                                            AS basic_eps_cont_neg4fy,
       text_to_numeric_safe(s."eps/adj. (fq)")                                                      AS eps_adj_fq,
       text_to_numeric_safe(s."eps/adj. (-1fqfq)")                                                  AS eps_adj_neg1fqfq,
       text_to_numeric_safe(s."eps/adj. (-3fqfq)")                                                  AS eps_adj_neg3fqfq,
       text_to_numeric_safe(s."eps/adj. (-4fqfq)")                                                  AS eps_adj_neg4fqfq,
       text_to_numeric_safe(s."eps/adj. (-2fqfq)")                                                  AS eps_adj_neg2fqfq,
       text_to_numeric_safe(s."eps/adj. (-2fy)")                                                    AS eps_adj_neg2fy,
       text_to_numeric_safe(s."eps/adj. (-3fy)")                                                    AS eps_adj_neg3fy,
       text_to_numeric_safe(s."eps/adj. (-4fy)")                                                    AS eps_adj_neg4fy,
       text_to_numeric_safe(s."gross profit (-1fqfq)")                                              AS gross_profit_neg1fqfq,
       text_to_numeric_safe(s."gross profit (-3fqfq)")                                              AS gross_profit_neg3fqfq,
       text_to_numeric_safe(s."gross profit (-4fqfq)")                                              AS gross_profit_neg4fqfq,
       text_to_numeric_safe(s."gross profit (-2fqfq)")                                              AS gross_profit_neg2fqfq,
       text_to_numeric_safe(s."gross profit (-1fy)")                                                AS gross_profit_neg1fy,
       text_to_numeric_safe(s."gross profit (-2fy)")                                                AS gross_profit_neg2fy,
       text_to_numeric_safe(s."gross profit (-3fy)")                                                AS gross_profit_neg3fy,
       text_to_numeric_safe(s."gross profit (-4fy)")                                                AS gross_profit_neg4fy,
       text_to_numeric_safe(s."fcf - est avg (fy1e)")                                               AS fcf_est_avg_fy1e,
       text_to_numeric_safe(s."fcf - est avg (fy2e)")                                               AS fcf_est_avg_fy2e,
       text_to_numeric_safe(s."fcf - est avg (fy3e)")                                               AS fcf_est_avg_fy3e,
       text_to_numeric_safe(s."fcf - est avg (fy4e)")                                               AS fcf_est_avg_fy4e,
       text_to_numeric_safe(s."fcf - est avg (fy5e)")                                               AS fcf_est_avg_fy5e,
       text_to_numeric_safe(s."eps (-0fyestimate)")                                                 AS eps_neg0fyestimate,
       text_to_numeric_safe(s."eps (-0fyactual)")                                                   AS eps_neg0fyactual,
       text_to_numeric_safe(s."eps (-0fysurprise %)")                                               AS eps_neg0fysurprise_pct,
       text_to_numeric_safe(s."eps (-1fyestimate)")                                                 AS eps_neg1fyestimate,
       text_to_numeric_safe(s."eps (-1fyactual)")                                                   AS eps_neg1fyactual,
       text_to_numeric_safe(s."eps (-1fysurprise %)")                                               AS eps_neg1fysurprise_pct,
       text_to_numeric_safe(s."eps (-2fyestimate)")                                                 AS eps_neg2fyestimate,
       text_to_numeric_safe(s."eps (-2fyactual)")                                                   AS eps_neg2fyactual,
       text_to_numeric_safe(s."eps (-2fysurprise %)")                                               AS eps_neg2fysurprise_pct,
       text_to_numeric_safe(s."eps (-3fyestimate)")                                                 AS eps_neg3fyestimate,
       text_to_numeric_safe(s."eps (-3fyactual)")                                                   AS eps_neg3fyactual,
       text_to_numeric_safe(s."eps (-3fysurprise %)")                                               AS eps_neg3fysurprise_pct,
       text_to_numeric_safe(s."eps (-4fyactual)")                                                   AS eps_neg4fyactual,
       text_to_numeric_safe(s."eps (-4fyestimate)")                                                 AS eps_neg4fyestimate,
       text_to_numeric_safe(s."eps (-4fysurprise %)")                                               AS eps_neg4fysurprise_pct,
       text_to_numeric_safe(s."eps (-5fyestimate)")                                                 AS eps_neg5fyestimate,
       text_to_numeric_safe(s."eps (-5fyactual)")                                                   AS eps_neg5fyactual,
       text_to_numeric_safe(s."eps (-5fysurprise %)")                                               AS eps_neg5fysurprise_pct,
       text_to_numeric_safe(s."eps (-0fqestimate)")                                                 AS eps_neg0fqestimate,
       text_to_numeric_safe(s."eps (-0fqactual)")                                                   AS eps_neg0fqactual,
       text_to_numeric_safe(s."eps (-0fqsurprise %)")                                               AS eps_neg0fqsurprise_pct,
       text_to_numeric_safe(s."eps (-1fqestimate)")                                                 AS eps_neg1fqestimate,
       text_to_numeric_safe(s."eps (-1fqactual)")                                                   AS eps_neg1fqactual,
       text_to_numeric_safe(s."eps (-1fqsurprise %)")                                               AS eps_neg1fqsurprise_pct,
       text_to_numeric_safe(s."eps (-2fqestimate)")                                                 AS eps_neg2fqestimate,
       text_to_numeric_safe(s."eps (-2fqactual)")                                                   AS eps_neg2fqactual,
       text_to_numeric_safe(s."eps (-2fqsurprise %)")                                               AS eps_neg2fqsurprise_pct,
       text_to_numeric_safe(s."eps (-3fqestimate)")                                                 AS eps_neg3fqestimate,
       text_to_numeric_safe(s."eps (-3fqactual)")                                                   AS eps_neg3fqactual,
       text_to_numeric_safe(s."eps (-3fqsurprise %)")                                               AS eps_neg3fqsurprise_pct,
       text_to_numeric_safe(s."eps (-4fqestimate)")                                                 AS eps_neg4fqestimate,
       text_to_numeric_safe(s."eps (-4fqactual)")                                                   AS eps_neg4fqactual,
       text_to_numeric_safe(s."eps (-4fqsurprise %)")                                               AS eps_neg4fqsurprise_pct,
       text_to_numeric_safe(s."fcf (ltm)")                                                          AS fcf_ltm,
       text_to_numeric_safe(s."fcf (fq)")                                                           AS fcf_fq,
       text_to_numeric_safe(s."fcf (-1fqfq)")                                                       AS fcf_neg1fqfq,
       text_to_numeric_safe(s."fcf (-3fqfq)")                                                       AS fcf_neg3fqfq,
       text_to_numeric_safe(s."fcf (-4fqfq)")                                                       AS fcf_neg4fqfq,
       text_to_numeric_safe(s."fcf (-2fqfq)")                                                       AS fcf_neg2fqfq,
       text_to_numeric_safe(s."fcf (fy)")                                                           AS fcf_fy,
       text_to_numeric_safe(s."fcf (-1fy)")                                                         AS fcf_neg1fy,
       text_to_numeric_safe(s."fcf (-3fy)")                                                         AS fcf_neg3fy,
       text_to_numeric_safe(s."fcf (-2fy)")                                                         AS fcf_neg2fy,
       text_to_numeric_safe(s."fcf (-4fy)")                                                         AS fcf_neg4fy,
       text_to_numeric_safe(s."fcf (-5fy)")                                                         AS fcf_neg5fy,
       text_to_numeric_safe(s."target % (avg)")                                                     AS target_pct_avg,
       text_to_numeric_safe(s."target % (med)")                                                     AS target_pct_med,
       text_to_numeric_safe(s."target % (low)")                                                     AS target_pct_low,
       text_to_numeric_safe(s."target % (high)")                                                    AS target_pct_high,
       text_to_numeric_safe(s."price target - stddev")                                              AS price_target_stddev,
       text_to_numeric_safe(s."price target - stddev (1w ago)")                                     AS price_target_stddev_1w_ago,
       text_to_numeric_safe(s."price target - stddev (1m ago)")                                     AS price_target_stddev_1m_ago,
       text_to_numeric_safe(s."price target - stddev (3m ago)")                                     AS price_target_stddev_3m_ago,
       text_to_numeric_safe(s."price target - stddev (6m ago)")                                     AS price_target_stddev_6m_ago,
       text_to_numeric_safe(s."price target - stddev (1y ago)")                                     AS price_target_stddev_1y_ago,
       text_to_numeric_safe(s."altman z-score (-1fy)")                                              AS altman_z_score_neg1fy,
       text_to_numeric_safe(s."altman z-score (-2fy)")                                              AS altman_z_score_neg2fy,
       text_to_numeric_safe(s."altman z-score (-3fy)")                                              AS altman_z_score_neg3fy,
       text_to_numeric_safe(s."altman z-score (-4fy)")                                              AS altman_z_score_neg4fy,
       text_to_numeric_safe(s."altman z-score (-5fy)")                                              AS altman_z_score_neg5fy,
       text_to_numeric_safe(s."altman z-score (-1fqfq)")                                            AS altman_z_score_neg1fqfq,
       text_to_numeric_safe(s."altman z-score (-2fqfq)")                                            AS altman_z_score_neg2fqfq,
       text_to_numeric_safe(s."altman z-score (-3fqfq)")                                            AS altman_z_score_neg3fqfq,
       text_to_numeric_safe(s."altman z-score (-4fqfq)")                                            AS altman_z_score_neg4fqfq,
       text_to_numeric_safe(s."altman z-score (-0fyyoyltm)")                                        AS altman_z_score_neg0fyyoyltm,
       text_to_numeric_safe(s."altman z-score (-1fyyoyltm)")                                        AS altman_z_score_neg1fyyoyltm,
       text_to_numeric_safe(s."altman z-score (-3fyyoyltm)")                                        AS altman_z_score_neg3fyyoyltm,
       text_to_numeric_safe(s."altman z-score (-4fyyoyltm)")                                        AS altman_z_score_neg4fyyoyltm,
       text_to_numeric_safe(s."altman z-score (-5fyyoyltm)")                                        AS altman_z_score_neg5fyyoyltm,
       text_to_numeric_safe(s."altman z-score (-2fyyoyltm)")                                        AS altman_z_score_neg2fyyoyltm,
       text_to_numeric_safe(s."p/e (est fy2)")                                                      AS p_e_est_fy2,
       text_to_numeric_safe(s."p/e (est fy3)")                                                      AS p_e_est_fy3,
       text_to_numeric_safe(s."p/e (est fy4)")                                                      AS p_e_est_fy4,
       text_to_numeric_safe(s."p/e (est fy5)")                                                      AS p_e_est_fy5,
       text_to_numeric_safe(s."p/e (-4fyltm)")                                                      AS p_e_neg4fyltm,
       text_to_numeric_safe(s."p/e (-4fqltm)")                                                      AS p_e_neg4fqltm,
       text_to_numeric_safe(s."p/e (3yavgntm)")                                                     AS p_e_3yavgntm,
       text_to_numeric_safe(s."p/e (5yavgntm)")                                                     AS p_e_5yavgntm,
       text_to_numeric_safe(s."eps norm - est avg (fq1e)")                                          AS eps_norm_est_avg_fq1e,
       text_to_numeric_safe(s."eps norm - est avg (fq2e)")                                          AS eps_norm_est_avg_fq2e,
       text_to_numeric_safe(s."eps norm - est avg (fq3e)")                                          AS eps_norm_est_avg_fq3e,
       text_to_numeric_safe(s."eps norm - est avg (fq4e)")                                          AS eps_norm_est_avg_fq4e,
       text_to_numeric_safe(s."eps norm - est avg (fy2e)")                                          AS eps_norm_est_avg_fy2e,
       text_to_numeric_safe(s."eps norm - est avg (fy3e)")                                          AS eps_norm_est_avg_fy3e,
       text_to_numeric_safe(s."eps norm - est avg (fy4e)")                                          AS eps_norm_est_avg_fy4e,
       text_to_numeric_safe(s."eps norm - est avg (fy5e)")                                          AS eps_norm_est_avg_fy5e,
       text_to_numeric_safe(s."capital expenditure (ltm)")                                          AS capital_expenditure_ltm,
       text_to_numeric_safe(s."capital expenditure (fq)")                                           AS capital_expenditure_fq,
       text_to_numeric_safe(s."capital expenditure (fy)")                                           AS capital_expenditure_fy,
       text_to_numeric_safe(s."capital expenditure (-1fqfq)")                                       AS capital_expenditure_neg1fqfq,
       text_to_numeric_safe(s."capital expenditure (-2fqfq)")                                       AS capital_expenditure_neg2fqfq,
       text_to_numeric_safe(s."capital expenditure (-3fqfq)")                                       AS capital_expenditure_neg3fqfq,
       text_to_numeric_safe(s."capital expenditure (-4fqfq)")                                       AS capital_expenditure_neg4fqfq,
       text_to_numeric_safe(s."capital expenditure (-1fy)")                                         AS capital_expenditure_neg1fy,
       text_to_numeric_safe(s."capital expenditure (-2fy)")                                         AS capital_expenditure_neg2fy,
       text_to_numeric_safe(s."capital expenditure (-4fy)")                                         AS capital_expenditure_neg4fy,
       text_to_numeric_safe(s."capital expenditure (-3fy)")                                         AS capital_expenditure_neg3fy,
       text_to_numeric_safe(s."capital expenditure (-5fy)")                                         AS capital_expenditure_neg5fy,
       text_to_numeric_safe(s."cff (ltm)")                                                          AS cff_ltm,
       text_to_numeric_safe(s."cff (fq)")                                                           AS cff_fq,
       text_to_numeric_safe(s."cff (fy)")                                                           AS cff_fy,
       text_to_numeric_safe(s."cff (-1fqfq)")                                                       AS cff_neg1fqfq,
       text_to_numeric_safe(s."cff (-2fqfq)")                                                       AS cff_neg2fqfq,
       text_to_numeric_safe(s."cff (-3fqfq)")                                                       AS cff_neg3fqfq,
       text_to_numeric_safe(s."cff (-4fqfq)")                                                       AS cff_neg4fqfq,
       text_to_numeric_safe(s."cff (-1fy)")                                                         AS cff_neg1fy,
       text_to_numeric_safe(s."cff (-2fy)")                                                         AS cff_neg2fy,
       text_to_numeric_safe(s."cff (-3fy)")                                                         AS cff_neg3fy,
       text_to_numeric_safe(s."cff (-4fy)")                                                         AS cff_neg4fy,
       text_to_numeric_safe(s."cfi (ltm)")                                                          AS cfi_ltm,
       text_to_numeric_safe(s."cfi (fq)")                                                           AS cfi_fq,
       text_to_numeric_safe(s."cfi (fy)")                                                           AS cfi_fy,
       text_to_numeric_safe(s."cfi (-1fqfq)")                                                       AS cfi_neg1fqfq,
       text_to_numeric_safe(s."cfi (-2fqfq)")                                                       AS cfi_neg2fqfq,
       text_to_numeric_safe(s."cfi (-3fqfq)")                                                       AS cfi_neg3fqfq,
       text_to_numeric_safe(s."cfi (-4fqfq)")                                                       AS cfi_neg4fqfq,
       text_to_numeric_safe(s."cfi (-1fy)")                                                         AS cfi_neg1fy,
       text_to_numeric_safe(s."cfi (-2fy)")                                                         AS cfi_neg2fy,
       text_to_numeric_safe(s."cfi (-3fy)")                                                         AS cfi_neg3fy,
       text_to_numeric_safe(s."cfi (-5fy)")                                                         AS cfi_neg5fy,
       text_to_numeric_safe(s."cfi (-4fy)")                                                         AS cfi_neg4fy,
       text_to_numeric_safe(s."cfo (ltm)")                                                          AS cfo_ltm,
       text_to_numeric_safe(s."cfo (fq)")                                                           AS cfo_fq,
       text_to_numeric_safe(s."cfo (fy)")                                                           AS cfo_fy,
       text_to_numeric_safe(s."cfo (-1fqfq)")                                                       AS cfo_neg1fqfq,
       text_to_numeric_safe(s."cfo (-2fqfq)")                                                       AS cfo_neg2fqfq,
       text_to_numeric_safe(s."cfo (-4fqfq)")                                                       AS cfo_neg4fqfq,
       text_to_numeric_safe(s."cfo (-3fqfq)")                                                       AS cfo_neg3fqfq,
       text_to_numeric_safe(s."cfo (-1fy)")                                                         AS cfo_neg1fy,
       text_to_numeric_safe(s."cfo (-2fy)")                                                         AS cfo_neg2fy,
       text_to_numeric_safe(s."cfo (-3fy)")                                                         AS cfo_neg3fy,
       text_to_numeric_safe(s."cfo (-4fy)")                                                         AS cfo_neg4fy,
       text_to_numeric_safe(s."cfo (-5fy)")                                                         AS cfo_neg5fy,
       text_to_numeric_safe(s."dividend per share (fq)")                                            AS dividend_per_share_fq,
       text_to_numeric_safe(s."dividend per share (fy)")                                            AS dividend_per_share_fy,
       text_to_numeric_safe(s."dividend per share (-1fqfq)")                                        AS dividend_per_share_neg1fqfq,
       text_to_numeric_safe(s."dividend per share (-2fqfq)")                                        AS dividend_per_share_neg2fqfq,
       text_to_numeric_safe(s."dividend per share (-3fqfq)")                                        AS dividend_per_share_neg3fqfq,
       text_to_numeric_safe(s."dividend per share (-4fqfq)")                                        AS dividend_per_share_neg4fqfq,
       text_to_numeric_safe(s."dividend per share (-1fy)")                                          AS dividend_per_share_neg1fy,
       text_to_numeric_safe(s."dividend per share (-2fy)")                                          AS dividend_per_share_neg2fy,
       text_to_numeric_safe(s."dividend per share (-3fy)")                                          AS dividend_per_share_neg3fy,
       text_to_numeric_safe(s."dividend per share (-4fy)")                                          AS dividend_per_share_neg4fy,
       text_to_numeric_safe(s."dividend per share (-5fy)")                                          AS dividend_per_share_neg5fy,
       text_to_numeric_safe(s."enterprise value (-1fq)")                                            AS enterprise_value_neg1fq,
       text_to_numeric_safe(s."enterprise value (-2fq)")                                            AS enterprise_value_neg2fq,
       text_to_numeric_safe(s."enterprise value (-3fq)")                                            AS enterprise_value_neg3fq,
       text_to_numeric_safe(s."enterprise value (-4fq)")                                            AS enterprise_value_neg4fq,
       text_to_numeric_safe(s."enterprise value (-1fy)")                                            AS enterprise_value_neg1fy,
       text_to_numeric_safe(s."enterprise value (-2fy)")                                            AS enterprise_value_neg2fy,
       text_to_numeric_safe(s."enterprise value (-3fy)")                                            AS enterprise_value_neg3fy,
       text_to_numeric_safe(s."enterprise value (-4fy)")                                            AS enterprise_value_neg4fy,
       text_to_numeric_safe(s."enterprise value (-5fy)")                                            AS enterprise_value_neg5fy,
       text_to_numeric_safe(s."volatility (1m)")                                                    AS volatility_1m,
       text_to_numeric_safe(s."volatility (3m)")                                                    AS volatility_3m,
       text_to_numeric_safe(s."volatility (6m)")                                                    AS volatility_6m,
       text_to_numeric_safe(s."volatility (1y)")                                                    AS volatility_1y,
       text_to_numeric_safe(s."eps gaap est avg rev % (fy1e - 1w)")                                 AS eps_gaap_est_avg_rev_pct_fy1e_1w,
       text_to_numeric_safe(s."eps gaap est avg rev % (fy1e - mtd)")                                AS eps_gaap_est_avg_rev_pct_fy1e_mtd,
       text_to_numeric_safe(s."eps gaap est avg rev % (fy1e - qtd)")                                AS eps_gaap_est_avg_rev_pct_fy1e_qtd,
       text_to_numeric_safe(s."eps gaap est avg rev % (fy1e - ytd)")                                AS eps_gaap_est_avg_rev_pct_fy1e_ytd,
       text_to_numeric_safe(s."price target - stddev (mtd ago)")                                    AS price_target_stddev_mtd_ago,
       text_to_numeric_safe(s."price target - stddev (qtd ago)")                                    AS price_target_stddev_qtd_ago,
       text_to_numeric_safe(s."price target - stddev (ytd ago)")                                    AS price_target_stddev_ytd_ago,
       text_to_numeric_safe(s."ebit (-0fqestimate)")                                                AS ebit_neg0fqestimate,
       text_to_numeric_safe(s."ebit (-1fqestimate)")                                                AS ebit_neg1fqestimate,
       text_to_numeric_safe(s."ebit (-2fqestimate)")                                                AS ebit_neg2fqestimate,
       text_to_numeric_safe(s."ebit (-3fqestimate)")                                                AS ebit_neg3fqestimate,
       text_to_numeric_safe(s."ebit (-4fqestimate)")                                                AS ebit_neg4fqestimate,
       text_to_numeric_safe(s."ebit (-0fqactual)")                                                  AS ebit_neg0fqactual,
       text_to_numeric_safe(s."ebit (-1fqactual)")                                                  AS ebit_neg1fqactual,
       text_to_numeric_safe(s."ebit (-2fqactual)")                                                  AS ebit_neg2fqactual,
       text_to_numeric_safe(s."ebit (-3fqactual)")                                                  AS ebit_neg3fqactual,
       text_to_numeric_safe(s."ebit (-4fqactual)")                                                  AS ebit_neg4fqactual,
       text_to_numeric_safe(s."ebit (-0fyactual)")                                                  AS ebit_neg0fyactual,
       text_to_numeric_safe(s."ebit (-1fyactual)")                                                  AS ebit_neg1fyactual,
       text_to_numeric_safe(s."ebit (-2fyactual)")                                                  AS ebit_neg2fyactual,
       text_to_numeric_safe(s."ebit (-3fyactual)")                                                  AS ebit_neg3fyactual,
       text_to_numeric_safe(s."ebit (-4fyactual)")                                                  AS ebit_neg4fyactual,
       text_to_numeric_safe(s."ebit (-5fyactual)")                                                  AS ebit_neg5fyactual,
       text_to_numeric_safe(s."ebit (-0fyestimate)")                                                AS ebit_neg0fyestimate,
       text_to_numeric_safe(s."ebit (-1fyestimate)")                                                AS ebit_neg1fyestimate,
       text_to_numeric_safe(s."ebit (-2fyestimate)")                                                AS ebit_neg2fyestimate,
       text_to_numeric_safe(s."ebit (-3fyestimate)")                                                AS ebit_neg3fyestimate,
       text_to_numeric_safe(s."ebit (-4fyestimate)")                                                AS ebit_neg4fyestimate,
       text_to_numeric_safe(s."ebit (-5fyestimate)")                                                AS ebit_neg5fyestimate,
       text_to_numeric_safe(s."ebit (-0fqsurprise %)")                                              AS ebit_neg0fqsurprise_pct,
       text_to_numeric_safe(s."ebit (-1fqsurprise %)")                                              AS ebit_neg1fqsurprise_pct,
       text_to_numeric_safe(s."ebit (-2fqsurprise %)")                                              AS ebit_neg2fqsurprise_pct,
       text_to_numeric_safe(s."ebit (-3fqsurprise %)")                                              AS ebit_neg3fqsurprise_pct,
       text_to_numeric_safe(s."ebit (-4fqsurprise %)")                                              AS ebit_neg4fqsurprise_pct,
       text_to_numeric_safe(s."ebit (-0fysurprise %)")                                              AS ebit_neg0fysurprise_pct,
       text_to_numeric_safe(s."ebit (-1fysurprise %)")                                              AS ebit_neg1fysurprise_pct,
       text_to_numeric_safe(s."ebit (-2fysurprise %)")                                              AS ebit_neg2fysurprise_pct,
       text_to_numeric_safe(s."ebit (-3fysurprise %)")                                              AS ebit_neg3fysurprise_pct,
       text_to_numeric_safe(s."ebit (-4fysurprise %)")                                              AS ebit_neg4fysurprise_pct,
       text_to_numeric_safe(s."ebit (-5fysurprise %)")                                              AS ebit_neg5fysurprise_pct,
       text_to_numeric_safe(s."ebitda (-0fqestimate)")                                              AS ebitda_neg0fqestimate,
       text_to_numeric_safe(s."ebitda (-1fqestimate)")                                              AS ebitda_neg1fqestimate,
       text_to_numeric_safe(s."ebitda (-2fqestimate)")                                              AS ebitda_neg2fqestimate,
       text_to_numeric_safe(s."ebitda (-3fqestimate)")                                              AS ebitda_neg3fqestimate,
       text_to_numeric_safe(s."ebitda (-4fqestimate)")                                              AS ebitda_neg4fqestimate,
       text_to_numeric_safe(s."ebitda (-0fyestimate)")                                              AS ebitda_neg0fyestimate,
       text_to_numeric_safe(s."ebitda (-1fyestimate)")                                              AS ebitda_neg1fyestimate,
       text_to_numeric_safe(s."ebitda (-2fyestimate)")                                              AS ebitda_neg2fyestimate,
       text_to_numeric_safe(s."ebitda (-3fyestimate)")                                              AS ebitda_neg3fyestimate,
       text_to_numeric_safe(s."ebitda (-4fyestimate)")                                              AS ebitda_neg4fyestimate,
       text_to_numeric_safe(s."ebitda (-5fyestimate)")                                              AS ebitda_neg5fyestimate,
       text_to_numeric_safe(s."ebitda (-0fqactual)")                                                AS ebitda_neg0fqactual,
       text_to_numeric_safe(s."ebitda (-1fqactual)")                                                AS ebitda_neg1fqactual,
       text_to_numeric_safe(s."ebitda (-2fqactual)")                                                AS ebitda_neg2fqactual,
       text_to_numeric_safe(s."ebitda (-3fqactual)")                                                AS ebitda_neg3fqactual,
       text_to_numeric_safe(s."ebitda (-4fqactual)")                                                AS ebitda_neg4fqactual,
       text_to_numeric_safe(s."ebitda (-0fyactual)")                                                AS ebitda_neg0fyactual,
       text_to_numeric_safe(s."ebitda (-1fyactual)")                                                AS ebitda_neg1fyactual,
       text_to_numeric_safe(s."ebitda (-2fyactual)")                                                AS ebitda_neg2fyactual,
       text_to_numeric_safe(s."ebitda (-3fyactual)")                                                AS ebitda_neg3fyactual,
       text_to_numeric_safe(s."ebitda (-4fyactual)")                                                AS ebitda_neg4fyactual,
       text_to_numeric_safe(s."ebitda (-5fyactual)")                                                AS ebitda_neg5fyactual,
       text_to_numeric_safe(s."ebitda (-0fqsurprise %)")                                            AS ebitda_neg0fqsurprise_pct,
       text_to_numeric_safe(s."ebitda (-1fqsurprise %)")                                            AS ebitda_neg1fqsurprise_pct,
       text_to_numeric_safe(s."ebitda (-2fqsurprise %)")                                            AS ebitda_neg2fqsurprise_pct,
       text_to_numeric_safe(s."ebitda (-3fqsurprise %)")                                            AS ebitda_neg3fqsurprise_pct,
       text_to_numeric_safe(s."ebitda (-4fqsurprise %)")                                            AS ebitda_neg4fqsurprise_pct,
       text_to_numeric_safe(s."ebitda (-0fysurprise %)")                                            AS ebitda_neg0fysurprise_pct,
       text_to_numeric_safe(s."ebitda (-1fysurprise %)")                                            AS ebitda_neg1fysurprise_pct,
       text_to_numeric_safe(s."ebitda (-2fysurprise %)")                                            AS ebitda_neg2fysurprise_pct,
       text_to_numeric_safe(s."ebitda (-3fysurprise %)")                                            AS ebitda_neg3fysurprise_pct,
       text_to_numeric_safe(s."ebitda (-4fysurprise %)")                                            AS ebitda_neg4fysurprise_pct,
       text_to_numeric_safe(s."ebitda (-5fysurprise %)")                                            AS ebitda_neg5fysurprise_pct,
       text_to_numeric_safe(s."price (1d ago)")                                                     AS price_1d_ago,
       text_to_numeric_safe(s."price (ytd ago)")                                                    AS price_ytd_ago,
       text_to_numeric_safe(s."price (mtd ago)")                                                    AS price_mtd_ago,
       text_to_numeric_safe(s."sales (-0fqestimate)")                                               AS sales_neg0fqestimate,
       text_to_numeric_safe(s."sales (-0fqactual)")                                                 AS sales_neg0fqactual,
       text_to_numeric_safe(s."sales (-0fqsurprise %)")                                             AS sales_neg0fqsurprise_pct,
       text_to_numeric_safe(s."sales (-1fqestimate)")                                               AS sales_neg1fqestimate,
       text_to_numeric_safe(s."sales (-1fqactual)")                                                 AS sales_neg1fqactual,
       text_to_numeric_safe(s."sales (-1fqsurprise %)")                                             AS sales_neg1fqsurprise_pct,
       text_to_numeric_safe(s."sales (-2fqestimate)")                                               AS sales_neg2fqestimate,
       text_to_numeric_safe(s."sales (-2fqactual)")                                                 AS sales_neg2fqactual,
       text_to_numeric_safe(s."sales (-2fqsurprise %)")                                             AS sales_neg2fqsurprise_pct,
       text_to_numeric_safe(s."sales (-3fqestimate)")                                               AS sales_neg3fqestimate,
       text_to_numeric_safe(s."sales (-3fqactual)")                                                 AS sales_neg3fqactual,
       text_to_numeric_safe(s."sales (-3fqsurprise %)")                                             AS sales_neg3fqsurprise_pct,
       text_to_numeric_safe(s."sales (-4fqestimate)")                                               AS sales_neg4fqestimate,
       text_to_numeric_safe(s."sales (-4fqactual)")                                                 AS sales_neg4fqactual,
       text_to_numeric_safe(s."sales (-4fqsurprise %)")                                             AS sales_neg4fqsurprise_pct,
       text_to_numeric_safe(s."sales (-0fyestimate)")                                               AS sales_neg0fyestimate,
       text_to_numeric_safe(s."sales (-0fyactual)")                                                 AS sales_neg0fyactual,
       text_to_numeric_safe(s."sales (-0fysurprise %)")                                             AS sales_neg0fysurprise_pct,
       text_to_numeric_safe(s."sales (-1fyestimate)")                                               AS sales_neg1fyestimate,
       text_to_numeric_safe(s."sales (-1fyactual)")                                                 AS sales_neg1fyactual,
       text_to_numeric_safe(s."sales (-1fysurprise %)")                                             AS sales_neg1fysurprise_pct,
       text_to_numeric_safe(s."sales (-2fyestimate)")                                               AS sales_neg2fyestimate,
       text_to_numeric_safe(s."sales (-2fyactual)")                                                 AS sales_neg2fyactual,
       text_to_numeric_safe(s."sales (-2fysurprise %)")                                             AS sales_neg2fysurprise_pct,
       text_to_numeric_safe(s."sales (-3fyestimate)")                                               AS sales_neg3fyestimate,
       text_to_numeric_safe(s."sales (-3fyactual)")                                                 AS sales_neg3fyactual,
       text_to_numeric_safe(s."sales (-3fysurprise %)")                                             AS sales_neg3fysurprise_pct,
       text_to_numeric_safe(s."sales (-4fyestimate)")                                               AS sales_neg4fyestimate,
       text_to_numeric_safe(s."sales (-4fyactual)")                                                 AS sales_neg4fyactual,
       text_to_numeric_safe(s."sales (-4fysurprise %)")                                             AS sales_neg4fysurprise_pct,
       text_to_numeric_safe(s."sales (-5fyestimate)")                                               AS sales_neg5fyestimate,
       text_to_numeric_safe(s."sales (-5fyactual)")                                                 AS sales_neg5fyactual,
       text_to_numeric_safe(s."sales (-5fysurprise %)")                                             AS sales_neg5fysurprise_pct,
       text_to_numeric_safe(s."fcf / share (ltm)")                                                  AS fcf_per_share_ltm,
       text_to_numeric_safe(s."fcf / share (fq)")                                                   AS fcf_per_share_fq,
       text_to_numeric_safe(s."fcf / share (fy)")                                                   AS fcf_per_share_fy,
       text_to_numeric_safe(s."fcf / share (-1fqfq)")                                               AS fcf_per_share_neg1fqfq,
       text_to_numeric_safe(s."fcf / share (-2fqfq)")                                               AS fcf_per_share_neg2fqfq,
       text_to_numeric_safe(s."fcf / share (-3fqfq)")                                               AS fcf_per_share_neg3fqfq,
       text_to_numeric_safe(s."fcf / share (-4fqfq)")                                               AS fcf_per_share_neg4fqfq,
       text_to_numeric_safe(s."fcf / share (-1fy)")                                                 AS fcf_per_share_neg1fy,
       text_to_numeric_safe(s."fcf / share (-2fy)")                                                 AS fcf_per_share_neg2fy,
       text_to_numeric_safe(s."fcf / share (-3fy)")                                                 AS fcf_per_share_neg3fy,
       text_to_numeric_safe(s."fcf / share (-4fy)")                                                 AS fcf_per_share_neg4fy,
       text_to_numeric_safe(s."fcf / share (-5fy)")                                                 AS fcf_per_share_neg5fy,
       text_to_numeric_safe(s."issuance of common stock (ltm)")                                     AS issuance_common_stock_ltm,
       text_to_numeric_safe(s."issuance of common stock (fq)")                                      AS issuance_common_stock_fq,
       text_to_numeric_safe(s."issuance of common stock (fy)")                                      AS issuance_common_stock_fy,
       text_to_numeric_safe(s."issuance of common stock (-1fqfq)")                                  AS issuance_common_stock_neg1fqfq,
       text_to_numeric_safe(s."issuance of common stock (-2fqfq)")                                  AS issuance_common_stock_neg2fqfq,
       text_to_numeric_safe(s."issuance of common stock (-3fqfq)")                                  AS issuance_common_stock_neg3fqfq,
       text_to_numeric_safe(s."issuance of common stock (-4fqfq)")                                  AS issuance_common_stock_neg4fqfq,
       text_to_numeric_safe(s."issuance of common stock (-1fy)")                                    AS issuance_common_stock_neg1fy,
       text_to_numeric_safe(s."issuance of common stock (-2fy)")                                    AS issuance_common_stock_neg2fy,
       text_to_numeric_safe(s."issuance of common stock (-3fy)")                                    AS issuance_common_stock_neg3fy,
       text_to_numeric_safe(s."issuance of common stock (-4fy)")                                    AS issuance_common_stock_neg4fy,
       text_to_numeric_safe(s."issuance of common stock (-5fy)")                                    AS issuance_common_stock_neg5fy,
       ROUND(text_to_numeric_safe(s."shrs out (-1fq)") / 1000000.0, 2)                              AS shrs_out_neg1fq,
       ROUND(text_to_numeric_safe(s."shrs out (-2fq)") / 1000000.0, 2)                              AS shrs_out_neg2fq,
       ROUND(text_to_numeric_safe(s."shrs out (-3fq)") / 1000000.0, 2)                              AS shrs_out_neg3fq,
       ROUND(text_to_numeric_safe(s."shrs out (-4fq)") / 1000000.0, 2)                              AS shrs_out_neg4fq,
       ROUND(text_to_numeric_safe(s."shrs out (-2fy)") / 1000000.0, 2)                              AS shrs_out_neg2fy,
       ROUND(text_to_numeric_safe(s."shrs out (-3fy)") / 1000000.0, 2)                              AS shrs_out_neg3fy,
       ROUND(text_to_numeric_safe(s."shrs out (-4fy)") / 1000000.0, 2)                              AS shrs_out_neg4fy,
       ROUND(text_to_numeric_safe(s."shrs out (-5fy)") / 1000000.0, 2)                              AS shrs_out_neg5fy,
       ROUND(text_to_numeric_safe(s."shrs out (3yavg)") / 1000000.0, 2)                             AS shrs_out_3yavg,
       ROUND(text_to_numeric_safe(s."shrs out (5yavg)") / 1000000.0, 2)                             AS shrs_out_5yavg,
       text_to_numeric_safe(s."repurchase of common stock (ltm)")                                   AS repurchase_common_stock_ltm,
       text_to_numeric_safe(s."repurchase of common stock (fq)")                                    AS repurchase_common_stock_fq,
       text_to_numeric_safe(s."repurchase of common stock (fy)")                                    AS repurchase_common_stock_fy,
       text_to_numeric_safe(s."repurchase of common stock (-1fqfq)")                                AS repurchase_common_stock_neg1fqfq,
       text_to_numeric_safe(s."repurchase of common stock (-2fqfq)")                                AS repurchase_common_stock_neg2fqfq,
       text_to_numeric_safe(s."repurchase of common stock (-3fqfq)")                                AS repurchase_common_stock_neg3fqfq,
       text_to_numeric_safe(s."repurchase of common stock (-4fqfq)")                                AS repurchase_common_stock_neg4fqfq,
       text_to_numeric_safe(s."repurchase of common stock (-1fy)")                                  AS repurchase_common_stock_neg1fy,
       text_to_numeric_safe(s."repurchase of common stock (-2fy)")                                  AS repurchase_common_stock_neg2fy,
       text_to_numeric_safe(s."repurchase of common stock (-3fy)")                                  AS repurchase_common_stock_neg3fy,
       text_to_numeric_safe(s."repurchase of common stock (-4fy)")                                  AS repurchase_common_stock_neg4fy,
       text_to_numeric_safe(s."repurchase of common stock (-5fy)")                                  AS repurchase_common_stock_neg5fy,
       text_to_numeric_safe(s."repurchase of common stock (3yavgfq)")                               AS repurchase_common_stock_3yavgfq,
       text_to_numeric_safe(s."repurchase of common stock (5yavgfq)")                               AS repurchase_common_stock_5yavgfq,
       text_to_numeric_safe(s."peg (ntm)")                                                          AS peg_ntm,
       text_to_numeric_safe(s."peg (-1fy)")                                                         AS peg_neg1fy,
       text_to_numeric_safe(s."peg (-2fy)")                                                         AS peg_neg2fy,
       text_to_numeric_safe(s."peg (-3fy)")                                                         AS peg_neg3fy,
       text_to_numeric_safe(s."peg (-4fy)")                                                         AS peg_neg4fy,
       text_to_numeric_safe(s."peg (-5fy)")                                                         AS peg_neg5fy,
       text_to_numeric_safe(s."peg (3yavg)")                                                        AS peg_3yavg,
       text_to_numeric_safe(s."peg (5yavg)")                                                        AS peg_5yavg,
       text_to_numeric_safe(s."gross profit margin % (fq)")                                         AS gross_profit_margin_pct_fq,
       text_to_numeric_safe(s."gross profit margin % (-1fqfq)")                                     AS gross_profit_margin_pct_neg1fqfq,
       text_to_numeric_safe(s."gross profit margin % (-2fqfq)")                                     AS gross_profit_margin_pct_neg2fqfq,
       text_to_numeric_safe(s."gross profit margin % (-3fqfq)")                                     AS gross_profit_margin_pct_neg3fqfq,
       text_to_numeric_safe(s."gross profit margin % (-4fqfq)")                                     AS gross_profit_margin_pct_neg4fqfq,
       text_to_numeric_safe(s."gross profit margin % (-1fy)")                                       AS gross_profit_margin_pct_neg1fy,
       text_to_numeric_safe(s."gross profit margin % (-2fy)")                                       AS gross_profit_margin_pct_neg2fy,
       text_to_numeric_safe(s."gross profit margin % (-3fy)")                                       AS gross_profit_margin_pct_neg3fy,
       text_to_numeric_safe(s."gross profit margin % (-4fy)")                                       AS gross_profit_margin_pct_neg4fy,
       text_to_numeric_safe(s."gross profit margin % (-5fy)")                                       AS gross_profit_margin_pct_neg5fy,
       text_to_numeric_safe(s."gross profit margin % (3yavgfq)")                                    AS gross_profit_margin_pct_3yavgfq,
       text_to_numeric_safe(s."gross profit margin % (5yavgfq)")                                    AS gross_profit_margin_pct_5yavgfq,
       text_to_numeric_safe(s."market cap (-1fq)")                                                  AS market_cap_neg1fq,
       text_to_numeric_safe(s."market cap (-2fq)")                                                  AS market_cap_neg2fq,
       text_to_numeric_safe(s."market cap (-3fq)")                                                  AS market_cap_neg3fq,
       text_to_numeric_safe(s."market cap (-4fq)")                                                  AS market_cap_neg4fq,
       text_to_numeric_safe(s."market cap (-1fy)")                                                  AS market_cap_neg1fy,
       text_to_numeric_safe(s."market cap (-2fy)")                                                  AS market_cap_neg2fy,
       text_to_numeric_safe(s."market cap (-3fy)")                                                  AS market_cap_neg3fy,
       text_to_numeric_safe(s."market cap (-4fy)")                                                  AS market_cap_neg4fy,
       text_to_numeric_safe(s."market cap (3yavg)")                                                 AS market_cap_3yavg,
       text_to_numeric_safe(s."market cap (5yavg)")                                                 AS market_cap_5yavg,
       text_to_numeric_safe(s."enterprise value (3yavg)")                                           AS enterprise_value_3yavg,
       text_to_numeric_safe(s."enterprise value (5yavg)")                                           AS enterprise_value_5yavg,
       text_to_numeric_safe(s."tot. return %/cagr (5y)")                                            AS tot_return_pct_cagr_5y,
       text_to_numeric_safe(s."tot. return %/cagr (1y)")                                            AS tot_return_pct_cagr_1y,
       text_to_numeric_safe(s."total return (1d)")                                                  AS total_return_1d,
       text_to_numeric_safe(s."total return (5d)")                                                  AS total_return_5d,
       text_to_numeric_safe(s."total return (1w)")                                                  AS total_return_1w,
       text_to_numeric_safe(s."total return (1m)")                                                  AS total_return_1m,
       text_to_numeric_safe(s."total return (3m)")                                                  AS total_return_3m,
       text_to_numeric_safe(s."total return (6m)")                                                  AS total_return_6m,
       text_to_numeric_safe(s."total return (1y)")                                                  AS total_return_1y,
       text_to_numeric_safe(s."total return (3y)")                                                  AS total_return_3y,
       text_to_numeric_safe(s."total return (mtd)")                                                 AS total_return_mtd,
       text_to_numeric_safe(s."total return (qtd)")                                                 AS total_return_qtd,
       text_to_numeric_safe(s."total return (2025)")                                                AS total_return_2025,
       text_to_numeric_safe(s."total return (2024)")                                                AS total_return_2024,
       text_to_numeric_safe(s."total return (2023)")                                                AS total_return_2023,
       text_to_numeric_safe(s."total return (2022)")                                                AS total_return_2022,
       text_to_numeric_safe(s."total return (2021)")                                                AS total_return_2021,
       text_to_numeric_safe(s."return on assets (roa) % (fq)")                                      AS return_on_assets_roa_pct_fq,
       text_to_numeric_safe(s."return on assets (roa) % (-1fqfq)")                                  AS return_on_assets_roa_pct_neg1fqfq,
       text_to_numeric_safe(s."return on assets (roa) % (-2fqfq)")                                  AS return_on_assets_roa_pct_neg2fqfq,
       text_to_numeric_safe(s."return on assets (roa) % (-3fqfq)")                                  AS return_on_assets_roa_pct_neg3fqfq,
       text_to_numeric_safe(s."return on assets (roa) % (-4fqfq)")                                  AS return_on_assets_roa_pct_neg4fqfq,
       text_to_numeric_safe(s."return on assets (roa) % (-1fy)")                                    AS return_on_assets_roa_pct_neg1fy,
       text_to_numeric_safe(s."return on assets (roa) % (-2fy)")                                    AS return_on_assets_roa_pct_neg2fy,
       text_to_numeric_safe(s."return on assets (roa) % (-3fy)")                                    AS return_on_assets_roa_pct_neg3fy,
       text_to_numeric_safe(s."return on assets (roa) % (-4fy)")                                    AS return_on_assets_roa_pct_neg4fy,
       text_to_numeric_safe(s."asset turnover (fq)")                                                AS asset_turnover_fq,
       text_to_numeric_safe(s."asset turnover (fy)")                                                AS asset_turnover_fy,
       text_to_numeric_safe(s."asset turnover (-1fqfq)")                                            AS asset_turnover_neg1fqfq,
       text_to_numeric_safe(s."asset turnover (-2fqfq)")                                            AS asset_turnover_neg2fqfq,
       text_to_numeric_safe(s."asset turnover (-3fqfq)")                                            AS asset_turnover_neg3fqfq,
       text_to_numeric_safe(s."asset turnover (-4fqfq)")                                            AS asset_turnover_neg4fqfq,
       text_to_numeric_safe(s."asset turnover (-1fy)")                                              AS asset_turnover_neg1fy,
       text_to_numeric_safe(s."asset turnover (-2fy)")                                              AS asset_turnover_neg2fy,
       text_to_numeric_safe(s."asset turnover (-3fy)")                                              AS asset_turnover_neg3fy,
       text_to_numeric_safe(s."asset turnover (-4fy)")                                              AS asset_turnover_neg4fy,
       text_to_numeric_safe(s."quick ratio (ltm)")                                                  AS quick_ratio_ltm,
       text_to_numeric_safe(s."quick ratio (fq)")                                                   AS quick_ratio_fq,
       text_to_numeric_safe(s."quick ratio (fy)")                                                   AS quick_ratio_fy,
       text_to_numeric_safe(s."quick ratio (-1fqfq)")                                               AS quick_ratio_neg1fqfq,
       text_to_numeric_safe(s."quick ratio (-2fqfq)")                                               AS quick_ratio_neg2fqfq,
       text_to_numeric_safe(s."quick ratio (-3fqfq)")                                               AS quick_ratio_neg3fqfq,
       text_to_numeric_safe(s."quick ratio (-4fqfq)")                                               AS quick_ratio_neg4fqfq,
       text_to_numeric_safe(s."quick ratio (-1fy)")                                                 AS quick_ratio_neg1fy,
       text_to_numeric_safe(s."quick ratio (-2fy)")                                                 AS quick_ratio_neg2fy,
       text_to_numeric_safe(s."quick ratio (-3fy)")                                                 AS quick_ratio_neg3fy,
       text_to_numeric_safe(s."quick ratio (-4fy)")                                                 AS quick_ratio_neg4fy,
       text_to_numeric_safe(s."current ratio (ltm)")                                                AS current_ratio_ltm,
       text_to_numeric_safe(s."current ratio (fq)")                                                 AS current_ratio_fq,
       text_to_numeric_safe(s."current ratio (fy)")                                                 AS current_ratio_fy,
       text_to_numeric_safe(s."current ratio (-1fqfq)")                                             AS current_ratio_neg1fqfq,
       text_to_numeric_safe(s."current ratio (-2fqfq)")                                             AS current_ratio_neg2fqfq,
       text_to_numeric_safe(s."current ratio (-3fqfq)")                                             AS current_ratio_neg3fqfq,
       text_to_numeric_safe(s."current ratio (-4fqfq)")                                             AS current_ratio_neg4fqfq,
       text_to_numeric_safe(s."current ratio (-1fy)")                                               AS current_ratio_neg1fy,
       text_to_numeric_safe(s."current ratio (-2fy)")                                               AS current_ratio_neg2fy,
       text_to_numeric_safe(s."current ratio (-3fy)")                                               AS current_ratio_neg3fy,
       text_to_numeric_safe(s."current ratio (-4fy)")                                               AS current_ratio_neg4fy,
       text_to_numeric_safe(s."long term debt / equity (ltm)")                                      AS long_term_debt_equity_ltm,
       text_to_numeric_safe(s."long term debt / equity (fq)")                                       AS long_term_debt_equity_fq,
       text_to_numeric_safe(s."long term debt / equity (fy)")                                       AS long_term_debt_equity_fy,
       text_to_numeric_safe(s."long term debt / equity (-1fqfq)")                                   AS long_term_debt_equity_neg1fqfq,
       text_to_numeric_safe(s."long term debt / equity (-2fqfq)")                                   AS long_term_debt_equity_neg2fqfq,
       text_to_numeric_safe(s."long term debt / equity (-3fqfq)")                                   AS long_term_debt_equity_neg3fqfq,
       text_to_numeric_safe(s."long term debt / equity (-4fqfq)")                                   AS long_term_debt_equity_neg4fqfq,
       text_to_numeric_safe(s."long term debt / equity (-1fy)")                                     AS long_term_debt_equity_neg1fy,
       text_to_numeric_safe(s."long term debt / equity (-2fy)")                                     AS long_term_debt_equity_neg2fy,
       text_to_numeric_safe(s."long term debt / equity (-3fy)")                                     AS long_term_debt_equity_neg3fy,
       text_to_numeric_safe(s."long term debt / equity (-4fy)")                                     AS long_term_debt_equity_neg4fy,
       text_to_numeric_safe(s."net income - (is) (ltm)")                                            AS net_income_ltm,
       text_to_numeric_safe(s."net income - (is) (fq)")                                             AS net_income_fq,
       text_to_numeric_safe(s."net income - (is) (fy)")                                             AS net_income_fy,
       text_to_numeric_safe(s."net income - (is) (-1fqfq)")                                         AS net_income_neg1fqfq,
       text_to_numeric_safe(s."net income - (is) (-2fqfq)")                                         AS net_income_neg2fqfq,
       text_to_numeric_safe(s."net income - (is) (-3fqfq)")                                         AS net_income_neg3fqfq,
       text_to_numeric_safe(s."net income - (is) (-4fqfq)")                                         AS net_income_neg4fqfq,
       text_to_numeric_safe(s."net income - (is) (-1fy)")                                           AS net_income_neg1fy,
       text_to_numeric_safe(s."net income - (is) (-2fy)")                                           AS net_income_neg2fy,
       text_to_numeric_safe(s."net income - (is) (-3fy)")                                           AS net_income_neg3fy,
       text_to_numeric_safe(s."net income - (is) (-4fy)")                                           AS net_income_neg4fy


FROM staging_header_buf s
;
-- FINAL VALIDATION
-- ===================================================================
\echo 'Final validation...'
SELECT 'Total rows in pml_df:' AS info, COUNT(*) AS count
FROM pml_df
;
SELECT 'Rows by Trading Region:' AS info, trading_region, COUNT(*) AS count
FROM pml_df
GROUP BY trading_region
ORDER BY trading_region
;
SELECT 'Rows by Sector (top 10):' AS info, sector, COUNT(*) AS count
FROM pml_df
GROUP BY sector
ORDER BY COUNT(*) DESC
LIMIT 10
;

-- Unmapped reference codes: the *_name() lookups fall back to the raw
-- code, so name = code flags a code missing from its reference table.
-- Expected result: zero rows. Extend the relevant *_ref table if any appear.
SELECT 'Unmapped reference codes:' AS info, kind, code, COUNT(*) AS count
FROM (SELECT 'country' AS kind, country AS code
      FROM pml_df
      WHERE country_name = country
	    AND country <> 'n/a'
      UNION ALL
      SELECT 'trading_country', trading_country
      FROM pml_df
      WHERE trading_country_name = trading_country
	    AND trading_country <> 'n/a'
      UNION ALL
      SELECT 'exchange', exchange
      FROM pml_df
      WHERE exchange_name = exchange
	    AND exchange <> 'n/a'
      UNION ALL
      SELECT 'currency', dividend_record_currency
      FROM pml_df
      WHERE dividend_record_currency_name = dividend_record_currency
	    AND dividend_record_currency <> 'n/a') unmapped
GROUP BY kind, code
ORDER BY kind, code
;

-- ===================================================================
-- CLEANUP
-- ===================================================================
DROP TABLE IF EXISTS staging_header_buf
;
DROP TABLE IF EXISTS country_ref
;
DROP TABLE IF EXISTS exchange_ref
;
DROP TABLE IF EXISTS currency_ref
;
DROP FUNCTION IF EXISTS country_name(TEXT)
;
DROP FUNCTION IF EXISTS exchange_name(TEXT)
;
DROP FUNCTION IF EXISTS currency_name(TEXT)
;
DROP FUNCTION IF EXISTS text_to_numeric_safe(TEXT)
;
DROP FUNCTION IF EXISTS text_to_date_safe(TEXT, TEXT)
;
DROP FUNCTION IF EXISTS month_abbrev_to_number(TEXT)
;
DROP FUNCTION IF EXISTS get_expected_reporting_lag_days(TEXT)
;
DROP FUNCTION IF EXISTS parse_fiscal_year_end_date(TEXT)
;
DROP FUNCTION IF EXISTS frequency_to_months(TEXT)
;
DROP FUNCTION IF EXISTS months_to_frequency(INTEGER)
;
DROP FUNCTION IF EXISTS derive_earnings_report_frequency(DATE, DATE)
;
DROP FUNCTION IF EXISTS calculate_fiscal_info(DATE, DATE, TEXT)
;
DROP FUNCTION IF EXISTS calculate_next_income_statement_report_date(DATE, TEXT)
;
DROP FUNCTION IF EXISTS calculate_next_fy_end_date(DATE)
;
DROP FUNCTION IF EXISTS calculate_next_fiscal_quarter(INTEGER, TEXT)
;
DROP FUNCTION IF EXISTS calculate_reporting_lag(DATE, DATE, TEXT)
;
DROP FUNCTION IF EXISTS calculate_expected_report_date(DATE, TEXT)
;
DROP FUNCTION IF EXISTS validate_fiscal_dates(DATE, DATE, DATE)
;

\echo 'Import complete!'