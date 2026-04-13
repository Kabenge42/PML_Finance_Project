-- ============================================================
-- SNB Currency Data: CSV Import + D1 Splitting (currency/unit)
-- ============================================================

DROP TABLE IF EXISTS currencies;
CREATE TABLE currencies
(
    "Date"         TEXT,
    D0             TEXT,
    D1             TEXT,
    currency       TEXT,
    unit           NUMERIC,
    "Value"        DOUBLE PRECISION,
    reference_date DATE
);

-- Function: parse YYYY-MM to end-of-month date
CREATE OR REPLACE FUNCTION parse_year_month_to_end_of_month(date_text TEXT)
    RETURNS DATE AS
$$
DECLARE
    year_val  INTEGER;
    month_val INTEGER;
BEGIN
    IF date_text IS NULL OR TRIM(date_text) = '' THEN
        RETURN NULL;
    END IF;

    IF date_text !~ '^\d{4}-\d{2}$' THEN
        RETURN NULL;
    END IF;

    year_val := SPLIT_PART(date_text, '-', 1)::INTEGER;
    month_val := SPLIT_PART(date_text, '-', 2)::INTEGER;

    IF year_val < 1900 OR year_val > 2100 OR month_val < 1 OR month_val > 12 THEN
        RETURN NULL;
    END IF;

    RETURN (MAKE_DATE(year_val, month_val, 1) + INTERVAL '1 month - 1 day')::DATE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function: extract non-numeric currency code from D1
-- e.g. 'CZK100' → 'CZK', 'USD3M' → 'USD', 'EUR1' → 'EUR'
CREATE OR REPLACE FUNCTION extract_currency(d1_value TEXT)
    RETURNS TEXT AS
$$
BEGIN
    RETURN REGEXP_REPLACE(TRIM(d1_value), '[0-9]+[A-Za-z]*$', '');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function: extract numeric unit from D1
-- e.g. 'CZK100' → 100, 'EUR1' → 1, 'USD3M' → NULL (non-standard)
CREATE OR REPLACE FUNCTION extract_unit(d1_value TEXT)
    RETURNS NUMERIC AS
$$
DECLARE
    num_part TEXT;
BEGIN
    num_part := SUBSTRING(TRIM(d1_value) FROM '([0-9]+)');
    IF num_part IS NULL OR num_part = '' THEN
        RETURN NULL;
    END IF;
    RETURN num_part::NUMERIC;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================
-- Load CSV via staging table (semicolon-delimited, quoted)
-- ============================================================
DROP TABLE IF EXISTS currencies_staging;
CREATE TEMP TABLE currencies_staging
(
    "Date"         TEXT,
    D0             TEXT,
    D1             TEXT,
    "Value"        TEXT,
    reference_date DATE
);

-- Import CSV — adjust the path to your server-accessible location
COPY currencies_staging ("Date", D0, D1, "Value")
    FROM 'C:/Users/markm/PycharmProjects/Finance_Analytics_Platform/snb_data/snb-data-devkum.csv'
    WITH (
    FORMAT CSV,
    DELIMITER ';',
    HEADER TRUE,
    QUOTE '"'
    );

-- Insert into final table with D1 split into currency + unit
INSERT INTO currencies ("Date", D0, D1, currency, unit, "Value")
SELECT TRIM(s."Date"),
       TRIM(s.D0),
       TRIM(s.D1),
       extract_currency(s.D1),
       extract_unit(s.D1),
       NULLIF(TRIM(s."Value"), '')::DOUBLE PRECISION
FROM currencies_staging s;

DROP TABLE IF EXISTS currencies_staging;

-- ============================================================
-- Verify: sample rows
-- ============================================================
SELECT "Date",
       D0,
       D1,
       currency,
       unit,
       "Value",
       parse_year_month_to_end_of_month("Date") AS reference_date
FROM currencies
LIMIT 20;
