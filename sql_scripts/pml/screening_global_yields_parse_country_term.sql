-- Parses pml.screening_global_yields into a "country" and "term" column.
--
-- country: the alphabetic prefix of `ticker` (e.g. "AU1Y" -> "AU"). Tickers
-- are consistently "<country code><tenor><Y>" with no separators, so this is
-- more reliable than parsing the country out of the free-text `name`.
--
-- term: the vendor `name` values are not uniform for the tenor either. Three
-- shapes are present:
--   1. "Australia 1Y Bond Yield"
--   2. "Australia Government Bond 10Y"   (and lower-case "...Bond 10y" for some rows)
--   3. "US Treasury yield 7 year"        (single vendor one-off)
-- Strategy: pull the trailing "<digits><Y|y|year(s)>" token out of `name` and
-- normalise it to e.g. "1Y", "10Y".

-- 1. Preview the parse before touching the table.
SELECT
    ticker,
    name,
    upper(substring(ticker FROM '^[A-Za-z]+')) AS country_preview,
    upper((regexp_match(name, '(\d+)\s*(?:years?|y)\y', 'i'))[1]) || 'Y' AS term_preview
FROM pml.screening_global_yields
ORDER BY ticker;

-- 2. Add the new columns.
ALTER TABLE pml.screening_global_yields
    ADD COLUMN IF NOT EXISTS country text,
    ADD COLUMN IF NOT EXISTS term text;

-- 3. Populate them.
UPDATE pml.screening_global_yields
SET
    country = upper(substring(ticker FROM '^[A-Za-z]+')),
    term = upper((regexp_match(name, '(\d+)\s*(?:years?|y)\y', 'i'))[1]) || 'Y';

-- 4. Verify: every row should have both columns populated, and country ||
-- term should reconstruct the ticker exactly.
SELECT ticker, name, country, term
FROM pml.screening_global_yields
WHERE country IS NULL
   OR term IS NULL
   OR ticker <> (country || term)
ORDER BY ticker;
