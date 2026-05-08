-- =============================================================================
-- SQL Feature Registry for PML Finance Project
-- Phase 9.3 Feature Engineering - PostgreSQL Implementation (OPTIMIZED)
-- =============================================================================
-- OPTIMIZATIONS APPLIED:
-- 1. STABLE modifier on all functions (enables query optimizer caching)
-- 2. Optional isin parameter for filtered access (uses idx_equities_isin)
-- 3. Materialized views for comprehensive functions
-- 4. PARALLEL SAFE where applicable
-- 5. Helper functions for common calculations (DRY principle)
-- =============================================================================

-- =============================================================================
-- HELPER FUNCTIONS: Extracted Common Calculations
-- =============================================================================

-- Safe division helper (avoids division by zero)
CREATE OR REPLACE FUNCTION safe_divide(
    numerator NUMERIC,
    denominator NUMERIC
)
    RETURNS NUMERIC
    IMMUTABLE
    PARALLEL SAFE
    LANGUAGE SQL
AS
$$
SELECT numerator / NULLIF(denominator, 0) AS result;
$$;

-- Percentage change helper
CREATE OR REPLACE FUNCTION public.pct_change(current_val NUMERIC, previous_val NUMERIC)
    RETURNS NUMERIC
    IMMUTABLE
    PARALLEL SAFE
AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) * 100 AS result;
$$ LANGUAGE SQL;

-- Momentum/change ratio helper (without percentage multiplier)
CREATE OR REPLACE FUNCTION public.calc_change_ratio(current_val NUMERIC, previous_val NUMERIC)
    RETURNS NUMERIC
    IMMUTABLE
    PARALLEL SAFE
AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) AS result;
$$ LANGUAGE SQL;

-- Score clamping helper (constrains value between 0 and 100)
CREATE OR REPLACE FUNCTION public.clamp_score(val NUMERIC, min_val NUMERIC DEFAULT 0, max_val NUMERIC DEFAULT 100)
    RETURNS NUMERIC
    IMMUTABLE
    PARALLEL SAFE
AS
$$
SELECT GREATEST(min_val, LEAST(max_val, val)) AS result;
$$ LANGUAGE SQL;

-- EMA crossover signal helper
CREATE OR REPLACE FUNCTION public.ema_crossover_signal(fast_ema NUMERIC, slow_ema NUMERIC)
    RETURNS INTEGER
    IMMUTABLE
    PARALLEL SAFE
AS
$$
SELECT CASE
           WHEN fast_ema > slow_ema THEN 1
           WHEN fast_ema < slow_ema THEN -1
           ELSE 0
           END AS result;
$$ LANGUAGE SQL;

-- =============================================================================
-- IDENTIFIER COLUMNS FROM CALCULATED_FEATURES_REGISTRY
-- =============================================================================
-- This view selects identifier columns based on the registry configuration
-- Used as the base for all feature views to ensure consistent identifier ordering

CREATE OR REPLACE VIEW vw_identifier_columns AS
SELECT e."ISIN"                              AS isin,
       e."Ticker"                            AS ticker,
       e."Name"                              AS name,
       e."Description"                       AS description,
       e."Region"                            AS region,
       e."Country"                           AS country,
       e."Trading Country"                   AS trading_country,
       e."Exchange"                          AS exchange,
       e."Sector"                            AS sector,
       e."Industry"                          AS industry,

       -- CATEGORICAL columns from equities_schema_metadata
       e."Dividend Record (Frequency)"       AS dividend_record_frequency,
       e."Earnings Report (Frequency)"       AS earnings_report_frequency,
       e."FY End"                            AS fy_end,
       e."Next Earnings (Report)"            AS next_earnings_report,
       e."Next Earnings (Status)"            AS next_earnings_status,
       e."Next Earnings (When)"              AS next_earnings_when,
       e."Next Fiscal Quarter"               AS next_fiscal_quarter,
       e."Reporting Interval"                AS reporting_interval,
       e."Size Class"                        AS size_class,
       e."Style Class"                       AS style_class,
       e."Unit"                              AS unit,

       -- DATE columns from equities_schema_metadata
       e."Dividend Record (Announce Date)"   AS dividend_record_announce_date,
       e."Dividend Record (Ex Date)"         AS dividend_record_ex_date,
       e."Dividend Record (Payable Date)"    AS dividend_record_payable_date,
       e."Dividend Record (Record Date)"     AS dividend_record_record_date,
       e."FY End Date"                       AS fy_end_date,
       e."Income Statement Report Date"      AS income_statement_report_date,
       e."Last Updated"                      AS last_updated,
       e."Next Earnings"                     AS next_earnings,
       e."Next FY End Date"                  AS next_fy_end_date,
       e."Next Income Statement Report Date" AS next_income_statement_report_date,
       e."Reference Date"                    AS reference_date

FROM postgres.public.equities e;

-- =============================================================================
-- SECTION 1: VALUATION FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_valuation_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin            TEXT,
                p_e_ratio       NUMERIC,
                p_b_ratio       NUMERIC,
                ev_ebitda_ratio NUMERIC,
                ev_sales_ratio  NUMERIC,
                dividend_yield  NUMERIC,
                peg_ratio       NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                     AS isin,
       "P/E (LTM)"::NUMERIC       AS p_e_ratio,
       "P/B (LTM)"::NUMERIC       AS p_b_ratio,
       "EV/EBITDA (LTM)"::NUMERIC AS ev_ebitda_ratio,
       "EV/Sales (LTM)"::NUMERIC  AS ev_sales_ratio,
       "Div Yield (LTM)"::NUMERIC AS dividend_yield,
       CASE
           WHEN "Net EPS - Basic (FY)" > 0 AND "Net EPS - Basic (-3FY)" > 0
               THEN public.safe_divide(
                   "P/E (LTM)"::NUMERIC,
                   ((POWER(
                             public.safe_divide("Net EPS - Basic (FY)"::NUMERIC, "Net EPS - Basic (-3FY)"::NUMERIC),
                             (1.0 / 3.0)::NUMERIC
                     ) - 1) * 100)::NUMERIC
                    )
           END                    AS peg_ratio
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_valuation_timeseries_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                       TEXT,
                ev_sales_trend_1y          NUMERIC,
                ev_ebitda_momentum         NUMERIC,
                p_e_momentum_yoy           NUMERIC,
                p_e_momentum_qoq           NUMERIC,
                ev_sales_vs_3y_avg         NUMERIC,
                ev_ebitda_vs_3y_avg        NUMERIC,
                p_e_vs_3y_avg              NUMERIC,
                ev_sales_forward_discount  NUMERIC,
                ev_ebitda_forward_discount NUMERIC,
                p_e_forward_discount       NUMERIC,
                p_b_vs_5y_avg              NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                                AS isin,
       public.calc_change_ratio("EV/Sales (LTM)"::NUMERIC, "EV/Sales (-1FYLTM)"::NUMERIC)    AS ev_sales_trend_1y,
       public.calc_change_ratio("EV/EBITDA (LTM)"::NUMERIC, "EV/EBITDA (-1FYLTM)"::NUMERIC)  AS ev_ebitda_momentum,
       public.calc_change_ratio("P/E (LTM)"::NUMERIC, "P/E (-1FYLTM)"::NUMERIC)              AS p_e_momentum_yoy,
       public.calc_change_ratio("P/E (LTM)"::NUMERIC, "P/E (-1FQLTM)"::NUMERIC)              AS p_e_momentum_qoq,
       public.calc_change_ratio("EV/Sales (LTM)"::NUMERIC, "EV/Sales (3YAVGLTM)"::NUMERIC)   AS ev_sales_vs_3y_avg,
       public.calc_change_ratio("EV/EBITDA (LTM)"::NUMERIC, "EV/EBITDA (3YAVGLTM)"::NUMERIC) AS ev_ebitda_vs_3y_avg,
       public.calc_change_ratio("P/E (LTM)"::NUMERIC, "P/E (3YAVGLTM)"::NUMERIC)             AS p_e_vs_3y_avg,
       public.calc_change_ratio("EV/Sales (NTM)"::NUMERIC,
                                "EV/Sales (LTM)"::NUMERIC)                                   AS ev_sales_forward_discount,
       public.calc_change_ratio("EV/EBITDA (NTM)"::NUMERIC,
                                "EV/EBITDA (LTM)"::NUMERIC)                                  AS ev_ebitda_forward_discount,
       public.calc_change_ratio("P/E (EST FY1)"::NUMERIC, "P/E (LTM)"::NUMERIC)              AS p_e_forward_discount,
       public.safe_divide("P/B (LTM)"::NUMERIC, "P/B (5YAVG)"::NUMERIC)                      AS p_b_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_extended_valuation_timeseries(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                     TEXT,
                ev_sales_qoq_1q          NUMERIC,
                ev_sales_qoq_2q          NUMERIC,
                ev_sales_qoq_3q          NUMERIC,
                ev_sales_qoq_4q          NUMERIC,
                p_e_vs_5y_avg            NUMERIC,
                p_e_percentile_proxy     NUMERIC,
                valuation_mean_reversion NUMERIC,
                ev_ebitda_qoq_trend      NUMERIC,
                p_b_momentum_yoy         NUMERIC,
                valuation_compression    NUMERIC,
                forward_pe_premium       NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                      AS isin,
       public.calc_change_ratio("EV/Sales (LTM)", "EV/Sales (-1FQLTM)")            AS ev_sales_qoq_1q,
       public.calc_change_ratio("EV/Sales (-1FQLTM)", "EV/Sales (-2FQLTM)")        AS ev_sales_qoq_2q,
       public.calc_change_ratio("EV/Sales (-2FQLTM)", "EV/Sales (-3FQLTM)")        AS ev_sales_qoq_3q,
       public.calc_change_ratio("EV/Sales (-3FQLTM)", "EV/Sales (-4FQLTM)")        AS ev_sales_qoq_4q,
       public.calc_change_ratio("P/E (LTM)", "P/E (5YAVGLTM)")                     AS p_e_vs_5y_avg,
       CASE
           WHEN "P/E (LTM)" IS NOT NULL AND "P/E (3YAVGLTM)" IS NOT NULL
               THEN ("P/E (LTM)" - "P/E (3YAVGLTM)") / NULLIF(ABS("P/E (3YAVGLTM)") * 0.5, 0)
           END                                                                     AS p_e_percentile_proxy,
       (public.calc_change_ratio("P/E (LTM)", "P/E (3YAVGLTM)") +
        public.calc_change_ratio("EV/Sales (LTM)", "EV/Sales (3YAVGLTM)") +
        public.calc_change_ratio("EV/EBITDA (LTM)", "EV/EBITDA (3YAVGLTM)")) / 3.0
                                                                                   AS valuation_mean_reversion,
       public.calc_change_ratio("EV/EBITDA (LTM)", "EV/EBITDA (-1FQLTM)")          AS ev_ebitda_qoq_trend,
       public.calc_change_ratio("P/B (LTM)", "P/B (-1FY)")                         AS p_b_momentum_yoy,
       (public.safe_divide("P/E (LTM)", "P/E (3YAVGLTM)") +
        public.safe_divide("EV/EBITDA (LTM)", "EV/EBITDA (3YAVGLTM)")) / 2.0 - 1.0 AS valuation_compression,
       public.calc_change_ratio("P/E (EST FY1)", "P/E (LTM)") * 100                AS forward_pe_premium
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 2: Technical Analysis FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_momentum_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                price_momentum_1m    NUMERIC,
                price_momentum_3m    NUMERIC,
                price_momentum_6m    NUMERIC,
                price_momentum_1y    NUMERIC,
                price_momentum_5d    NUMERIC,
                ema_crossover_20_50  INTEGER,
                ema_crossover_50_250 INTEGER,
                price_vs_ema_20d     NUMERIC,
                price_vs_ema_250d    NUMERIC,
                pct_off_52w_high     NUMERIC,
                pct_above_52w_low    NUMERIC,
                range_52w_position   NUMERIC,
                beta_momentum        NUMERIC,
                volatility_regime    NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                     AS isin,
       public.pct_change("Last Price"::NUMERIC, "Price (1M Ago)"::NUMERIC)        AS price_momentum_1m,
       public.pct_change("Last Price"::NUMERIC, "Price (3M Ago)"::NUMERIC)        AS price_momentum_3m,
       public.pct_change("Last Price"::NUMERIC, "Price (6M Ago)"::NUMERIC)        AS price_momentum_6m,
       public.pct_change("Last Price"::NUMERIC, "Price (1Y Ago)"::NUMERIC)        AS price_momentum_1y,
       public.pct_change("Last Price"::NUMERIC, "Price (5D Ago)"::NUMERIC)        AS price_momentum_5d,
       public.ema_crossover_signal("EMA (20D)"::NUMERIC, "EMA (50D)"::NUMERIC)    AS ema_crossover_20_50,
       public.ema_crossover_signal("EMA (50D)"::NUMERIC, "EMA (250D)"::NUMERIC)   AS ema_crossover_50_250,
       public.calc_change_ratio("Last Price"::NUMERIC, "EMA (20D)"::NUMERIC)      AS price_vs_ema_20d,
       public.calc_change_ratio("Last Price"::NUMERIC, "EMA (250D)"::NUMERIC)     AS price_vs_ema_250d,
       public.calc_change_ratio(("52W High/Adj"::NUMERIC - "Last Price"::NUMERIC),
                                "52W High/Adj"::NUMERIC)                          AS pct_off_52w_high,
       public.calc_change_ratio(("Last Price"::NUMERIC - "52W Low/Adj"::NUMERIC),
                                "52W Low/Adj"::NUMERIC)                           AS pct_above_52w_low,
       public.clamp_score(public.safe_divide(("Last Price"::NUMERIC - "52W Low/Adj"::NUMERIC),
                                             ("52W High/Adj"::NUMERIC - "52W Low/Adj"::NUMERIC)), 0,
                          1)                                                      AS range_52w_position,
       "Beta (1Y)"::NUMERIC - "Beta (5Y)"::NUMERIC                                AS beta_momentum,
       public.safe_divide("Volatility (1M)"::NUMERIC, "Volatility (1Y)"::NUMERIC) AS volatility_regime
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_technical_analysis_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                ema_slope_20d             NUMERIC,
                ema_trend_consistency     INTEGER,
                price_vs_ema_100d         NUMERIC,
                near_52w_high_flag        INTEGER,
                near_52w_low_flag         INTEGER,
                volume_momentum_score     NUMERIC,
                breakout_signal           INTEGER,
                high_volume_flag          INTEGER,
                low_volume_flag           INTEGER,
                volatility_compression    NUMERIC,
                volatility_term_structure NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                        AS isin,
       ("EMA (20D)" - "EMA (50D)") / NULLIF("EMA (50D)", 0)          AS ema_slope_20d,
       CASE
           WHEN "EMA (20D)" > "EMA (50D)" AND "EMA (50D)" > "EMA (100D)"
               AND "EMA (100D)" > "EMA (250D)" THEN 1
           WHEN "EMA (20D)" < "EMA (50D)" AND "EMA (50D)" < "EMA (100D)"
               AND "EMA (100D)" < "EMA (250D)" THEN -1
           ELSE 0
           END                                                       AS ema_trend_consistency,
       ("Last Price" - "EMA (100D)") / NULLIF("EMA (100D)", 0) * 100 AS price_vs_ema_100d,
       CASE
           WHEN ("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS near_52w_high_flag,
       CASE
           WHEN ("Last Price" - "52W Low/Adj") / NULLIF("52W Low/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS near_52w_low_flag,
       "Rel. Volume" * "Price Chg. % (1M)"                           AS volume_momentum_score,
       CASE
           WHEN "EMA (20D)" > "EMA (50D)"
               AND ("52W High/Adj" - "Last Price") / NULLIF("52W High/Adj", 0) <= 0.05
               THEN 1
           ELSE 0
           END                                                       AS breakout_signal,
       CASE WHEN "Rel. Volume" > 1.5 THEN 1 ELSE 0 END               AS high_volume_flag,
       CASE WHEN "Rel. Volume" < 0.5 THEN 1 ELSE 0 END               AS low_volume_flag,
       "Volatility (1Y)" - "Volatility (1M)"                         AS volatility_compression,
       "Volatility (3M)" - "Volatility (6M)"                         AS volatility_term_structure
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 3: PROFITABILITY FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_profitability_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                roe                  NUMERIC,
                roa                  NUMERIC,
                gross_margin_pct     NUMERIC,
                operating_margin_pct NUMERIC,
                net_margin_pct       NUMERIC,
                ebitda_margin_pct    NUMERIC,
                roic                 NUMERIC,
                rnd_intensity        NUMERIC,
                equity_multiplier    NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                             AS isin,
       "Return On Equity % (LTM)"                                         AS roe,
       "Return on Assets (ROA) % (LTM)"                                   AS roa,
       "Gross Profit Margin % (LTM)"                                      AS gross_margin_pct,
       "Operating Income (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100 AS operating_margin_pct,
       "Net Income Margin % (LTM)"                                        AS net_margin_pct,
       "EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100           AS ebitda_margin_pct,
       "EBIT (LTM)" * (1 - 0.25) / NULLIF("Total Equity (LTM)" + "Total Debt (LTM)" - "Cash And Equivalents (LTM)", 0) *
       100                                                                AS roic,
       "R&D Expenses (LTM)" / NULLIF("Total Revenues (LTM)", 0)           AS rnd_intensity,
       "Total Assets (LTM)" / NULLIF("Total Equity (LTM)", 0)             AS equity_multiplier
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_margin_trends(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                   TEXT,
                gross_margin_trend_yoy NUMERIC,
                operating_margin_trend NUMERIC,
                net_margin_trend_yoy   NUMERIC,
                ebitda_margin_trend    NUMERIC,
                margin_expansion_flag  INTEGER,
                margin_stability_score NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                               AS isin,
       ("Gross Profit Margin % (LTM)" - "Gross Profit Margin % (FY)")       AS gross_margin_trend_yoy,
       (("Operating Income (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
        ("Operating Income (FY)" / NULLIF("Total Revenues (FY)", 0))) * 100 AS operating_margin_trend,
       ("Net Income Margin % (LTM)" - "Net Income Margin % (FY)")           AS net_margin_trend_yoy,
       (("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0)) -
        ("EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0))) * 100           AS ebitda_margin_trend,
       CASE
           WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)"
               AND "Net Income Margin % (LTM)" > "Net Income Margin % (FY)"
               AND ("EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0)) >
                   ("EBITDA (FY)" / NULLIF("Total Revenues (FY)", 0))
               THEN 1
           ELSE 0
           END                                                              AS margin_expansion_flag,
       GREATEST(0, LEAST(100,
                         100 - (ABS("Gross Profit Margin % (LTM)" - "Gross Profit Margin % (FY)") +
                                ABS("Net Income Margin % (LTM)" - "Net Income Margin % (FY)")) / 2
                   ))                                                       AS margin_stability_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 4: QUALITY & RISK FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_quality_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                        TEXT,
                has_goodwill_impairment     INTEGER,
                has_asset_writedown         INTEGER,
                has_restructuring           INTEGER,
                goodwill_to_assets_pct      NUMERIC,
                intangible_intensity        NUMERIC,
                exceptional_items_to_ebitda NUMERIC,
                altman_z_score              NUMERIC,
                altman_z_trend              NUMERIC,
                current_ratio               NUMERIC,
                quick_ratio                 NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                                            AS isin,
       CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 1 ELSE 0 END                                   AS has_goodwill_impairment,
       CASE WHEN "Asset Writedown (LTM)" <> 0 THEN 1 ELSE 0 END                                          AS has_asset_writedown,
       CASE WHEN "Restructuring Charges (LTM)" <> 0 THEN 1 ELSE 0 END                                    AS has_restructuring,
       "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100                                          AS goodwill_to_assets_pct,
       "Gross Intangible Assets (LTM)" / NULLIF("Total Assets (LTM)", 0)                                 AS intangible_intensity,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) /
       NULLIF(ABS("EBITDA (LTM)"), 0)                                                                    AS exceptional_items_to_ebitda,
       "Altman Z-Score (LTM)"                                                                            AS altman_z_score,
       "Altman Z-Score (FY)" - "Altman Z-Score (LTM)"                                                    AS altman_z_trend,
       "Current Ratio (LTM)"                                                                             AS current_ratio,
       ("Total Current Assets (LTM)" - "Inventory (LTM)") / NULLIF("Total Current Liabilities (LTM)", 0) AS quick_ratio
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_financial_distress_features(p_isin TEXT DEFAULT NULL::TEXT)
    RETURNS TABLE
            (
                isin                     TEXT,
                distress_risk_score      NUMERIC,
                liquidity_stress_score   NUMERIC,
                working_capital_trend    NUMERIC,
                cash_runway_months       NUMERIC,
                combined_distress_score  NUMERIC,
                wc_deteriorating_flag    INTEGER,
                retained_earnings_growth NUMERIC,
                accumulated_deficit_flag INTEGER,
                adequate_cash_buffer     INTEGER
            )
    STABLE
    PARALLEL SAFE
    LANGUAGE sql
AS
$$
SELECT "ISIN"                                                   AS isin,

       -- distress_risk_score: unchanged
       GREATEST(0, LEAST(100,
                         (("Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 100)
                   ))                                           AS distress_risk_score,

       -- liquidity_stress_score: add a graduated middle band
       CASE
           WHEN "Current Ratio (LTM)" < 0.8 THEN 40.0
           WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
           WHEN "Current Ratio (LTM)" < 1.2 THEN 20.0
           WHEN "Current Ratio (LTM)" < 1.5 THEN 10.0
           ELSE 0.0
           END                                                  AS liquidity_stress_score,

       -- working_capital_trend: unchanged
       ("Working Capital (FQ)" - "Working Capital (FY)") /
       NULLIF(ABS("Working Capital (FY)"), 0)                   AS working_capital_trend,

       -- cash_runway_months: use NET cash burn (OpEx - Revenue) with floor at 1
       -- For profitable companies (Revenue > OpEx), runway is effectively infinite â†’ cap at 120
       CASE
           WHEN "Total Operating Expenses (LTM)" - COALESCE("Total Revenues (LTM)", 0) <= 0
               THEN 120.0 -- net cash-positive: no burn
           ELSE GREATEST(0,
                         "Cash And Equivalents (FQ)" /
                         NULLIF(("Total Operating Expenses (LTM)" - COALESCE("Total Revenues (LTM)", 0)) / 12.0, 0)
                )
           END                                                  AS cash_runway_months,

       -- combined_distress_score: unchanged formula (will benefit from improved sub-scores)
       GREATEST(0, LEAST(100,
                         (("Altman Z-Score (LTM)" - 1.8) / NULLIF(3.0 - 1.8, 0) * 70) +
                         (100 - CASE
                                    WHEN "Current Ratio (LTM)" < 0.8 THEN 40.0
                                    WHEN "Current Ratio (LTM)" < 1.0 THEN 30.0
                                    WHEN "Current Ratio (LTM)" < 1.2 THEN 20.0
                                    WHEN "Current Ratio (LTM)" < 1.5 THEN 10.0
                                    ELSE 0.0
                             END) * 0.30
                   ))                                           AS combined_distress_score,

       -- wc_deteriorating_flag: unchanged
       CASE
           WHEN ("Working Capital (FQ)" - "Working Capital (FY)") /
                NULLIF(ABS("Working Capital (FY)"), 0) < -0.2
               THEN 1
           ELSE 0
           END                                                  AS wc_deteriorating_flag,

       -- retained_earnings_growth: unchanged
       ("Retained Earnings (FQ)" - "Retained Earnings (FY)") /
       NULLIF(ABS("Retained Earnings (FY)"), 0)                 AS retained_earnings_growth,

       -- accumulated_deficit_flag: unchanged
       CASE WHEN "Retained Earnings (FQ)" < 0 THEN 1 ELSE 0 END AS accumulated_deficit_flag,

       -- adequate_cash_buffer: lower threshold to 3 months for net-burn basis
       CASE
           WHEN "Total Operating Expenses (LTM)" - COALESCE("Total Revenues (LTM)", 0) <= 0
               THEN 1 -- net cash-positive: always adequate
           WHEN "Cash And Equivalents (FQ)" /
                NULLIF(("Total Operating Expenses (LTM)" - COALESCE("Total Revenues (LTM)", 0)) / 12.0, 0) > 6
               THEN 1
           ELSE 0
           END                                                  AS adequate_cash_buffer

FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

CREATE OR REPLACE FUNCTION calc_accounting_quality_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                        TEXT,
                goodwill_change_rate        NUMERIC,
                restructuring_intensity     NUMERIC,
                exceptional_items_frequency INTEGER,
                merger_impact_ratio         NUMERIC,
                non_operating_income_share  NUMERIC,
                asset_sale_boost            INTEGER,
                accounting_quality_score    NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                AS isin,
       ("Goodwill (LTM)" - "Goodwill (-1FY)") / NULLIF("Goodwill (-1FY)", 0) AS goodwill_change_rate,
       "Restructuring Charges (LTM)" / NULLIF("Total Assets (LTM)", 0)       AS restructuring_intensity,
       (CASE WHEN ABS("Impairment of Goodwill (FQ)") > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS("Asset Writedown (FQ)") > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS("Restructuring Charges (FQ)") > 0 THEN 1 ELSE 0 END)   AS exceptional_items_frequency,
       "Merger & Restructuring Charges (LTM)" / NULLIF("Market Cap", 0)      AS merger_impact_ratio,
       "Interest And Investment Income (LTM)" / NULLIF(ABS("Net Income - (IS) (LTM)"), 0)
                                                                             AS non_operating_income_share,
       CASE WHEN "Gain (Loss) On Sale Of Assets (LTM)" > 0 THEN 1 ELSE 0 END AS asset_sale_boost,
       GREATEST(0, LEAST(100,
                         100 -
                         (CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 25 ELSE 0 END) -
                         (CASE WHEN "Asset Writedown (LTM)" <> 0 THEN 10 ELSE 0 END) -
                         (CASE WHEN "Restructuring Charges (LTM)" <> 0 THEN 15 ELSE 0 END) -
                         (CASE WHEN "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.30 THEN 15 ELSE 0 END) -
                         (CASE
                              WHEN (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
                                    ABS("Restructuring Charges (LTM)")) /
                                   NULLIF(ABS("Net Income - (IS) (LTM)"), 0) > 0.10 THEN 15
                              ELSE 0 END)
                   ))                                                        AS accounting_quality_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 5: LEVERAGE & LIQUIDITY FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_leverage_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                debt_to_equity        NUMERIC,
                debt_to_assets        NUMERIC,
                equity_ratio          NUMERIC,
                interest_coverage     NUMERIC,
                current_ratio         NUMERIC,
                cash_ratio            NUMERIC,
                working_capital_ratio NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                      AS isin,
       "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0)                        AS debt_to_equity,
       "Total Debt (LTM)" / NULLIF("Total Assets (LTM)", 0)                        AS debt_to_assets,
       "Total Equity (LTM)" / NULLIF("Total Assets (LTM)", 0)                      AS equity_ratio,
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)                    AS interest_coverage,
       "Current Ratio (LTM)"                                                       AS current_ratio,
       "Cash And Equivalents (LTM)" / NULLIF("Total Current Liabilities (LTM)", 0) AS cash_ratio,
       "Working Capital (LTM)" / NULLIF("Total Assets (LTM)", 0)                   AS working_capital_ratio
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_efficiency_ratios(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                asset_turnover        NUMERIC,
                inventory_turnover    NUMERIC,
                receivables_days      NUMERIC,
                working_capital_turns NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                        AS isin,
       "Total Revenues (LTM)" / NULLIF("Total Assets (LTM)", 0)                      AS asset_turnover,
       "Cost Of Revenues (LTM)" / NULLIF("Inventory (LTM)", 0)                       AS inventory_turnover,
       ("Accounts Receivable/Total (FY)" / NULLIF("Total Revenues (FY)" / 365.0, 0)) AS receivables_days,
       "Total Revenues (LTM)" / NULLIF("Working Capital (LTM)", 0)                   AS working_capital_turns
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_balance_sheet_dynamics(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                cash_to_assets_pct        NUMERIC,
                cash_change_qoq           NUMERIC,
                cash_vs_5y_avg            NUMERIC,
                inventory_change_yoy      NUMERIC,
                inventory_vs_5y_avg       NUMERIC,
                receivables_change_yoy    NUMERIC,
                receivables_vs_5y_avg     NUMERIC,
                working_capital_vs_5y_avg NUMERIC,
                retained_earnings_vs_5y   NUMERIC,
                intangibles_growth_flag   INTEGER,
                asset_quality_score       NUMERIC,
                balance_sheet_strength    NUMERIC,
                debt_maturity_risk        NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                    AS isin,
       "Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100      AS cash_to_assets_pct,
       ("Cash And Equivalents (FQ)" - "Cash And Equivalents (FY)") /
       NULLIF(ABS("Cash And Equivalents (FY)"), 0)                               AS cash_change_qoq,
       "Cash And Equivalents (FQ)" / NULLIF("Cash And Equivalents (5YAVGFQ)", 0) AS cash_vs_5y_avg,
       ("Inventory (FY)" - "Inventory (FQ)") / NULLIF(ABS("Inventory (FQ)"), 0)  AS inventory_change_yoy,
       "Inventory (FQ)" / NULLIF("Inventory (5YAVGFQ)", 0)                       AS inventory_vs_5y_avg,
       ("Accounts Receivable/Total (FY)" - "Accounts Receivable/Total (-1FY)") /
       NULLIF(ABS("Accounts Receivable/Total (-1FY)"), 0)                        AS receivables_change_yoy,
       "Accounts Receivable/Total (FY)" / NULLIF("Accounts Receivable/Total (5YAVGFQ)", 0)
                                                                                 AS receivables_vs_5y_avg,
       "Working Capital (FQ)" / NULLIF("Working Capital (5YAVGFY)", 0)           AS working_capital_vs_5y_avg,
       "Retained Earnings (FQ)" / NULLIF("Retained Earnings (5YAVGFQ)", 0)       AS retained_earnings_vs_5y,
       CASE
           WHEN "Gross Intangible Assets (FY)" / NULLIF("Gross Intangible Assets (5YAVGFQ)", 0) > 1.5
               THEN 1
           ELSE 0
           END                                                                   AS intangibles_growth_flag,
       GREATEST(0, LEAST(100,
                         50 + ("Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100) -
                         ("Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100)
                   ))                                                            AS asset_quality_score,
       GREATEST(0, LEAST(100,
                         (CASE
                              WHEN "Cash And Equivalents (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.10 THEN 25
                              ELSE 0 END) +
                         (CASE WHEN "Total Equity (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.40 THEN 25 ELSE 0 END) +
                         (CASE WHEN "Working Capital (LTM)" > 0 THEN 25 ELSE 0 END) +
                         (CASE WHEN "Current Ratio (LTM)" > 1.5 THEN 25 ELSE 0 END)
                   ))                                                            AS balance_sheet_strength,
       "Total Debt (LTM)" / NULLIF("EBITDA (LTM)", 0)                            AS debt_maturity_risk
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;


-- =============================================================================
-- SECTION 6: ANALYST SENTIMENT FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_sentiment_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                analyst_bullish_pct       NUMERIC,
                analyst_bearish_pct       NUMERIC,
                analyst_neutral_pct       NUMERIC, -- NEW: Hold ratings percentage
                analyst_conviction        NUMERIC,
                upside_potential          NUMERIC,
                price_target_spread_pct   NUMERIC,
                price_target_revision_1m  NUMERIC,
                price_target_revision_3m  NUMERIC,
                eps_revision_momentum     NUMERIC,
                analyst_rating_normalized NUMERIC,
                analyst_coverage_quality  NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                   AS isin,
       CASE
           WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" + "# No Opinion Ratings" +
                 "# Sell Ratings" + "# Strong Sell Ratings") > 0
               THEN ("# Strong Buys Ratings" + "# Buys Ratings") /
                    NULLIF("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                           "# Sell Ratings" + "# Strong Sell Ratings", 0) * 100
           END                                                                  AS analyst_bullish_pct,
       CASE
           WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" + "# No Opinion Ratings" +
                 "# Sell Ratings" + "# Strong Sell Ratings") > 0
               THEN ("# Sell Ratings" + "# Strong Sell Ratings") /
                    NULLIF("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" + "# No Opinion Ratings" +
                           "# Sell Ratings" + "# Strong Sell Ratings", 0) * 100
           END                                                                  AS analyst_bearish_pct,
       -- NEW: Neutral sentiment (Hold ratings)
       CASE
           WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" + "# No Opinion Ratings" +
                 "# Sell Ratings" + "# Strong Sell Ratings") > 0
               THEN "# Hold Ratings" /
                    NULLIF("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" + "# No Opinion Ratings" +
                           "# Sell Ratings" + "# Strong Sell Ratings", 0) * 100
           END                                                                  AS analyst_neutral_pct,
       ABS(
               CASE
                   WHEN ("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" + "# No Opinion Ratings" +
                         "# Sell Ratings" + "# Strong Sell Ratings") > 0
                       THEN (("# Strong Buys Ratings" + "# Buys Ratings") -
                             ("# Sell Ratings" + "# Strong Sell Ratings")) /
                            NULLIF("# Strong Buys Ratings" + "# Buys Ratings" + "# Hold Ratings" +
                                   "# No Opinion Ratings" +
                                   "# Sell Ratings" + "# Strong Sell Ratings", 0) * 100
                   END
       )                                                                        AS analyst_conviction,
       ("Price Target - Median" - "Last Price") / NULLIF("Last Price", 0) AS upside_potential,
       ("Price Target - High" - "Price Target - Low") / NULLIF("Price Target - Median", 0) *
       100                                                                      AS price_target_spread_pct,
       ("Price Target" - "Price Target (1M Ago)") /
       NULLIF("Price Target (1M Ago)", 0)                                       AS price_target_revision_1m,
       ("Price Target" - "Price Target (3M Ago)") /
       NULLIF("Price Target (3M Ago)", 0)                                       AS price_target_revision_3m,
       COALESCE("EPS Est Avg Rev % (FY1E - 1W)", 0) * 0.30 +
       COALESCE("EPS Est Avg Rev % (FY1E - 1M)", 0) * 0.25 +
       COALESCE("EPS Est Avg Rev % (FY1E - 3M)", 0) * 0.20 +
       COALESCE("EPS Est Avg Rev % (FY1E - 6M)", 0) * 0.15 +
       COALESCE("EPS Est Avg Rev % (FY1E - 1Y)", 0) *
       0.10                                                                     AS eps_revision_momentum,
       ("Analyst Rating" - 1) * 25                                              AS analyst_rating_normalized,
       "Price Target - #" / NULLIF(LN(1 + "Market Cap"), 0)                     AS analyst_coverage_quality
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_price_target_dynamics(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                       TEXT,
                pt_momentum_1w             NUMERIC,
                pt_momentum_1m             NUMERIC,
                pt_momentum_3m             NUMERIC,
                pt_momentum_6m             NUMERIC,
                pt_momentum_1y             NUMERIC,
                pt_median_momentum_1m      NUMERIC,
                pt_median_momentum_3m      NUMERIC,
                pt_acceleration_short      NUMERIC,
                pt_acceleration_long       NUMERIC,
                pt_consensus_convergence   NUMERIC,
                analyst_coverage_change_1m INTEGER,
                analyst_coverage_change_3m INTEGER,
                analyst_coverage_change_1y INTEGER,
                pt_vs_price_momentum       NUMERIC,
                analyst_coverage_trend     NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                            AS isin,
       ("Price Target" - "Price Target (1W Ago)") / NULLIF("Price Target (1W Ago)", 0)   AS pt_momentum_1w,
       ("Price Target" - "Price Target (1M Ago)") / NULLIF("Price Target (1M Ago)", 0)   AS pt_momentum_1m,
       ("Price Target" - "Price Target (3M Ago)") / NULLIF("Price Target (3M Ago)", 0)   AS pt_momentum_3m,
       ("Price Target" - "Price Target (6M Ago)") / NULLIF("Price Target (6M Ago)", 0)   AS pt_momentum_6m,
       ("Price Target" - "Price Target (1Y Ago)") / NULLIF("Price Target (1Y Ago)", 0)   AS pt_momentum_1y,
       ("Price Target - Median" - "Price Target - Median (1M Ago)") /
       NULLIF("Price Target - Median (1M Ago)", 0)                                       AS pt_median_momentum_1m,
       ("Price Target - Median" - "Price Target - Median (3M Ago)") /
       NULLIF("Price Target - Median (3M Ago)", 0)                                       AS pt_median_momentum_3m,
       (("Price Target" - "Price Target (1M Ago)") / NULLIF("Price Target (1M Ago)", 0)) -
       (("Price Target" - "Price Target (3M Ago)") / NULLIF("Price Target (3M Ago)", 0)) AS pt_acceleration_short,
       (("Price Target" - "Price Target (3M Ago)") / NULLIF("Price Target (3M Ago)", 0)) -
       (("Price Target" - "Price Target (1Y Ago)") / NULLIF("Price Target (1Y Ago)", 0)) AS pt_acceleration_long,
       (("Price Target - High (3M Ago)" - "Price Target - Low (3M Ago)") /
        NULLIF("Price Target - Median (3M Ago)", 0)) -
       (("Price Target - High" - "Price Target - Low") /
        NULLIF("Price Target - Median", 0))                                              AS pt_consensus_convergence,
       ("Price Target - #" - "Price Target - # (1M Ago)")::INTEGER                       AS analyst_coverage_change_1m,
       ("Price Target - #" - "Price Target - # (3M Ago)")::INTEGER                       AS analyst_coverage_change_3m,
       ("Price Target - #" - "Price Target - # (1Y Ago)")::INTEGER                       AS analyst_coverage_change_1y,
       (("Price Target" / NULLIF("Last Price", 0)) -
        ("Price Target (3M Ago)" / NULLIF("Price (3M Ago)", 0))) /
       NULLIF(("Price Target (3M Ago)" / NULLIF("Price (3M Ago)", 0)), 0)                AS pt_vs_price_momentum,
       (COALESCE("Price Target - #" - "Price Target - # (1M Ago)", 0) * 0.40 +
        COALESCE("Price Target - #" - "Price Target - # (3M Ago)", 0) * 0.35 +
        COALESCE("Price Target - #" - "Price Target - # (6M Ago)", 0) * 0.25) /
       NULLIF("Price Target - #"::NUMERIC, 0)                                            AS analyst_coverage_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 7: EARNINGS FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_earnings_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                eps_surprise_pct        NUMERIC,
                revenue_surprise_pct    NUMERIC,
                eps_adjustment_ratio    NUMERIC,
                gaap_adj_eps_gap_pct    NUMERIC,
                ebitda_adjustment_ratio NUMERIC,
                eps_quarterly_trend     NUMERIC,
                eps_yoy_growth          NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                AS isin,
       CASE
           WHEN ABS("EPS Norm - Est Avg (FY1E)") > 0
               THEN ("EPS/Adj. (LTM)" - "EPS Norm - Est Avg (FY1E)") /
                    NULLIF(ABS("EPS Norm - Est Avg (FY1E)"), 0) * 100
           END                                               AS eps_surprise_pct,
       CASE
           WHEN ABS("Revenues - Est Avg (FY1E)") > 0
               THEN ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") /
                    NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100
           END                                               AS revenue_surprise_pct,
       "EPS/Adj. (LTM)" / NULLIF("Net EPS - Basic (LTM)", 0) AS eps_adjustment_ratio,
       CASE
           WHEN ABS("EPS Norm - Est Avg (FY1E)") > 0
               THEN ("EPS GAAP - Est Avg (FY1E)" - "EPS Norm - Est Avg (FY1E)") /
                    NULLIF(ABS("EPS Norm - Est Avg (FY1E)"), 0) * 100
           END                                               AS gaap_adj_eps_gap_pct,
       "EBITDA/Adj. (LTM)" / NULLIF("EBITDA (LTM)", 0)       AS ebitda_adjustment_ratio,
       CASE
           WHEN ABS("Net EPS - Basic (-4FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-4FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-4FQFQ)"), 0)
           END                                               AS eps_quarterly_trend,
       CASE
           WHEN ABS("Net EPS - Basic (-1FY)") > 0
               THEN ("Net EPS - Basic (FY)" - "Net EPS - Basic (-1FY)") /
                    NULLIF(ABS("Net EPS - Basic (-1FY)"), 0) * 100
           END                                               AS eps_yoy_growth
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_eps_trajectory_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                eps_qoq_growth        NUMERIC,
                eps_yoy_quarterly     NUMERIC,
                eps_positive_streak   INTEGER,
                eps_cagr_3y           NUMERIC,
                eps_cagr_5y           NUMERIC,
                eps_growth_accel      NUMERIC,
                eps_vs_5y_avg         NUMERIC,
                eps_improvement_count INTEGER,
                eps_trajectory_score  NUMERIC,
                eps_stability         NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                AS isin,
       CASE
           WHEN ABS("Net EPS - Basic (-1FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-1FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-1FQFQ)"), 0) * 100
           END                                                               AS eps_qoq_growth,
       CASE
           WHEN ABS("Net EPS - Basic (-4FQFQ)") > 0
               THEN ("Net EPS - Basic (FQ)" - "Net EPS - Basic (-4FQFQ)") /
                    NULLIF(ABS("Net EPS - Basic (-4FQFQ)"), 0) * 100
           END                                                               AS eps_yoy_quarterly,
       (CASE WHEN "Net EPS - Basic (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER AS eps_positive_streak,
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                               AS eps_cagr_3y,
       CASE
           WHEN "Net EPS - Basic (-5FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1) * 100
           END                                                               AS eps_cagr_5y,
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (-5FY)" > 0
               AND "Net EPS - Basic (FY)" > 0
               THEN ((POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) -
                     (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-5FY)", 0), 1.0 / 5.0) - 1)) * 100
           END                                                               AS eps_growth_accel,
       CASE
           WHEN ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                     "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0) > 0
               THEN ("Net EPS - Basic (FY)" -
                     (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                       "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0)) /
                    NULLIF(ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                                "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0), 0) * 100
           END                                                               AS eps_vs_5y_avg,
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END)::INTEGER
                                                                             AS eps_improvement_count,
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 * 100                                                     AS eps_trajectory_score,
       CASE
           WHEN ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" + "Net EPS - Basic (-2FY)" +
                     "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0) > 0
               THEN 1.0 - LEAST(1.0,
                                SQRT(
                                        (POWER("Net EPS - Basic (FY)" -
                                               (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" +
                                                 "Net EPS - Basic (-2FY)" + "Net EPS - Basic (-3FY)" +
                                                 "Net EPS - Basic (-4FY)") / 5.0), 2) +
                                         POWER("Net EPS - Basic (-1FY)" -
                                               (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" +
                                                 "Net EPS - Basic (-2FY)" + "Net EPS - Basic (-3FY)" +
                                                 "Net EPS - Basic (-4FY)") / 5.0), 2) +
                                         POWER("Net EPS - Basic (-2FY)" -
                                               (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" +
                                                 "Net EPS - Basic (-2FY)" + "Net EPS - Basic (-3FY)" +
                                                 "Net EPS - Basic (-4FY)") / 5.0), 2) +
                                         POWER("Net EPS - Basic (-3FY)" -
                                               (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" +
                                                 "Net EPS - Basic (-2FY)" + "Net EPS - Basic (-3FY)" +
                                                 "Net EPS - Basic (-4FY)") / 5.0), 2) +
                                         POWER("Net EPS - Basic (-4FY)" -
                                               (("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" +
                                                 "Net EPS - Basic (-2FY)" + "Net EPS - Basic (-3FY)" +
                                                 "Net EPS - Basic (-4FY)") / 5.0), 2)
                                            ) / 5.0
                                ) / NULLIF(ABS(("Net EPS - Basic (FY)" + "Net EPS - Basic (-1FY)" +
                                                "Net EPS - Basic (-2FY)" +
                                                "Net EPS - Basic (-3FY)" + "Net EPS - Basic (-4FY)") / 5.0), 0)
                          )
           END                                                               AS eps_stability -- 0 = chaotic, 1 = perfectly stable
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_gaap_adjusted_analytics(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                                TEXT,
                -- EPS Adjustments
                eps_adjustment_spread_ltm           NUMERIC,
                eps_adjustment_spread_fy            NUMERIC,
                eps_adjustment_spread_1fy           NUMERIC,
                eps_adjustment_spread_fq            NUMERIC,
                eps_adjustment_spread_1fqfq         NUMERIC,
                eps_adjustment_spread_2fqfq         NUMERIC,
                eps_adjustment_spread_3fqfq         NUMERIC,
                eps_adjustment_spread_4fqfq         NUMERIC,
                eps_adjustment_spread_2fy           NUMERIC,
                eps_adjustment_spread_3fy           NUMERIC,
                eps_adjustment_spread_4fy           NUMERIC,
                eps_adjustment_pct                  NUMERIC,
                -- Net Income Adjustments
                net_income_adjustment_ratio_ltm     NUMERIC,
                net_income_adjustment_ratio_fy      NUMERIC,
                net_income_adjustment_ratio_1fy     NUMERIC,
                net_income_adjustment_ratio_fq      NUMERIC,
                net_income_adjustment_ratio_5yavgfq NUMERIC,
                net_income_adjustment_ratio_1fqfq   NUMERIC,
                net_income_adjustment_ratio_2fqfq   NUMERIC,
                net_income_adjustment_ratio_3fqfq   NUMERIC,
                net_income_adjustment_ratio_4fqfq   NUMERIC,
                net_income_adjustment_ratio_2fy     NUMERIC,
                net_income_adjustment_ratio_3fy     NUMERIC,
                net_income_adjustment_ratio_4fy     NUMERIC,
                net_income_adjustment_pct           NUMERIC,
                -- EBITDA Adjustments
                ebitda_adjustment_pct_ltm           NUMERIC,
                ebitda_adjustment_pct_fy            NUMERIC,
                ebitda_adjustment_pct_1fy           NUMERIC,
                ebitda_adjustment_pct_fq            NUMERIC,
                ebitda_adjustment_pct_1fqfq         NUMERIC,
                ebitda_adjustment_pct_2fqfq         NUMERIC,
                ebitda_adjustment_pct_3fqfq         NUMERIC,
                ebitda_adjustment_pct_4fqfq         NUMERIC,
                ebitda_adjustment_pct_2fy           NUMERIC,
                ebitda_adjustment_pct_3fy           NUMERIC,
                ebitda_adjustment_pct_4fy           NUMERIC,
                -- EBIT Adjustments
                ebit_adjustment_pct_ltm             NUMERIC,
                ebit_adjustment_pct_fy              NUMERIC,
                ebit_adjustment_pct_1fy             NUMERIC,
                ebit_adjustment_pct_fq              NUMERIC,
                ebit_adjustment_pct_1fqfq           NUMERIC,
                ebit_adjustment_pct_2fqfq           NUMERIC,
                ebit_adjustment_pct_3fqfq           NUMERIC,
                ebit_adjustment_pct_4fqfq           NUMERIC,
                ebit_adjustment_pct_2fy             NUMERIC,
                ebit_adjustment_pct_3fy             NUMERIC,
                ebit_adjustment_pct_4fy             NUMERIC,
                -- Quality Scores
                earnings_quality_score              NUMERIC,
                earnings_quality_warning            INTEGER,
                forward_eps_gaap_adj_spread         NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                       AS isin,
       -- EPS Adjustment Spreads (EPS/Adj. - Net EPS - Basic)
       "EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)"                                   AS eps_adjustment_spread_ltm,
       "EPS/Adj. (FY)" - "Net EPS - Basic (FY)"                                     AS eps_adjustment_spread_fy,
       "EPS/Adj. (-1FY)" - "Net EPS - Basic (-1FY)"                                 AS eps_adjustment_spread_1fy,
       "EPS/Adj. (FQ)" - "Net EPS - Basic (FQ)"                                     AS eps_adjustment_spread_fq,
       "EPS/Adj. (-1FQFQ)" - "Net EPS - Basic (-1FQFQ)"                             AS eps_adjustment_spread_1fqfq,
       "EPS/Adj. (-2FQFQ)" - "Net EPS - Basic (-2FQFQ)"                             AS eps_adjustment_spread_2fqfq,
       "EPS/Adj. (-3FQFQ)" - "Net EPS - Basic (-3FQFQ)"                             AS eps_adjustment_spread_3fqfq,
       "EPS/Adj. (-4FQFQ)" - "Net EPS - Basic (-4FQFQ)"                             AS eps_adjustment_spread_4fqfq,
       "EPS/Adj. (-2FY)" - "Net EPS - Basic (-2FY)"                                 AS eps_adjustment_spread_2fy,
       "EPS/Adj. (-3FY)" - "Net EPS - Basic (-3FY)"                                 AS eps_adjustment_spread_3fy,
       "EPS/Adj. (-4FY)" - "Net EPS - Basic (-4FY)"                                 AS eps_adjustment_spread_4fy,
       ("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
       NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100                                AS eps_adjustment_pct,

       -- Net Income Adjustment Ratios (Net Income/Adj. / Net Income - (IS))
       "Net Income/Adj. (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)               AS net_income_adjustment_ratio_ltm,
       "Net Income/Adj. (FY)" / NULLIF("Net Income - (IS) (FY)", 0)                 AS net_income_adjustment_ratio_fy,
       "Net Income/Adj. (-1FY)" / NULLIF("Net Income - (IS) (-1FY)", 0)             AS net_income_adjustment_ratio_1fy,
       "Net Income/Adj. (FQ)" / NULLIF("Net Income - (IS) (FQ)", 0)                 AS net_income_adjustment_ratio_fq,
       "Net Income/Adj. (5YAVGFQ)" / NULLIF("Net Income - (IS) (5YAVGFQ)", 0)       AS net_income_adjustment_ratio_5yavgfq,
       "Net Income/Adj. (-1FQFQ)" / NULLIF("Net Income - (IS) (-1FQFQ)", 0)         AS net_income_adjustment_ratio_1fqfq,
       "Net Income/Adj. (-2FQFQ)" / NULLIF("Net Income - (IS) (-2FQFQ)", 0)         AS net_income_adjustment_ratio_2fqfq,
       "Net Income/Adj. (-3FQFQ)" / NULLIF("Net Income - (IS) (-3FQFQ)", 0)         AS net_income_adjustment_ratio_3fqfq,
       "Net Income/Adj. (-4FQFQ)" / NULLIF("Net Income - (IS) (-4FQFQ)", 0)         AS net_income_adjustment_ratio_4fqfq,
       "Net Income/Adj. (-2FY)" / NULLIF("Net Income - (IS) (-2FY)", 0)             AS net_income_adjustment_ratio_2fy,
       "Net Income/Adj. (-3FY)" / NULLIF("Net Income - (IS) (-3FY)", 0)             AS net_income_adjustment_ratio_3fy,
       "Net Income/Adj. (-4FY)" / NULLIF("Net Income - (IS) (-4FY)", 0)             AS net_income_adjustment_ratio_4fy,
       ("Net Income/Adj. (LTM)" - "Net Income - (IS) (LTM)") /
       NULLIF(ABS("Net Income - (IS) (LTM)"), 0) *
       100                                                                          AS net_income_adjustment_pct,

       -- EBITDA Adjustment Percentages (EBITDA/Adj. - EBITDA) / |EBITDA| * 100
       ("EBITDA/Adj. (LTM)" - "EBITDA (LTM)") / NULLIF(ABS("EBITDA (LTM)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_ltm,
       ("EBITDA/Adj. (FY)" - "EBITDA (FY)") / NULLIF(ABS("EBITDA (FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_fy,
       ("EBITDA/Adj. (-1FY)" - "EBITDA (-1FY)") / NULLIF(ABS("EBITDA (-1FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_1fy,
       ("EBITDA/Adj. (FQ)" - "EBITDA (FQ)") / NULLIF(ABS("EBITDA (FQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_fq,
       ("EBITDA/Adj. (-1FQFQ)" - "EBITDA (-1FQFQ)") / NULLIF(ABS("EBITDA (-1FQFQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_1fqfq,
       ("EBITDA/Adj. (-2FQFQ)" - "EBITDA (-2FQFQ)") / NULLIF(ABS("EBITDA (-2FQFQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_2fqfq,
       ("EBITDA/Adj. (-3FQFQ)" - "EBITDA (-3FQFQ)") / NULLIF(ABS("EBITDA (-3FQFQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_3fqfq,
       ("EBITDA/Adj. (-4FQFQ)" - "EBITDA (-4FQFQ)") / NULLIF(ABS("EBITDA (-4FQFQ)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_4fqfq,
       ("EBITDA/Adj. (-2FY)" - "EBITDA (-2FY)") / NULLIF(ABS("EBITDA (-2FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_2fy,
       ("EBITDA/Adj. (-3FY)" - "EBITDA (-3FY)") / NULLIF(ABS("EBITDA (-3FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_3fy,
       ("EBITDA/Adj. (-4FY)" - "EBITDA (-4FY)") / NULLIF(ABS("EBITDA (-4FY)"), 0) *
       100                                                                          AS ebitda_adjustment_pct_4fy,

       -- EBIT Adjustment Percentages (EBIT/Adj. - EBIT) / |EBIT| * 100
       ("EBIT/Adj. (LTM)" - "EBIT (LTM)") / NULLIF(ABS("EBIT (LTM)"), 0) *
       100                                                                          AS ebit_adjustment_pct_ltm,
       ("EBIT/Adj. (FY)" - "EBIT (FY)") / NULLIF(ABS("EBIT (FY)"), 0) * 100         AS ebit_adjustment_pct_fy,
       ("EBIT/Adj. (-1FY)" - "EBIT (-1FY)") / NULLIF(ABS("EBIT (-1FY)"), 0) *
       100                                                                          AS ebit_adjustment_pct_1fy,
       ("EBIT/Adj. (FQ)" - "EBIT (FQ)") / NULLIF(ABS("EBIT (FQ)"), 0) * 100         AS ebit_adjustment_pct_fq,
       ("EBIT/Adj. (-1FQFQ)" - "EBIT (-1FQFQ)") / NULLIF(ABS("EBIT (-1FQFQ)"), 0) *
       100                                                                          AS ebit_adjustment_pct_1fqfq,
       ("EBIT/Adj. (-2FQFQ)" - "EBIT (-2FQFQ)") / NULLIF(ABS("EBIT (-2FQFQ)"), 0) *
       100                                                                          AS ebit_adjustment_pct_2fqfq,
       ("EBIT/Adj. (-3FQFQ)" - "EBIT (-3FQFQ)") / NULLIF(ABS("EBIT (-3FQFQ)"), 0) *
       100                                                                          AS ebit_adjustment_pct_3fqfq,
       ("EBIT/Adj. (-4FQFQ)" - "EBIT (-4FQFQ)") / NULLIF(ABS("EBIT (-4FQFQ)"), 0) *
       100                                                                          AS ebit_adjustment_pct_4fqfq,
       ("EBIT/Adj. (-2FY)" - "EBIT (-2FY)") / NULLIF(ABS("EBIT (-2FY)"), 0) *
       100                                                                          AS ebit_adjustment_pct_2fy,
       ("EBIT/Adj. (-3FY)" - "EBIT (-3FY)") / NULLIF(ABS("EBIT (-3FY)"), 0) *
       100                                                                          AS ebit_adjustment_pct_3fy,
       ("EBIT/Adj. (-4FY)" - "EBIT (-4FY)") / NULLIF(ABS("EBIT (-4FY)"), 0) *
       100                                                                          AS ebit_adjustment_pct_4fy,

       -- Quality Scores (based on LTM EPS adjustment)
       GREATEST(0, LEAST(100,
                         100 - ABS(("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
                                   NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100))) AS earnings_quality_score,
       CASE
           WHEN ABS(("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)") /
                    NULLIF(ABS("Net EPS - Basic (LTM)"), 0) * 100) > 15
               THEN 1
           ELSE 0
           END                                                                      AS earnings_quality_warning,
       "EPS Norm - Est Avg (FY1E)" - "EPS GAAP - Est Avg (FY1E)"                    AS forward_eps_gaap_adj_spread
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_gaap_revision_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                         TEXT,
                gaap_revision_momentum       NUMERIC,
                gaap_revision_1m             NUMERIC,
                gaap_revision_3m             NUMERIC,
                gaap_revision_6m             NUMERIC,
                gaap_revision_1y             NUMERIC,
                gaap_vs_norm_revision_spread NUMERIC,
                gaap_revision_acceleration   NUMERIC,
                gaap_positive_revision_flag  INTEGER,
                revision_quality_divergence  NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                      AS isin,
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1M)", 0) * 0.35 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 3M)", 0) * 0.30 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 6M)", 0) * 0.20 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1Y)", 0) * 0.15                    AS gaap_revision_momentum,
       "EPS GAAP Est Avg Rev % (FY1E - 1M)"                                        AS gaap_revision_1m,
       "EPS GAAP Est Avg Rev % (FY1E - 3M)"                                        AS gaap_revision_3m,
       "EPS GAAP Est Avg Rev % (FY1E - 6M)"                                        AS gaap_revision_6m,
       "EPS GAAP Est Avg Rev % (FY1E - 1Y)"                                        AS gaap_revision_1y,
       "EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)"      AS gaap_vs_norm_revision_spread,
       "EPS GAAP Est Avg Rev % (FY1E - 1M)" - "EPS GAAP Est Avg Rev % (FY1E - 6M)" AS gaap_revision_acceleration,
       CASE
           WHEN "EPS GAAP Est Avg Rev % (FY1E - 1M)" > 0
               AND "EPS GAAP Est Avg Rev % (FY1E - 3M)" > 0
               AND "EPS GAAP Est Avg Rev % (FY1E - 6M)" > 0
               THEN 1
           ELSE 0
           END                                                                     AS gaap_positive_revision_flag,
       ABS(("EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)") -
           ("EPS Est Avg Rev % (FY1E - 1M)" - "EPS GAAP Est Avg Rev % (FY1E - 1M)"))
                                                                                   AS revision_quality_divergence
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 8: GROWTH FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_growth_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                revenue_growth_yoy      NUMERIC,
                ebitda_growth_yoy       NUMERIC,
                operating_income_growth NUMERIC,
                fcf_growth              NUMERIC,
                revenue_cagr_5y         NUMERIC,
                forward_revenue_growth  NUMERIC,
                revenue_vs_5y_avg       NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                          AS isin,
       CASE
           WHEN ABS("Total Revenues (-1FY)") > 0
               THEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                    NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100
           END                                                         AS revenue_growth_yoy,
       CASE
           WHEN ABS("EBITDA (-1FY)") > 0
               THEN ("EBITDA (FY)" - "EBITDA (-1FY)") / NULLIF(ABS("EBITDA (-1FY)"), 0) * 100
           END                                                         AS ebitda_growth_yoy,
       CASE
           WHEN ABS("Operating Income (FY)") > 0
               THEN ("Operating Income (LTM)" - "Operating Income (FY)") /
                    NULLIF(ABS("Operating Income (FY)"), 0) * 100
           END                                                         AS operating_income_growth,
       CASE
           WHEN ABS("FCF (FY)") > 0
               THEN ("FCF (LTM)" - "FCF (FY)") / NULLIF(ABS("FCF (FY)"), 0) * 100
           END                                                         AS fcf_growth,
       "Total Revenues/CAGR (5Y FY)"                                   AS revenue_cagr_5y,
       "Revenues - Est YoY % (FY1E)"                                   AS forward_revenue_growth,
       "Total Revenues (LTM)" / NULLIF("Total Revenues (5YAVGLTM)", 0) AS revenue_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_revenue_forecast_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                       TEXT,
                revenue_est_spread         NUMERIC,
                revenue_beat_potential     NUMERIC,
                revenue_est_revision_trend NUMERIC,
                ebitda_est_vs_actual       NUMERIC,
                forward_revenue_multiple   NUMERIC,
                revenue_estimate_count     NUMERIC,
                revenue_guidance_gap       NUMERIC,
                consensus_revenue_growth   NUMERIC,
                ebit_estimate_spread       NUMERIC,
                forward_ebitda_margin      NUMERIC,
                revenue_acceleration       NUMERIC,
                estimate_confidence_score  NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                   AS isin,
       ("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
       NULLIF("Revenues - Est Med (FY1E)", 0) * 100                             AS revenue_est_spread,
       ("Total Revenues (LTM)" - "Revenues - Est Avg (FY1E)") /
       NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100                        AS revenue_beat_potential,
       "Revenues - Est YoY % (FY1E)"                                            AS revenue_est_revision_trend,
       ("EBITDA (LTM)" - "EBITDA - Est Avg (FY1E)") /
       NULLIF(ABS("EBITDA - Est Avg (FY1E)"), 0) * 100                          AS ebitda_est_vs_actual,
       "Enterprise Value" / NULLIF("Revenues - Est Avg (FY1E)", 0)              AS forward_revenue_multiple,
       "EPS Norm - Est # (FY1E)"                                                AS revenue_estimate_count,
       ("Revenues - Est Avg (NTM)" - "Revenues - Est Avg (FY1E)") /
       NULLIF(ABS("Revenues - Est Avg (FY1E)"), 0) * 100                        AS revenue_guidance_gap,
       ("Revenues - Est Avg (FY1E)" - "Total Revenues (FY)") /
       NULLIF(ABS("Total Revenues (FY)"), 0) * 100                              AS consensus_revenue_growth,
       ("EBIT - Est Med (FY1E)" - "EBIT - Est Med (NTM)") /
       NULLIF(ABS("EBIT - Est Med (NTM)"), 0) * 100                             AS ebit_estimate_spread,
       "EBITDA - Est Avg (FY1E)" / NULLIF("Revenues - Est Avg (FY1E)", 0) * 100 AS forward_ebitda_margin,
       "Revenues - Est YoY % (FY1E)" - "Total Revenues/CAGR (5Y FY)"            AS revenue_acceleration,
       GREATEST(0, LEAST(100,
                         100 - ABS(("Revenues - Est Avg (FY1E)" - "Revenues - Est Med (FY1E)") /
                                   NULLIF("Revenues - Est Med (FY1E)", 0) * 100)
                   ))                                                           AS estimate_confidence_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 9: DIVIDEND FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_dividend_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                        TEXT,
                dividend_streak             INTEGER,
                dividend_yield_ltm          NUMERIC,
                dividend_yield_ntm          NUMERIC,
                dividend_payout_ratio       NUMERIC,
                fcf_dividend_coverage       NUMERIC,
                buyback_yield               NUMERIC,
                total_shareholder_yield     NUMERIC,
                dividend_growth_expectation NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                  AS isin,
       "Dividend Streak"::INTEGER                                              AS dividend_streak,
       "Div Yield (LTM)"                                                       AS dividend_yield_ltm,
       "Div Yield (NTM)"                                                       AS dividend_yield_ntm,
       ABS("Common Dividends Paid (LTM)") / NULLIF("Net Income/Adj. (LTM)", 0) AS dividend_payout_ratio,
       CASE
           WHEN ABS("Common Dividends Paid (LTM)") > 0
               THEN "FCF (LTM)" / NULLIF(ABS("Common Dividends Paid (LTM)"), 0)
           END                                                                 AS fcf_dividend_coverage,
       "Buyback Yield (LTM)"                                                   AS buyback_yield,
       COALESCE("Buyback Yield (LTM)", 0) + COALESCE("Div Yield (LTM)", 0)     AS total_shareholder_yield,
       "Div Yield (NTM)" - "Div Yield (LTM)"                                   AS dividend_growth_expectation
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_dividend_timing(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                     TEXT,
                days_since_ex_date       INTEGER,
                days_to_payment          INTEGER,
                dividend_announced_flag  INTEGER,
                ex_date_approaching_flag INTEGER,
                dividend_frequency_score INTEGER,
                dividend_consistency     NUMERIC,
                recent_dividend_change   NUMERIC,
                dividend_yield_vs_5y_avg NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                     AS isin,
       (CURRENT_DATE - "Dividend Record (Ex Date)")::INTEGER      AS days_since_ex_date,
       ("Dividend Record (Payable Date)" - CURRENT_DATE)::INTEGER AS days_to_payment,
       CASE
           WHEN (CURRENT_DATE - "Dividend Record (Announce Date)") <= 30
               THEN 1
           ELSE 0
           END                                                    AS dividend_announced_flag,
       CASE
           WHEN ("Dividend Record (Ex Date)" - CURRENT_DATE) BETWEEN 0 AND 14
               THEN 1
           ELSE 0
           END                                                    AS ex_date_approaching_flag,
       CASE "Dividend Record (Frequency)"
           WHEN 'Quarterly' THEN 4
           WHEN 'Semi-Annual' THEN 2
           WHEN 'Annual' THEN 1
           WHEN 'Monthly' THEN 12
           ELSE 0
           END                                                    AS dividend_frequency_score,
       LEAST(1.0, "Dividend Streak"::NUMERIC / 10.0)              AS dividend_consistency,
       CASE
           WHEN "Div Yield (-1FYInd)" > 0
               THEN ("Div Yield (Ind)" - "Div Yield (-1FYInd)") /
                    NULLIF("Div Yield (-1FYInd)", 0) * 100
           END                                                    AS recent_dividend_change,
       "Div Yield (LTM)" / NULLIF("Div Yield (5YAVGLTM)", 0)      AS dividend_yield_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 10: EMPLOYMENT FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_employment_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                revenue_per_employee NUMERIC,
                profit_per_employee  NUMERIC,
                ebitda_per_employee  NUMERIC,
                assets_per_employee  NUMERIC,
                fte_growth_1y_pct    NUMERIC,
                fte_growth_3y_pct    NUMERIC,
                workforce_stability  NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"  AS isin,
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Total Revenues (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END AS revenue_per_employee,
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Normalized Net Income (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END AS profit_per_employee,
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "EBITDA (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END AS ebitda_per_employee,
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Total Assets (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END AS assets_per_employee,
       CASE
           WHEN "Full Time Employees (-1FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                    NULLIF("Full Time Employees (-1FY)", 0) * 100
           END AS fte_growth_1y_pct,
       CASE
           WHEN "Full Time Employees (-3FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-3FY)") /
                    NULLIF("Full Time Employees (-3FY)", 0) * 100
           END AS fte_growth_3y_pct,
       CASE
           WHEN "Avg Employees (5YAVGFY)" > 0
               THEN "Full Time Employees (FY)" / NULLIF("Avg Employees (5YAVGFY)", 0)
           END AS workforce_stability
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_employment_dynamics(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                fte_growth_2y_pct         NUMERIC,
                fte_acceleration          NUMERIC,
                workforce_volatility      NUMERIC,
                hiring_intensity          NUMERIC,
                productivity_trend        NUMERIC,
                headcount_vs_revenue      NUMERIC,
                workforce_efficiency_gain NUMERIC,
                layoff_risk_flag          INTEGER,
                rapid_hiring_flag         INTEGER,
                sustainable_growth_flag   INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                             AS isin,
       CASE
           WHEN "Full Time Employees (-2FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-2FY)") /
                    NULLIF("Full Time Employees (-2FY)", 0) * 100
           END                                            AS fte_growth_2y_pct,
       CASE
           WHEN "Full Time Employees (-1FY)" > 0 AND "Full Time Employees (-3FY)" > 0
               THEN (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) -
                    (POWER("Full Time Employees (FY)" / NULLIF("Full Time Employees (-3FY)", 0), 1.0 / 3.0) - 1)
           END * 100                                      AS fte_acceleration,
       ABS(("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
           NULLIF("Full Time Employees (-1FY)", 0) -
           ("Full Time Employees (-1FY)" - "Full Time Employees (-2FY)") /
           NULLIF("Full Time Employees (-2FY)", 0)) * 100 AS workforce_volatility,
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) > 0
               THEN (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) /
                    NULLIF((("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                            NULLIF(ABS("Total Revenues (-1FY)"), 0)), 0)
           END                                            AS hiring_intensity,
       CASE
           WHEN "Full Time Employees (FY)" > 0 AND "Full Time Employees (-1FY)" > 0
               THEN (("Total Revenues (FY)" / "Full Time Employees (FY)") -
                     ("Total Revenues (-1FY)" / "Full Time Employees (-1FY)")) /
                    NULLIF(ABS("Total Revenues (-1FY)" / "Full Time Employees (-1FY)"), 0) * 100
           END                                            AS productivity_trend,
       (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
        NULLIF("Full Time Employees (-1FY)", 0) * 100) -
       (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
        NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100)    AS headcount_vs_revenue,
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) >
                ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0)
               THEN (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                     NULLIF(ABS("Total Revenues (-1FY)"), 0) -
                     ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) * 100
           ELSE 0
           END                                            AS workforce_efficiency_gain,
       CASE
           WHEN "Full Time Employees (FY)" < "Full Time Employees (-1FY)"
               AND "Total Revenues (FY)" < "Total Revenues (-1FY)"
               THEN 1
           ELSE 0
           END                                            AS layoff_risk_flag,
       CASE
           WHEN ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0) > 0.20
               THEN 1
           ELSE 0
           END                                            AS rapid_hiring_flag,
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) >
                ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0)
               AND ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") > 0
               THEN 1
           ELSE 0
           END                                            AS sustainable_growth_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 11: CASH FLOW FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_cashflow_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                cfo_to_net_income     NUMERIC,
                fcf_to_net_income     NUMERIC,
                fcf_margin            NUMERIC,
                cfo_growth_yoy        NUMERIC,
                fcf_positive_ratio    NUMERIC,
                acquisition_intensity NUMERIC,
                self_funding_ratio    NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                 AS isin,
       "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)     AS cfo_to_net_income,
       "FCF (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)     AS fcf_to_net_income,
       "FCF (LTM)" / NULLIF("Total Revenues (LTM)", 0)        AS fcf_margin,
       ("CFO (LTM)" - "CFO (-1FY)") / NULLIF("CFO (-1FY)", 0) AS cfo_growth_yoy,
       (CASE WHEN "FCF (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FQFQ)" > 0 THEN 1 ELSE 0 END) / 5.0 AS fcf_positive_ratio,
       ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
       ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))         AS acquisition_intensity,
       CASE
           WHEN ABS("CFI (LTM)") > 0
               THEN "CFO (LTM)" / NULLIF(ABS("CFI (LTM)"), 0)
           END                                                AS self_funding_ratio
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_enhanced_cashflow_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                -- Existing features
                fcf_positive_years      INTEGER,
                fcf_always_positive     INTEGER,
                capex_vs_5y_avg         NUMERIC,
                underinvestment_flag    INTEGER,
                cfo_share_of_cf         NUMERIC,
                cfi_share_of_cf         NUMERIC,
                cff_share_of_cf         NUMERIC,
                self_funding_flag       INTEGER,
                acquisition_to_fcf      NUMERIC,
                sustainable_ma_flag     INTEGER,
                fcf_4q_improvement      NUMERIC,
                cash_flow_quality_score NUMERIC,
                -- NEW: CapEx temporal analysis
                capex_yoy_growth        NUMERIC,
                capex_qoq_growth        NUMERIC,
                capex_3y_trend          NUMERIC,
                capex_volatility        NUMERIC,
                capex_acceleration      INTEGER,
                capex_cut_flag          INTEGER,
                overinvestment_flag     INTEGER,
                -- NEW: Cash Acquisitions temporal analysis
                acquisitions_yoy_growth NUMERIC,
                acquisitions_vs_5y_avg  NUMERIC,
                acquisitions_ltm_total  NUMERIC,
                ma_intensity_score      NUMERIC,
                serial_acquirer_flag    INTEGER,
                acquisition_pause_flag  INTEGER,
                -- NEW: Combined investment metrics
                total_investment_to_cfo NUMERIC,
                organic_vs_inorganic    NUMERIC,
                investment_efficiency   NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                            AS isin,
       -- Existing features (unchanged)
       (CASE WHEN "FCF (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER           AS fcf_positive_years,
       CASE
           WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0
               AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0
               THEN 1
           ELSE 0
           END                                                           AS fcf_always_positive,
       ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0)
                                                                         AS capex_vs_5y_avg,
       CASE
           WHEN ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0) < 0.7
               THEN 1
           ELSE 0
           END                                                           AS underinvestment_flag,
       ABS("CFO (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cfo_share_of_cf,
       ABS("CFI (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cfi_share_of_cf,
       ABS("CFF (LTM)") /
       NULLIF(ABS("CFO (LTM)") + ABS("CFI (LTM)") + ABS("CFF (LTM)"), 0) AS cff_share_of_cf,
       CASE
           WHEN "CFO (LTM)" / NULLIF(ABS("CFI (LTM)"), 0) > 1
               THEN 1
           ELSE 0
           END                                                           AS self_funding_flag,
       (ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
        ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))) /
       NULLIF(ABS("FCF (LTM)"), 0)                                       AS acquisition_to_fcf,
       CASE
           WHEN (ABS(COALESCE("Cash Acquisitions (FQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-1FQFQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-2FQFQ)", 0)) +
                 ABS(COALESCE("Cash Acquisitions (-3FQFQ)", 0))) /
                NULLIF(ABS("FCF (LTM)"), 0) < 0.5
               THEN 1
           ELSE 0
           END                                                           AS sustainable_ma_flag,
       ("FCF (FQ)" - "FCF (-4FQFQ)") / NULLIF(ABS("FCF (-4FQFQ)"), 0)    AS fcf_4q_improvement,
       (CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
        CASE
            WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0
                AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0 THEN 25
            ELSE 0 END +
        CASE WHEN "CFO (LTM)" > ABS("CFI (LTM)") THEN 25 ELSE 0 END +
        CASE WHEN "FCF (LTM)" > 0 THEN 25 ELSE 0 END)::NUMERIC           AS cash_flow_quality_score,

       -- NEW: CapEx YoY growth (FY vs -1FY)
       (ABS("Capital Expenditure (FY)") - ABS("Capital Expenditure (-1FY)")) /
       NULLIF(ABS("Capital Expenditure (-1FY)"), 0) * 100                AS capex_yoy_growth,

       -- NEW: CapEx QoQ growth (FQ vs -1FQFQ)
       (ABS("Capital Expenditure (FQ)") - ABS("Capital Expenditure (-1FQFQ)")) /
       NULLIF(ABS("Capital Expenditure (-1FQFQ)"), 0) * 100              AS capex_qoq_growth,

       -- NEW: CapEx 3-year trend (FY vs -3FY)
       (ABS("Capital Expenditure (FY)") - ABS("Capital Expenditure (-3FY)")) /
       NULLIF(ABS("Capital Expenditure (-3FY)"), 0) * 100                AS capex_3y_trend,

       -- NEW: CapEx volatility (variation across quarters)
       (ABS(ABS("Capital Expenditure (FQ)") - ABS("Capital Expenditure (-1FQFQ)")) +
        ABS(ABS("Capital Expenditure (-1FQFQ)") - ABS("Capital Expenditure (-2FQFQ)")) +
        ABS(ABS("Capital Expenditure (-2FQFQ)") - ABS("Capital Expenditure (-3FQFQ)")) +
        ABS(ABS("Capital Expenditure (-3FQFQ)") - ABS("Capital Expenditure (-4FQFQ)"))) /
       NULLIF((ABS("Capital Expenditure (FQ)") + ABS("Capital Expenditure (-1FQFQ)") +
               ABS("Capital Expenditure (-2FQFQ)") + ABS("Capital Expenditure (-3FQFQ)") +
               ABS("Capital Expenditure (-4FQFQ)")) / 5.0, 0)            AS capex_volatility,

       -- NEW: CapEx acceleration flag (increasing investment rate)
       CASE
           WHEN ABS("Capital Expenditure (FY)") > ABS("Capital Expenditure (-1FY)")
               AND ABS("Capital Expenditure (-1FY)") > ABS("Capital Expenditure (-2FY)")
               THEN 1
           ELSE 0
           END                                                           AS capex_acceleration,

       -- NEW: CapEx cut flag (significant decline may signal distress or maturity)
       CASE
           WHEN (ABS("Capital Expenditure (FY)") - ABS("Capital Expenditure (-1FY)")) /
                NULLIF(ABS("Capital Expenditure (-1FY)"), 0) < -0.25
               THEN 1
           ELSE 0
           END                                                           AS capex_cut_flag,

       -- NEW: Overinvestment flag (CapEx significantly above historical average)
       CASE
           WHEN ABS("Capital Expenditure (FQ)") / NULLIF(ABS("Capital Expenditure (5YAVGFQ)"), 0) > 1.5
               THEN 1
           ELSE 0
           END                                                           AS overinvestment_flag,

       -- NEW: Cash Acquisitions YoY growth
       (ABS(COALESCE("Cash Acquisitions (FY)", 0)) - ABS(COALESCE("Cash Acquisitions (-1FY)", 0))) /
       NULLIF(ABS(COALESCE("Cash Acquisitions (-1FY)", 0)), 0) * 100     AS acquisitions_yoy_growth,

       -- NEW: Cash Acquisitions vs 5Y average
       ABS(COALESCE("Cash Acquisitions (FQ)", 0)) /
       NULLIF(ABS(COALESCE("Cash Acquisitions (5YAVGFQ)", 0)), 0)        AS acquisitions_vs_5y_avg,

       -- NEW: LTM total acquisitions
       ABS(COALESCE("Cash Acquisitions (LTM)", 0))                       AS acquisitions_ltm_total,

       -- NEW: M&A intensity score (acquisitions relative to market cap proxy via total assets)
       ABS(COALESCE("Cash Acquisitions (LTM)", 0)) /
       NULLIF("Total Assets (LTM)", 0) * 100                             AS ma_intensity_score,

       -- NEW: Serial acquirer flag (significant acquisitions in 3+ of last 4 years)
       CASE
           WHEN (CASE WHEN ABS(COALESCE("Cash Acquisitions (FY)", 0)) > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS(COALESCE("Cash Acquisitions (-1FY)", 0)) > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS(COALESCE("Cash Acquisitions (-2FY)", 0)) > 0 THEN 1 ELSE 0 END +
                 CASE WHEN ABS(COALESCE("Cash Acquisitions (-3FY)", 0)) > 0 THEN 1 ELSE 0 END) >= 3
               THEN 1
           ELSE 0
           END                                                           AS serial_acquirer_flag,

       -- NEW: Acquisition pause flag (no recent acquisitions after historical activity)
       CASE
           WHEN ABS(COALESCE("Cash Acquisitions (FY)", 0)) = 0
               AND (ABS(COALESCE("Cash Acquisitions (-1FY)", 0)) > 0
                   OR ABS(COALESCE("Cash Acquisitions (-2FY)", 0)) > 0)
               THEN 1
           ELSE 0
           END                                                           AS acquisition_pause_flag,

       -- NEW: Total investment (CapEx + Acquisitions) to CFO ratio
       (ABS(COALESCE("Capital Expenditure (LTM)", 0)) + ABS(COALESCE("Cash Acquisitions (LTM)", 0))) /
       NULLIF(ABS("CFO (LTM)"), 0)                                       AS total_investment_to_cfo,

       -- NEW: Organic vs Inorganic growth ratio (CapEx / Acquisitions)
       ABS(COALESCE("Capital Expenditure (LTM)", 0)) /
       NULLIF(ABS(COALESCE("Cash Acquisitions (LTM)", 0)), 0)            AS organic_vs_inorganic,

       -- NEW: Investment efficiency (revenue growth per unit of total investment)
       CASE
           WHEN (ABS(COALESCE("Capital Expenditure (-1FY)", 0)) + ABS(COALESCE("Cash Acquisitions (-1FY)", 0))) > 0
               THEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                    NULLIF(ABS(COALESCE("Capital Expenditure (-1FY)", 0)) +
                           ABS(COALESCE("Cash Acquisitions (-1FY)", 0)), 0)
           END                                                           AS investment_efficiency

FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_cashflow_temporal_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                cfo_quarterly_trend   NUMERIC,
                cfo_yoy_quarterly     NUMERIC,
                cfi_quarterly_trend   NUMERIC,
                cff_quarterly_trend   NUMERIC,
                fcf_quarterly_trend   NUMERIC,
                cfo_positive_quarters INTEGER,
                cfi_negative_quarters INTEGER,
                cff_pattern_score     NUMERIC,
                cash_burn_rate        NUMERIC,
                cf_volatility_score   NUMERIC,
                operating_cf_momentum NUMERIC,
                financing_dependency  NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                               AS isin,
       ("CFO (FQ)" - "CFO (-4FQFQ)") / NULLIF(ABS("CFO (-4FQFQ)"), 0) * 100 AS cfo_quarterly_trend,
       CASE
           WHEN ABS("CFO (-4FQFQ)") > 0
               THEN ("CFO (FQ)" - "CFO (-4FQFQ)") / NULLIF(ABS("CFO (-4FQFQ)"), 0) * 100
           END                                                              AS cfo_yoy_quarterly,
       ("CFI (FQ)" - "CFI (-4FQFQ)") / NULLIF(ABS("CFI (-4FQFQ)"), 0) * 100 AS cfi_quarterly_trend,
       ("CFF (FQ)" - "CFF (-4FQFQ)") / NULLIF(ABS("CFF (-4FQFQ)"), 0) * 100 AS cff_quarterly_trend,
       ("FCF (FQ)" - "FCF (-4FQFQ)") / NULLIF(ABS("FCF (-4FQFQ)"), 0) * 100 AS fcf_quarterly_trend,
       (CASE WHEN "CFO (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER            AS cfo_positive_quarters,
       (CASE WHEN "CFI (FQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-1FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-2FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-3FQFQ)" < 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFI (-4FQFQ)" < 0 THEN 1 ELSE 0 END)::INTEGER            AS cfi_negative_quarters,
       CASE
           WHEN ("CFF (FQ)" + "CFF (-1FQFQ)" + "CFF (-2FQFQ)" + "CFF (-3FQFQ)") > 0
               THEN -1
           WHEN ("CFF (FQ)" + "CFF (-1FQFQ)" + "CFF (-2FQFQ)" + "CFF (-3FQFQ)") < 0
               THEN 1
           ELSE 0
           END::NUMERIC                                                     AS cff_pattern_score,
       CASE
           WHEN "FCF (LTM)" < 0
               THEN ABS("FCF (LTM)") / NULLIF("Cash And Equivalents (FQ)", 0) / 12.0
           ELSE 0
           END                                                              AS cash_burn_rate,
       (ABS("CFO (FQ)" - "CFO (-1FQFQ)") + ABS("CFO (-1FQFQ)" - "CFO (-2FQFQ)") +
        ABS("CFO (-2FQFQ)" - "CFO (-3FQFQ)") + ABS("CFO (-3FQFQ)" - "CFO (-4FQFQ)")) /
       NULLIF(ABS("CFO (FQ)" + "CFO (-1FQFQ)" + "CFO (-2FQFQ)" +
                  "CFO (-3FQFQ)" + "CFO (-4FQFQ)") / 5.0, 0)                AS cf_volatility_score,
       (("CFO (FQ)" + "CFO (-1FQFQ)") - ("CFO (-3FQFQ)" + "CFO (-4FQFQ)")) /
       NULLIF(ABS("CFO (-3FQFQ)" + "CFO (-4FQFQ)"), 0) * 100                AS operating_cf_momentum,
       ABS("CFF (LTM)") / NULLIF(ABS("CFO (LTM)"), 0)                       AS financing_dependency
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- FCF Growth Estimates (NEW)
-- Estimated free cash flow growth rates from consensus FCF forecasts
-- Source columns: FCF - Est Avg (FY1E/FY2E/FY3E/FY4E/FY5E), FCF (LTM/FY),
--                 Total Revenues (LTM), Market Cap, Capital Expenditure (LTM)
-- =============================================================================
CREATE OR REPLACE FUNCTION calc_fcf_growth_estimates(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                        TEXT,
                -- Forward FCF estimates (raw pass-through for downstream use)
                fcf_est_fy1                 NUMERIC,
                fcf_est_fy2                 NUMERIC,
                fcf_est_fy3                 NUMERIC,
                fcf_est_fy4                 NUMERIC,
                fcf_est_fy5                 NUMERIC,

                -- YoY estimated growth rates
                fcf_est_growth_fy1_vs_ltm   NUMERIC, -- FY1E vs current LTM
                fcf_est_growth_fy2_vs_fy1   NUMERIC, -- FY2E vs FY1E
                fcf_est_growth_fy3_vs_fy2   NUMERIC, -- FY3E vs FY2E
                fcf_est_growth_fy4_vs_fy3   NUMERIC, -- FY4E vs FY3E
                fcf_est_growth_fy5_vs_fy4   NUMERIC, -- FY5E vs FY4E

                -- Multi-year estimated CAGRs
                fcf_est_cagr_3y             NUMERIC, -- (FY3E / LTM)^(1/3) - 1
                fcf_est_cagr_5y             NUMERIC, -- (FY5E / LTM)^(1/5) - 1

                -- Forward FCF margin estimates
                fcf_est_margin_fy1          NUMERIC, -- FY1E FCF / LTM Revenue
                fcf_est_yield_fy1           NUMERIC, -- FY1E FCF / Market Cap

                -- Growth acceleration / deceleration
                fcf_est_growth_acceleration NUMERIC, -- FY2-FY1 growth minus FY1-LTM growth
                fcf_est_growth_deceleration INTEGER, -- 1 if growth is slowing across estimates

                -- Estimate spread (dispersion across forward years)
                fcf_est_trajectory_score    NUMERIC, -- Pct of forward years with positive FCF
                fcf_est_always_positive     INTEGER, -- All 5 forward estimates positive

                -- Conversion quality: estimated vs historical
                fcf_est_vs_historical       NUMERIC, -- FY1E growth vs last actual YoY growth
                fcf_est_capex_implied_ratio NUMERIC  -- FY1E FCF / (LTM CFO - LTM CapEx proxy)
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                      AS isin,

       -- Raw forward estimates
       "FCF - Est Avg (FY1E)"                                                      AS fcf_est_fy1,
       "FCF - Est Avg (FY2E)"                                                      AS fcf_est_fy2,
       "FCF - Est Avg (FY3E)"                                                      AS fcf_est_fy3,
       "FCF - Est Avg (FY4E)"                                                      AS fcf_est_fy4,
       "FCF - Est Avg (FY5E)"                                                      AS fcf_est_fy5,

       -- YoY estimated growth rates (as percentages)
       ("FCF - Est Avg (FY1E)" - "FCF (LTM)") /
       NULLIF(ABS("FCF (LTM)"), 0) * 100                                           AS fcf_est_growth_fy1_vs_ltm,

       ("FCF - Est Avg (FY2E)" - "FCF - Est Avg (FY1E)") /
       NULLIF(ABS("FCF - Est Avg (FY1E)"), 0) * 100                                AS fcf_est_growth_fy2_vs_fy1,

       ("FCF - Est Avg (FY3E)" - "FCF - Est Avg (FY2E)") /
       NULLIF(ABS("FCF - Est Avg (FY2E)"), 0) * 100                                AS fcf_est_growth_fy3_vs_fy2,

       ("FCF - Est Avg (FY4E)" - "FCF - Est Avg (FY3E)") /
       NULLIF(ABS("FCF - Est Avg (FY3E)"), 0) * 100                                AS fcf_est_growth_fy4_vs_fy3,

       ("FCF - Est Avg (FY5E)" - "FCF - Est Avg (FY4E)") /
       NULLIF(ABS("FCF - Est Avg (FY4E)"), 0) * 100                                AS fcf_est_growth_fy5_vs_fy4,

       -- 3-year estimated CAGR: (FY3E / LTM)^(1/3) - 1
       CASE
           WHEN "FCF (LTM)" > 0 AND "FCF - Est Avg (FY3E)" > 0
               THEN (POWER("FCF - Est Avg (FY3E)" /
                           NULLIF("FCF (LTM)", 0), 1.0 / 3.0) - 1) * 100
           END                                                                     AS fcf_est_cagr_3y,

       -- 5-year estimated CAGR: (FY5E / LTM)^(1/5) - 1
       CASE
           WHEN "FCF (LTM)" > 0 AND "FCF - Est Avg (FY5E)" > 0
               THEN (POWER("FCF - Est Avg (FY5E)" /
                           NULLIF("FCF (LTM)", 0), 1.0 / 5.0) - 1) * 100
           END                                                                     AS fcf_est_cagr_5y,

       -- Forward FCF margin (FY1E FCF as % of current revenue)
       "FCF - Est Avg (FY1E)" /
       NULLIF("Total Revenues (LTM)", 0) * 100                                     AS fcf_est_margin_fy1,

       -- Forward FCF yield (FY1E FCF as % of market cap)
       "FCF - Est Avg (FY1E)" /
       NULLIF("Market Cap", 0) * 100                                               AS fcf_est_yield_fy1,

       -- Growth acceleration: is FY2â†’FY1 growth faster than FY1â†’LTM growth?
       (("FCF - Est Avg (FY2E)" - "FCF - Est Avg (FY1E)") /
        NULLIF(ABS("FCF - Est Avg (FY1E)"), 0) * 100) -
       (("FCF - Est Avg (FY1E)" - "FCF (LTM)") /
        NULLIF(ABS("FCF (LTM)"), 0) * 100)                                         AS fcf_est_growth_acceleration,

       -- Growth deceleration flag: each subsequent growth rate is lower
       CASE
           WHEN ("FCF - Est Avg (FY2E)" - "FCF - Est Avg (FY1E)") /
                NULLIF(ABS("FCF - Est Avg (FY1E)"), 0) <
                ("FCF - Est Avg (FY1E)" - "FCF (LTM)") /
                NULLIF(ABS("FCF (LTM)"), 0)
               AND ("FCF - Est Avg (FY3E)" - "FCF - Est Avg (FY2E)") /
                   NULLIF(ABS("FCF - Est Avg (FY2E)"), 0) <
                   ("FCF - Est Avg (FY2E)" - "FCF - Est Avg (FY1E)") /
                   NULLIF(ABS("FCF - Est Avg (FY1E)"), 0)
               THEN 1
           ELSE 0
           END                                                                     AS fcf_est_growth_deceleration,

       -- Forward trajectory score: how many of 5 forward years have positive FCF
       (CASE WHEN "FCF - Est Avg (FY1E)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF - Est Avg (FY2E)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF - Est Avg (FY3E)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF - Est Avg (FY4E)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF - Est Avg (FY5E)" > 0 THEN 1 ELSE 0 END) / 5.0 * 100        AS fcf_est_trajectory_score,

       -- All 5 forward estimates positive
       CASE
           WHEN "FCF - Est Avg (FY1E)" > 0
               AND "FCF - Est Avg (FY2E)" > 0
               AND "FCF - Est Avg (FY3E)" > 0
               AND "FCF - Est Avg (FY4E)" > 0
               AND "FCF - Est Avg (FY5E)" > 0
               THEN 1
           ELSE 0
           END                                                                     AS fcf_est_always_positive,

       -- Estimated vs historical: compare forward FY1 growth to last actual FY growth
       (("FCF - Est Avg (FY1E)" - "FCF (LTM)") /
        NULLIF(ABS("FCF (LTM)"), 0) * 100) -
       (("FCF (FY)" - "FCF (-1FY)") /
        NULLIF(ABS("FCF (-1FY)"), 0) * 100)                                        AS fcf_est_vs_historical,

       -- Implied CapEx conversion: FY1E FCF relative to current LTM operating CF
       "FCF - Est Avg (FY1E)" /
       NULLIF(ABS("CFO (LTM)") - ABS(COALESCE("Capital Expenditure (LTM)", 0)), 0) AS fcf_est_capex_implied_ratio

FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

COMMENT ON FUNCTION calc_fcf_growth_estimates(TEXT) IS
    'Estimated free cash flow growth rates from consensus FCF forecasts (FY1E-FY5E).
     Calculates YoY growth rates, 3Y/5Y CAGRs, growth acceleration, forward margins/yields,
     and trajectory quality scores. Source: FCF - Est Avg (FY1E through FY5E).';

-- =============================================================================
-- SECTION 12: TEMPORAL FEATURES (OPTIMIZED)
-- =============================================================================

CREATE OR REPLACE FUNCTION calc_temporal_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                fiscal_quarter          INTEGER,
                fiscal_month            INTEGER,
                fiscal_year             INTEGER,
                days_to_earnings        INTEGER,
                earnings_report_recency INTEGER,
                reporting_lag           NUMERIC,
                fiscal_year_progress    NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                          AS isin,
       "Fiscal Quarter"                                AS fiscal_quarter,
       "Fiscal Month"                                  AS fiscal_month,
       "Fiscal Year"                                   AS fiscal_year,
       ("Next Earnings" - CURRENT_DATE)                AS days_to_earnings,
       (CURRENT_DATE - "Income Statement Report Date") AS earnings_report_recency,
       "Reporting Lag"                                 AS reporting_lag,
       "Fiscal Month" / 12.0                           AS fiscal_year_progress
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_fiscal_calendar_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                days_since_last_report    INTEGER,
                days_to_fy_end            INTEGER,
                is_quarter_end_month      INTEGER,
                is_fy_end_month           INTEGER,
                earnings_season_flag      INTEGER,
                pre_earnings_window       INTEGER,
                post_earnings_window      INTEGER,
                reporting_freshness_score NUMERIC,
                fiscal_quarter_progress   NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                   AS isin,
       (CURRENT_DATE - "Income Statement Report Date")::INTEGER AS days_since_last_report,
       ("FY End Date" - CURRENT_DATE)::INTEGER                  AS days_to_fy_end,
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (3, 6, 9, 12)
               THEN 1
           ELSE 0
           END                                                  AS is_quarter_end_month,
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) = EXTRACT(MONTH FROM "FY End Date")
               THEN 1
           ELSE 0
           END                                                  AS is_fy_end_month,
       CASE
           WHEN EXTRACT(MONTH FROM CURRENT_DATE) IN (1, 2, 4, 5, 7, 8, 10, 11)
               THEN 1
           ELSE 0
           END                                                  AS earnings_season_flag,
       CASE
           WHEN ("Next Earnings" - CURRENT_DATE) BETWEEN 0 AND 14
               THEN 1
           ELSE 0
           END                                                  AS pre_earnings_window,
       CASE
           WHEN (CURRENT_DATE - "Income Statement Report Date") BETWEEN 0 AND 7
               THEN 1
           ELSE 0
           END                                                  AS post_earnings_window,
       GREATEST(0, LEAST(100,
                         100 - ((CURRENT_DATE - "Income Statement Report Date")::NUMERIC / 90.0 * 100)
                   ))                                           AS reporting_freshness_score,
       CASE
           WHEN "Fiscal Month" IS NOT NULL
               THEN (("Fiscal Month" - 1) % 3 + 1) / 3.0
           END                                                  AS fiscal_quarter_progress
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 13: COMPOSITE SCORES (OPTIMIZED)
-- =============================================================================

-- Drop composite wrapper FIRST (it depends on the sub-functions)
DROP FUNCTION IF EXISTS calc_composite_scores(TEXT) CASCADE;

-- Drop and recreate atomic functions to ensure clean state
DROP FUNCTION IF EXISTS calc_piotroski_f_score(TEXT);
DROP FUNCTION IF EXISTS calc_shareholder_dilution_features(TEXT);
DROP FUNCTION IF EXISTS calc_quality_momentum_composite(TEXT);

-- Decomposed from calc_composite_scores: Piotroski F-Score (standalone)
CREATE OR REPLACE FUNCTION calc_piotroski_f_score(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin              TEXT,
                piotroski_f_score INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"         AS isin,
       (CASE WHEN "Return on Assets (ROA) % (LTM)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (LTM)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Return on Assets (ROA) % (LTM)" > "Return on Assets (ROA) % (FY)" THEN 1 ELSE 0 END +
        CASE WHEN "CFO (LTM)" > "Net Income - (IS) (LTM)" THEN 1 ELSE 0 END +
        CASE
            WHEN "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0) <
                 "Total Debt (FY)" / NULLIF("Total Equity (FY)", 0) THEN 1
            ELSE 0 END +
        CASE WHEN "Current Ratio (LTM)" > "Current Ratio (FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Shrs Out" <= "Shrs Out (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Asset Turnover (LTM)" > "Asset Turnover (FY)" THEN 1 ELSE 0 END
           )::INTEGER AS piotroski_f_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- Decomposed from calc_composite_scores: Shareholder Dilution (standalone)
CREATE OR REPLACE FUNCTION calc_shareholder_dilution_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin           TEXT,
                dilution_score NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"         AS isin,
       GREATEST(0, LEAST(100,
                         50 - (("Shrs Out" - "Shrs Out (-1FY)") / NULLIF("Shrs Out (-1FY)", 0)) * 100
                   )) AS dilution_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- Decomposed from calc_composite_scores: Quality Momentum Composite (standalone)
CREATE OR REPLACE FUNCTION calc_quality_momentum_composite(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                   TEXT,
                quality_momentum_score NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN" AS isin,
       (((CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
          CASE WHEN "Return On Equity % (LTM)" > 15 THEN 25 ELSE 0 END +
          CASE WHEN "Total Debt (LTM)" / NULLIF("Total Equity (LTM)", 0) < 1 THEN 25 ELSE 0 END +
          CASE WHEN "Current Ratio (LTM)" > 1.5 THEN 25 ELSE 0 END) * 0.40) +
        (LEAST(100, GREATEST(0,
                             (("Last Price" - "Price (3M Ago)") / NULLIF("Price (3M Ago)", 0) * 100 + 50))) * 0.30) +
        (CASE
             WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100 > 20 THEN 100
             WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100 > 10 THEN 75
             WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                  NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100 > 0 THEN 50
             ELSE 25
             END * 0.30)
           )  AS quality_momentum_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- Retained for backward compatibility, delegates to atomic functions
-- Note: eps_trajectory_score removed (single source of truth: calc_eps_trajectory_features)
-- Uses plpgsql to avoid SQL-inlining resolution issues with sub-function calls
CREATE OR REPLACE FUNCTION calc_composite_scores(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                   TEXT,
                piotroski_f_score      INTEGER,
                dilution_score         NUMERIC,
                quality_momentum_score NUMERIC
            )
    STABLE
    PARALLEL SAFE
    LANGUAGE plpgsql
AS
$$
BEGIN
    RETURN QUERY
        SELECT p.isin,
               p.piotroski_f_score,
               d.dilution_score,
               q.quality_momentum_score
        FROM public.calc_piotroski_f_score(p_isin) p
                 JOIN public.calc_shareholder_dilution_features(p_isin) d ON p.isin = d.isin
                 JOIN public.calc_quality_momentum_composite(p_isin) q ON p.isin = q.isin;
END;
$$;

-- =============================================================================
-- SECTION 14: COMPREHENSIVE FUNCTIONS (OPTIMIZED WITH MATERIALIZED VIEWS)
-- =============================================================================

-- Note: Due to size, comprehensive functions are provided as materialized views
-- for better performance. The functions remain available for ad-hoc queries.

CREATE OR REPLACE FUNCTION calc_ebit_ebitda_comprehensive(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                -- Current values (existing)
                ebit_fq               NUMERIC,
                ebit_ltm              NUMERIC,
                ebit_fy               NUMERIC,
                ebit_1fy              NUMERIC,
                ebitda_fq             NUMERIC,
                ebitda_ltm            NUMERIC,
                ebitda_fy             NUMERIC,
                ebitda_1fy            NUMERIC,
                -- NEW: Extended historical FY
                ebit_2fy              NUMERIC,
                ebit_3fy              NUMERIC,
                ebit_4fy              NUMERIC,
                ebitda_2fy            NUMERIC,
                ebitda_3fy            NUMERIC,
                ebitda_4fy            NUMERIC,
                -- NEW: Quarterly historical
                ebit_1fqfq            NUMERIC,
                ebit_2fqfq            NUMERIC,
                ebit_3fqfq            NUMERIC,
                ebit_4fqfq            NUMERIC,
                ebitda_1fqfq          NUMERIC,
                ebitda_2fqfq          NUMERIC,
                ebitda_3fqfq          NUMERIC,
                ebitda_4fqfq          NUMERIC,
                -- NEW: 5-year averages
                ebit_5yavgfq          NUMERIC,
                ebit_5yavgltm         NUMERIC,
                ebitda_5yavgfq        NUMERIC,
                ebitda_5yavgltm       NUMERIC,
                -- NEW: Adjusted variants
                ebit_adj_fq           NUMERIC,
                ebit_adj_ltm          NUMERIC,
                ebit_adj_fy           NUMERIC,
                ebitda_adj_fq         NUMERIC,
                ebitda_adj_ltm        NUMERIC,
                ebitda_adj_fy         NUMERIC,
                -- Derived metrics (existing + enhanced)
                ebit_growth_yoy       NUMERIC,
                ebitda_growth_yoy     NUMERIC,
                ebit_margin_ltm       NUMERIC,
                ebitda_margin_ltm     NUMERIC,
                ebit_positive_years   INTEGER,
                ebitda_positive_years INTEGER,
                -- NEW: Quarterly momentum
                ebit_qoq_growth       NUMERIC,
                ebitda_qoq_growth     NUMERIC,
                -- NEW: Multi-year CAGR
                ebit_cagr_3y          NUMERIC,
                ebitda_cagr_3y        NUMERIC,
                -- NEW: vs 5Y average
                ebit_vs_5y_avg        NUMERIC,
                ebitda_vs_5y_avg      NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                    AS isin,
       -- Current values
       "EBIT (FQ)"                                                               AS ebit_fq,
       "EBIT (LTM)"                                                              AS ebit_ltm,
       "EBIT (FY)"                                                               AS ebit_fy,
       "EBIT (-1FY)"                                                             AS ebit_1fy,
       "EBITDA (FQ)"                                                             AS ebitda_fq,
       "EBITDA (LTM)"                                                            AS ebitda_ltm,
       "EBITDA (FY)"                                                             AS ebitda_fy,
       "EBITDA (-1FY)"                                                           AS ebitda_1fy,
       -- Extended historical FY
       "EBIT (-2FY)"                                                             AS ebit_2fy,
       "EBIT (-3FY)"                                                             AS ebit_3fy,
       "EBIT (-4FY)"                                                             AS ebit_4fy,
       "EBITDA (-2FY)"                                                           AS ebitda_2fy,
       "EBITDA (-3FY)"                                                           AS ebitda_3fy,
       "EBITDA (-4FY)"                                                           AS ebitda_4fy,
       -- Quarterly historical
       "EBIT (-1FQFQ)"                                                           AS ebit_1fqfq,
       "EBIT (-2FQFQ)"                                                           AS ebit_2fqfq,
       "EBIT (-3FQFQ)"                                                           AS ebit_3fqfq,
       "EBIT (-4FQFQ)"                                                           AS ebit_4fqfq,
       "EBITDA (-1FQFQ)"                                                         AS ebitda_1fqfq,
       "EBITDA (-2FQFQ)"                                                         AS ebitda_2fqfq,
       "EBITDA (-3FQFQ)"                                                         AS ebitda_3fqfq,
       "EBITDA (-4FQFQ)"                                                         AS ebitda_4fqfq,
       -- 5-year averages
       "EBIT (5YAVGFQ)"                                                          AS ebit_5yavgfq,
       "EBIT (5YAVGLTM)"                                                         AS ebit_5yavgltm,
       "EBITDA (5YAVGFQ)"                                                        AS ebitda_5yavgfq,
       "EBITDA (5YAVGLTM)"                                                       AS ebitda_5yavgltm,
       -- Adjusted variants
       "EBIT/Adj. (FQ)"                                                          AS ebit_adj_fq,
       "EBIT/Adj. (LTM)"                                                         AS ebit_adj_ltm,
       "EBIT/Adj. (FY)"                                                          AS ebit_adj_fy,
       "EBITDA/Adj. (FQ)"                                                        AS ebitda_adj_fq,
       "EBITDA/Adj. (LTM)"                                                       AS ebitda_adj_ltm,
       "EBITDA/Adj. (FY)"                                                        AS ebitda_adj_fy,
       -- Derived metrics
       ("EBIT (FY)" - "EBIT (-1FY)") / NULLIF(ABS("EBIT (-1FY)"), 0) * 100       AS ebit_growth_yoy,
       ("EBITDA (FY)" - "EBITDA (-1FY)") / NULLIF(ABS("EBITDA (-1FY)"), 0) * 100 AS ebitda_growth_yoy,
       "EBIT (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                    AS ebit_margin_ltm,
       "EBITDA (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100                  AS ebitda_margin_ltm,
       (CASE WHEN "EBIT (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBIT (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER                  AS ebit_positive_years,
       (CASE WHEN "EBITDA (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "EBITDA (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER                AS ebitda_positive_years,
       -- NEW: Quarterly momentum
       ("EBIT (FQ)" - "EBIT (-1FQFQ)") / NULLIF(ABS("EBIT (-1FQFQ)"), 0) * 100   AS ebit_qoq_growth,
       ("EBITDA (FQ)" - "EBITDA (-1FQFQ)") / NULLIF(ABS("EBITDA (-1FQFQ)"), 0) * 100
                                                                                 AS ebitda_qoq_growth,
       -- NEW: Multi-year CAGR (3-year)
       CASE
           WHEN "EBIT (-3FY)" > 0 AND "EBIT (FY)" > 0
               THEN (POWER("EBIT (FY)" / NULLIF("EBIT (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                                   AS ebit_cagr_3y,
       CASE
           WHEN "EBITDA (-3FY)" > 0 AND "EBITDA (FY)" > 0
               THEN (POWER("EBITDA (FY)" / NULLIF("EBITDA (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                                   AS ebitda_cagr_3y,
       -- NEW: vs 5Y average
       "EBIT (LTM)" / NULLIF("EBIT (5YAVGLTM)", 0)                               AS ebit_vs_5y_avg,
       "EBITDA (LTM)" / NULLIF("EBITDA (5YAVGLTM)", 0)                           AS ebitda_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_net_income_comprehensive(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                       TEXT,
                -- Base values (existing)
                net_income_is_fq           NUMERIC,
                net_income_is_ltm          NUMERIC,
                net_income_is_fy           NUMERIC,
                net_income_adj_ltm         NUMERIC,
                normalized_ni_ltm          NUMERIC,
                -- NEW: Extended quarterly historical
                net_income_is_1fqfq        NUMERIC,
                net_income_is_2fqfq        NUMERIC,
                net_income_is_3fqfq        NUMERIC,
                net_income_is_4fqfq        NUMERIC,
                -- NEW: Extended yearly historical
                net_income_is_1fy          NUMERIC,
                net_income_is_2fy          NUMERIC,
                net_income_is_3fy          NUMERIC,
                net_income_is_4fy          NUMERIC,
                -- NEW: 5-year averages
                net_income_is_5yavgfq      NUMERIC,
                net_income_is_5yavgltm     NUMERIC,
                normalized_ni_5yavgfq      NUMERIC,
                normalized_ni_5yavgltm     NUMERIC,
                -- Derived metrics (existing + enhanced)
                net_income_growth_yoy      NUMERIC,
                net_income_margin_ltm      NUMERIC,
                ni_adjustment_ratio        NUMERIC,
                net_income_positive_years  INTEGER,
                earnings_quality_composite NUMERIC,
                -- NEW: Quarterly trends
                net_income_qoq_growth      NUMERIC,
                net_income_yoy_quarterly   NUMERIC,
                -- NEW: vs 5Y averages
                net_income_vs_5y_avg       NUMERIC,
                normalized_ni_vs_5y_avg    NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                      AS isin,
       -- Base values
       "Net Income - (IS) (FQ)"                                    AS net_income_is_fq,
       "Net Income - (IS) (LTM)"                                   AS net_income_is_ltm,
       "Net Income - (IS) (FY)"                                    AS net_income_is_fy,
       "Net Income/Adj. (LTM)"                                     AS net_income_adj_ltm,
       "Normalized Net Income (LTM)"                               AS normalized_ni_ltm,
       -- Extended quarterly historical
       "Net Income - (IS) (-1FQFQ)"                                AS net_income_is_1fqfq,
       "Net Income - (IS) (-2FQFQ)"                                AS net_income_is_2fqfq,
       "Net Income - (IS) (-3FQFQ)"                                AS net_income_is_3fqfq,
       "Net Income - (IS) (-4FQFQ)"                                AS net_income_is_4fqfq,
       -- Extended yearly historical
       "Net Income - (IS) (-1FY)"                                  AS net_income_is_1fy,
       "Net Income - (IS) (-2FY)"                                  AS net_income_is_2fy,
       "Net Income - (IS) (-3FY)"                                  AS net_income_is_3fy,
       "Net Income - (IS) (-4FY)"                                  AS net_income_is_4fy,
       -- 5-year averages
       "Net Income - (IS) (5YAVGFQ)"                               AS net_income_is_5yavgfq,
       "Net Income - (IS) (5YAVGLTM)"                              AS net_income_is_5yavgltm,
       "Normalized Net Income (5YAVGFQ)"                           AS normalized_ni_5yavgfq,
       "Normalized Net Income (5YAVGLTM)"                          AS normalized_ni_5yavgltm,
       -- Derived metrics
       public.pct_change("Net Income - (IS) (FY)"::NUMERIC,
                         "Net Income - (IS) (-1FY)"::NUMERIC)      AS net_income_growth_yoy,
       "Net Income Margin % (LTM)"::NUMERIC                        AS net_income_margin_ltm,
       public.safe_divide("Net Income/Adj. (LTM)"::NUMERIC,
                          "Net Income - (IS) (LTM)"::NUMERIC)      AS ni_adjustment_ratio,
       (CASE WHEN "Net Income - (IS) (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net Income - (IS) (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN "Net Income - (IS) (-4FY)" > 0 THEN 1
            ELSE 0 END)::INTEGER                                   AS net_income_positive_years,
       public.clamp_score(
               50 +
               (CASE WHEN "Net Income - (IS) (FY)" > 0 THEN 10 ELSE -10 END) +
               (CASE WHEN "Net Income - (IS) (-1FY)" > 0 THEN 5 ELSE -5 END) +
               (CASE WHEN "Net Income - (IS) (-2FY)" > 0 THEN 5 ELSE -5 END) +
               (CASE
                    WHEN ABS(public.safe_divide(("Net Income/Adj. (LTM)"::NUMERIC - "Net Income - (IS) (LTM)"::NUMERIC),
                                                "Net Income - (IS) (LTM)"::NUMERIC)) < 0.10 THEN 15
                    ELSE -15 END) +
               (CASE WHEN "Net Income - (IS) (FY)" > "Net Income - (IS) (-1FY)" THEN 10 ELSE -5 END) +
               (CASE WHEN "Net Income - (IS) (-1FY)" > "Net Income - (IS) (-2FY)" THEN 5 ELSE -5 END)
       )                                                           AS earnings_quality_composite,
       -- Quarterly trends
       public.pct_change("Net Income - (IS) (FQ)"::NUMERIC,
                         "Net Income - (IS) (-1FQFQ)"::NUMERIC)    AS net_income_qoq_growth,
       public.pct_change("Net Income - (IS) (FQ)"::NUMERIC,
                         "Net Income - (IS) (-4FQFQ)"::NUMERIC)    AS net_income_yoy_quarterly,
       -- vs 5Y averages
       public.safe_divide("Net Income - (IS) (LTM)"::NUMERIC,
                          "Net Income - (IS) (5YAVGLTM)"::NUMERIC) AS net_income_vs_5y_avg,
       public.safe_divide("Normalized Net Income (LTM)"::NUMERIC, "Normalized Net Income (5YAVGLTM)"::NUMERIC)
                                                                   AS normalized_ni_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_total_revenues_temporal(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                -- Base values
                revenue_fq            NUMERIC,
                revenue_ltm           NUMERIC,
                revenue_fy            NUMERIC,
                revenue_1fy           NUMERIC,
                -- 5-year averages
                revenue_5yavgfq       NUMERIC,
                revenue_5yavgltm      NUMERIC,
                -- Growth metrics
                revenue_growth_yoy    NUMERIC,
                revenue_vs_5y_avg_fq  NUMERIC,
                revenue_vs_5y_avg_ltm NUMERIC,
                -- Trend indicators
                revenue_fq_vs_avg     NUMERIC,
                revenue_momentum      NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                   AS isin,
       "Total Revenues (FQ)"                                    AS revenue_fq,
       "Total Revenues (LTM)"                                   AS revenue_ltm,
       "Total Revenues (FY)"                                    AS revenue_fy,
       "Total Revenues (-1FY)"                                  AS revenue_1fy,
       "Total Revenues (5YAVGFQ)"                               AS revenue_5yavgfq,
       "Total Revenues (5YAVGLTM)"                              AS revenue_5yavgltm,
       public.pct_change("Total Revenues (FY)"::NUMERIC,
                         "Total Revenues (-1FY)"::NUMERIC)      AS revenue_growth_yoy,
       public.safe_divide("Total Revenues (FQ)"::NUMERIC,
                          "Total Revenues (5YAVGFQ)"::NUMERIC)  AS revenue_vs_5y_avg_fq,
       public.safe_divide("Total Revenues (LTM)"::NUMERIC,
                          "Total Revenues (5YAVGLTM)"::NUMERIC) AS revenue_vs_5y_avg_ltm,
       public.safe_divide(("Total Revenues (FQ)"::NUMERIC - "Total Revenues (5YAVGFQ)"::NUMERIC),
                          "Total Revenues (5YAVGFQ)"::NUMERIC) * 100
                                                                AS revenue_fq_vs_avg,
       public.calc_change_ratio("Total Revenues (LTM)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC) *
       100                                                      AS revenue_momentum
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_working_capital_temporal(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                -- Current values
                wc_fq                NUMERIC,
                wc_fy                NUMERIC,
                wc_ltm               NUMERIC,
                wc_5yavgfy           NUMERIC,
                -- Quarterly historical (FQ style)
                wc_1fq               NUMERIC,
                wc_2fq               NUMERIC,
                wc_3fq               NUMERIC,
                wc_4fq               NUMERIC,
                -- Yearly historical
                wc_1fy               NUMERIC,
                wc_2fy               NUMERIC,
                wc_3fy               NUMERIC,
                wc_4fy               NUMERIC,
                -- Trend metrics
                wc_qoq_change        NUMERIC,
                wc_yoy_change        NUMERIC,
                wc_4q_trend          NUMERIC,
                wc_vs_5y_avg         NUMERIC,
                wc_positive_quarters INTEGER,
                wc_improving_flag    INTEGER,
                wc_volatility        NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                                    AS isin,
       -- Current values
       "Working Capital (FQ)"                                                                    AS wc_fq,
       "Working Capital (FY)"                                                                    AS wc_fy,
       "Working Capital (LTM)"                                                                   AS wc_ltm,
       "Working Capital (5YAVGFY)"                                                               AS wc_5yavgfy,
       -- Quarterly historical
       "Working Capital (-1FQ)"                                                                  AS wc_1fq,
       "Working Capital (-2FQ)"                                                                  AS wc_2fq,
       "Working Capital (-3FQ)"                                                                  AS wc_3fq,
       "Working Capital (-4FQ)"                                                                  AS wc_4fq,
       -- Yearly historical
       "Working Capital (-1FY)"                                                                  AS wc_1fy,
       "Working Capital (-2FY)"                                                                  AS wc_2fy,
       "Working Capital (-3FY)"                                                                  AS wc_3fy,
       "Working Capital (-4FY)"                                                                  AS wc_4fy,
       -- Trend metrics
       public.pct_change("Working Capital (FQ)"::NUMERIC, "Working Capital (-1FQ)"::NUMERIC)     AS wc_qoq_change,
       public.pct_change("Working Capital (FY)"::NUMERIC, "Working Capital (-1FY)"::NUMERIC)     AS wc_yoy_change,
       public.pct_change("Working Capital (FQ)"::NUMERIC, "Working Capital (-4FQ)"::NUMERIC)     AS wc_4q_trend,
       public.safe_divide("Working Capital (FQ)"::NUMERIC, "Working Capital (5YAVGFY)"::NUMERIC) AS wc_vs_5y_avg,
       (CASE WHEN "Working Capital (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Working Capital (-1FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Working Capital (-2FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Working Capital (-3FQ)" > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN "Working Capital (-4FQ)" > 0 THEN 1
            ELSE 0 END)::INTEGER                                                                 AS wc_positive_quarters,
       CASE
           WHEN "Working Capital (FQ)" > "Working Capital (-1FQ)"
               AND "Working Capital (-1FQ)" > "Working Capital (-2FQ)"
               THEN 1
           ELSE 0 END                                                                            AS wc_improving_flag,
       -- Volatility: coefficient of variation across quarters
       (ABS("Working Capital (FQ)" - "Working Capital (-1FQ)") +
        ABS("Working Capital (-1FQ)" - "Working Capital (-2FQ)") +
        ABS("Working Capital (-2FQ)" - "Working Capital (-3FQ)") +
        ABS("Working Capital (-3FQ)" - "Working Capital (-4FQ)")) /
       NULLIF(ABS(("Working Capital (FQ)" + "Working Capital (-1FQ)" +
                   "Working Capital (-2FQ)" + "Working Capital (-3FQ)" +
                   "Working Capital (-4FQ)") / 5.0), 0)                                          AS wc_volatility
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_total_debt_temporal(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                -- Current values
                debt_fq              NUMERIC,
                debt_fy              NUMERIC,
                debt_ltm             NUMERIC,
                -- Quarterly historical
                debt_1fq             NUMERIC,
                debt_2fq             NUMERIC,
                debt_3fq             NUMERIC,
                debt_4fq             NUMERIC,
                -- Yearly historical
                debt_1fy             NUMERIC,
                debt_2fy             NUMERIC,
                debt_3fy             NUMERIC,
                debt_4fy             NUMERIC,
                -- Trend metrics
                debt_qoq_change      NUMERIC,
                debt_yoy_change      NUMERIC,
                debt_4q_trend        NUMERIC,
                debt_3y_cagr         NUMERIC,
                debt_deleveraging    INTEGER,
                debt_to_equity_trend NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                         AS isin,
       -- Current values
       "Total Debt (FQ)"                                                              AS debt_fq,
       "Total Debt (FY)"                                                              AS debt_fy,
       "Total Debt (LTM)"                                                             AS debt_ltm,
       -- Quarterly historical
       "Total Debt (-1FQ)"                                                            AS debt_1fq,
       "Total Debt (-2FQ)"                                                            AS debt_2fq,
       "Total Debt (-3FQ)"                                                            AS debt_3fq,
       "Total Debt (-4FQ)"                                                            AS debt_4fq,
       -- Yearly historical
       "Total Debt (-1FY)"                                                            AS debt_1fy,
       "Total Debt (-2FY)"                                                            AS debt_2fy,
       "Total Debt (-3FY)"                                                            AS debt_3fy,
       "Total Debt (-4FY)"                                                            AS debt_4fy,
       -- Trend metrics
       public.pct_change("Total Debt (FQ)"::NUMERIC, "Total Debt (-1FQ)"::NUMERIC)    AS debt_qoq_change,
       public.pct_change("Total Debt (FY)"::NUMERIC, "Total Debt (-1FY)"::NUMERIC)    AS debt_yoy_change,
       public.pct_change("Total Debt (FQ)"::NUMERIC, "Total Debt (-4FQ)"::NUMERIC)    AS debt_4q_trend,
       CASE
           WHEN "Total Debt (-3FY)" > 0
               THEN
               (POWER(public.safe_divide("Total Debt (FY)"::NUMERIC, "Total Debt (-3FY)"::NUMERIC), 1.0 / 3.0) - 1) *
               100
           END                                                                        AS debt_3y_cagr,
       CASE
           WHEN "Total Debt (FQ)" < "Total Debt (-1FQ)"
               AND "Total Debt (-1FQ)" < "Total Debt (-2FQ)"
               THEN 1
           ELSE 0 END                                                                 AS debt_deleveraging,
       public.safe_divide("Total Debt (FY)"::NUMERIC, "Total Equity (FY)"::NUMERIC) -
       public.safe_divide("Total Debt (-1FY)"::NUMERIC, "Total Equity (FY)"::NUMERIC) AS debt_to_equity_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_total_assets_temporal(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin               TEXT,
                -- Current values
                assets_fq          NUMERIC,
                assets_fy          NUMERIC,
                assets_ltm         NUMERIC,
                -- Quarterly historical
                assets_1fq         NUMERIC,
                assets_2fq         NUMERIC,
                assets_3fq         NUMERIC,
                assets_4fq         NUMERIC,
                -- Yearly historical
                assets_1fy         NUMERIC,
                assets_2fy         NUMERIC,
                assets_3fy         NUMERIC,
                assets_4fy         NUMERIC,
                -- Trend metrics
                assets_qoq_growth  NUMERIC,
                assets_yoy_growth  NUMERIC,
                assets_3y_cagr     NUMERIC,
                asset_growth_accel NUMERIC,
                asset_base_stable  INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                            AS isin,
       "Total Assets (FQ)"                                                               AS assets_fq,
       "Total Assets (FY)"                                                               AS assets_fy,
       "Total Assets (LTM)"                                                              AS assets_ltm,
       "Total Assets (-1FQ)"                                                             AS assets_1fq,
       "Total Assets (-2FQ)"                                                             AS assets_2fq,
       "Total Assets (-3FQ)"                                                             AS assets_3fq,
       "Total Assets (-4FQ)"                                                             AS assets_4fq,
       "Total Assets (-1FY)"                                                             AS assets_1fy,
       "Total Assets (-2FY)"                                                             AS assets_2fy,
       "Total Assets (-3FY)"                                                             AS assets_3fy,
       "Total Assets (-4FY)"                                                             AS assets_4fy,
       public.pct_change("Total Assets (FQ)"::NUMERIC, "Total Assets (-1FQ)"::NUMERIC)   AS assets_qoq_growth,
       public.pct_change("Total Assets (FY)"::NUMERIC, "Total Assets (-1FY)"::NUMERIC)   AS assets_yoy_growth,
       CASE
           WHEN "Total Assets (-3FY)" > 0
               THEN
               (POWER(public.safe_divide("Total Assets (FY)"::NUMERIC, "Total Assets (-3FY)"::NUMERIC), 1.0 / 3.0) -
                1) *
               100
           END                                                                           AS assets_3y_cagr,
       -- Growth acceleration: recent growth vs historical
       public.pct_change("Total Assets (FY)"::NUMERIC, "Total Assets (-1FY)"::NUMERIC) -
       public.pct_change("Total Assets (-1FY)"::NUMERIC, "Total Assets (-2FY)"::NUMERIC) AS asset_growth_accel,
       -- Stability flag: growing consistently
       CASE
           WHEN "Total Assets (FY)" >= "Total Assets (-1FY)"
               AND "Total Assets (-1FY)" >= "Total Assets (-2FY)"
               AND "Total Assets (-2FY)" >= "Total Assets (-3FY)"
               THEN 1
           ELSE 0 END                                                                    AS asset_base_stable
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_gross_profit_temporal(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                -- Current values
                gp_fq                NUMERIC,
                gp_fy                NUMERIC,
                gp_ltm               NUMERIC,
                -- Quarterly historical (FQFQ style)
                gp_1fqfq             NUMERIC,
                gp_2fqfq             NUMERIC,
                gp_3fqfq             NUMERIC,
                gp_4fqfq             NUMERIC,
                -- Yearly historical
                gp_1fy               NUMERIC,
                gp_2fy               NUMERIC,
                gp_3fy               NUMERIC,
                gp_4fy               NUMERIC,
                -- Derived metrics
                gp_qoq_growth        NUMERIC,
                gp_yoy_growth        NUMERIC,
                gp_margin_fq         NUMERIC,
                gp_margin_trend      NUMERIC,
                gp_positive_quarters INTEGER,
                gp_margin_expansion  INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                              AS isin,
       "Gross Profit (FQ)"                                 AS gp_fq,
       "Gross Profit (FY)"                                 AS gp_fy,
       "Gross Profit (LTM)"                                AS gp_ltm,
       "Gross Profit (-1FQFQ)"                             AS gp_1fqfq,
       "Gross Profit (-2FQFQ)"                             AS gp_2fqfq,
       "Gross Profit (-3FQFQ)"                             AS gp_3fqfq,
       "Gross Profit (-4FQFQ)"                             AS gp_4fqfq,
       "Gross Profit (-1FY)"                               AS gp_1fy,
       "Gross Profit (-2FY)"                               AS gp_2fy,
       "Gross Profit (-3FY)"                               AS gp_3fy,
       "Gross Profit (-4FY)"                               AS gp_4fy,
       public.pct_change("Gross Profit (FQ)"::NUMERIC,
                         "Gross Profit (-1FQFQ)"::NUMERIC) AS gp_qoq_growth,
       public.pct_change("Gross Profit (FY)"::NUMERIC,
                         "Gross Profit (-1FY)"::NUMERIC)   AS gp_yoy_growth,
       public.safe_divide("Gross Profit (FQ)"::NUMERIC, "Total Revenues (FQ)"::NUMERIC) *
       100                                                 AS gp_margin_fq,
       (public.safe_divide("Gross Profit (FQ)"::NUMERIC, "Total Revenues (FQ)"::NUMERIC) -
        public.safe_divide("Gross Profit (-4FQFQ)"::NUMERIC, "Total Revenues (5YAVGFQ)"::NUMERIC)) *
       100                                                 AS gp_margin_trend,
       (CASE WHEN "Gross Profit (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN "Gross Profit (-4FQFQ)" > 0 THEN 1
            ELSE 0 END)::INTEGER                           AS gp_positive_quarters,
       CASE
           WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)"
               THEN 1
           ELSE 0 END                                      AS gp_margin_expansion
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_quality_features_comprehensive(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                          TEXT,
                goodwill_impairment_ltm       NUMERIC,
                asset_writedown_ltm           NUMERIC,
                restructuring_ltm             NUMERIC,
                has_goodwill_impairment_ltm   INTEGER,
                goodwill_impairment_frequency INTEGER,
                asset_writedown_frequency     INTEGER,
                restructuring_frequency       INTEGER,
                exceptional_items_total_ltm   NUMERIC,
                exceptional_items_to_ebitda   NUMERIC,
                quality_issues_count_5y       INTEGER,
                accounting_quality_score      NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                       AS isin,
       "Impairment of Goodwill (LTM)"                                               AS goodwill_impairment_ltm,
       "Asset Writedown (LTM)"                                                      AS asset_writedown_ltm,
       "Restructuring Charges (LTM)"                                                AS restructuring_ltm,
       CASE WHEN "Impairment of Goodwill (LTM)" <> 0 THEN 1 ELSE 0 END              AS has_goodwill_impairment_ltm,
       (CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER  AS goodwill_impairment_frequency,
       (CASE WHEN "Asset Writedown (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Asset Writedown (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER         AS asset_writedown_frequency,
       (CASE WHEN "Restructuring Charges (FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-1FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-2FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-3FY)" <> 0 THEN 1 ELSE 0 END +
        CASE WHEN "Restructuring Charges (-4FY)" <> 0 THEN 1 ELSE 0 END)::INTEGER   AS restructuring_frequency,
       ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
       ABS("Restructuring Charges (LTM)")                                           AS exceptional_items_total_ltm,
       (ABS("Impairment of Goodwill (LTM)") + ABS("Asset Writedown (LTM)") +
        ABS("Restructuring Charges (LTM)")) / NULLIF(ABS("EBITDA (LTM)"), 0)        AS exceptional_items_to_ebitda,
       ((CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END) +
        (CASE WHEN "Asset Writedown (FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-1FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-2FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-3FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Asset Writedown (-4FY)" <> 0 THEN 1 ELSE 0 END) +
        (CASE WHEN "Restructuring Charges (FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-1FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-2FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-3FY)" <> 0 THEN 1 ELSE 0 END +
         CASE WHEN "Restructuring Charges (-4FY)" <> 0 THEN 1 ELSE 0 END))::INTEGER AS quality_issues_count_5y,
       GREATEST(0, LEAST(100,
                         100 -
                         ((CASE WHEN "Impairment of Goodwill (FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-1FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-2FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-3FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Impairment of Goodwill (-4FY)" <> 0 THEN 1 ELSE 0 END) * 8) -
                         ((CASE WHEN "Asset Writedown (FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-1FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-2FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-3FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Asset Writedown (-4FY)" <> 0 THEN 1 ELSE 0 END) * 4) -
                         ((CASE WHEN "Restructuring Charges (FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-1FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-2FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-3FY)" <> 0 THEN 1 ELSE 0 END +
                           CASE WHEN "Restructuring Charges (-4FY)" <> 0 THEN 1 ELSE 0 END) * 4)
                   ))                                                               AS accounting_quality_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_eps_comprehensive(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                eps_basic_fq         NUMERIC,
                eps_basic_ltm        NUMERIC,
                eps_basic_fy         NUMERIC,
                eps_adj_ltm          NUMERIC,
                eps_norm_est_fy1e    NUMERIC,
                eps_growth_yoy       NUMERIC,
                eps_cagr_3y          NUMERIC,
                eps_adjustment_ratio NUMERIC,
                eps_positive_years   INTEGER,
                eps_trajectory_score NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                              AS isin,
       "Net EPS - Basic (FQ)"                                              AS eps_basic_fq,
       "Net EPS - Basic (LTM)"                                             AS eps_basic_ltm,
       "Net EPS - Basic (FY)"                                              AS eps_basic_fy,
       "EPS/Adj. (LTM)"                                                    AS eps_adj_ltm,
       "EPS Norm - Est Avg (FY1E)"                                         AS eps_norm_est_fy1e,
       ("Net EPS - Basic (FY)" - "Net EPS - Basic (-1FY)") /
       NULLIF(ABS("Net EPS - Basic (-1FY)"), 0) * 100                      AS eps_growth_yoy,
       CASE
           WHEN "Net EPS - Basic (-3FY)" > 0 AND "Net EPS - Basic (FY)" > 0
               THEN (POWER("Net EPS - Basic (FY)" / NULLIF("Net EPS - Basic (-3FY)", 0), 1.0 / 3.0) - 1) * 100
           END                                                             AS eps_cagr_3y,
       "EPS/Adj. (LTM)" / NULLIF("Net EPS - Basic (LTM)", 0)               AS eps_adjustment_ratio,
       (CASE WHEN "Net EPS - Basic (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER AS eps_positive_years,
       (CASE WHEN "Net EPS - Basic (FY)" > "Net EPS - Basic (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-1FY)" > "Net EPS - Basic (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-2FY)" > "Net EPS - Basic (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-3FY)" > "Net EPS - Basic (-4FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Net EPS - Basic (-4FY)" > "Net EPS - Basic (-5FY)" THEN 1 ELSE 0 END
           ) / 5.0 * 100                                                   AS eps_trajectory_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- EPS Continuing Operations Features (NEW)
-- Uses Basic EPS - Cont columns for core operations earnings quality
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_eps_continuing_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                -- Current period values
                eps_cont_ltm              NUMERIC,
                eps_cont_fq               NUMERIC,
                eps_cont_fy               NUMERIC,
                -- Historical FQ
                eps_cont_1fqfq            NUMERIC,
                eps_cont_2fqfq            NUMERIC,
                eps_cont_3fqfq            NUMERIC,
                eps_cont_4fqfq            NUMERIC,
                -- Historical FY
                eps_cont_1fy              NUMERIC,
                eps_cont_2fy              NUMERIC,
                eps_cont_3fy              NUMERIC,
                eps_cont_4fy              NUMERIC,
                -- Derived analytics
                eps_cont_qoq_growth       NUMERIC,
                eps_cont_yoy_growth       NUMERIC,
                eps_cont_cagr_3y          NUMERIC,
                eps_cont_vs_total_eps     NUMERIC,
                eps_cont_positive_streak  INTEGER,
                eps_cont_trajectory_score NUMERIC,
                -- Quality flags
                discontinued_ops_impact   NUMERIC,
                core_earnings_stability   NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                                    AS isin,
       -- Current period values
       "Basic EPS - Cont (LTM)"                                                                  AS eps_cont_ltm,
       "Basic EPS - Cont (FQ)"                                                                   AS eps_cont_fq,
       "Basic EPS - Cont (FY)"                                                                   AS eps_cont_fy,
       -- Historical FQ
       "Basic EPS - Cont (-1FQFQ)"                                                               AS eps_cont_1fqfq,
       "Basic EPS - Cont (-2FQFQ)"                                                               AS eps_cont_2fqfq,
       "Basic EPS - Cont (-3FQFQ)"                                                               AS eps_cont_3fqfq,
       "Basic EPS - Cont (-4FQFQ)"                                                               AS eps_cont_4fqfq,
       -- Historical FY
       "Basic EPS - Cont (-1FY)"                                                                 AS eps_cont_1fy,
       "Basic EPS - Cont (-2FY)"                                                                 AS eps_cont_2fy,
       "Basic EPS - Cont (-3FY)"                                                                 AS eps_cont_3fy,
       "Basic EPS - Cont (-4FY)"                                                                 AS eps_cont_4fy,
       -- QoQ growth
       public.pct_change("Basic EPS - Cont (FQ)"::NUMERIC, "Basic EPS - Cont (-1FQFQ)"::NUMERIC) AS eps_cont_qoq_growth,
       -- YoY growth
       public.pct_change("Basic EPS - Cont (FY)"::NUMERIC, "Basic EPS - Cont (-1FY)"::NUMERIC)   AS eps_cont_yoy_growth,
       -- 3-year CAGR
       CASE
           WHEN "Basic EPS - Cont (-3FY)" > 0 AND "Basic EPS - Cont (FY)" > 0
               THEN
               (POWER("Basic EPS - Cont (FY)"::NUMERIC / NULLIF("Basic EPS - Cont (-3FY)"::NUMERIC, 0), 1.0 / 3.0) -
                1) * 100
           END                                                                                   AS eps_cont_cagr_3y,
       -- Continuing vs Total EPS ratio (how much comes from continuing ops)
       public.safe_divide("Basic EPS - Cont (LTM)"::NUMERIC,
                          "Net EPS - Basic (LTM)"::NUMERIC)                                      AS eps_cont_vs_total_eps,
       -- Positive streak count
       (CASE WHEN "Basic EPS - Cont (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-4FQFQ)" > 0 THEN 1 ELSE 0 END)::INTEGER
                                                                                                 AS eps_cont_positive_streak,
       -- Trajectory score (improving trend = higher score)
       (CASE WHEN "Basic EPS - Cont (FY)" > "Basic EPS - Cont (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-1FY)" > "Basic EPS - Cont (-2FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-2FY)" > "Basic EPS - Cont (-3FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Basic EPS - Cont (-3FY)" > "Basic EPS - Cont (-4FY)" THEN 1 ELSE 0 END
           ) / 4.0 *
       100                                                                                       AS eps_cont_trajectory_score,
       -- Discontinued operations impact (difference between total and continuing)
       (("Net EPS - Basic (LTM)" - "Basic EPS - Cont (LTM)") /
        NULLIF(ABS("Net EPS - Basic (LTM)"), 0)) *
       100                                                                                       AS discontinued_ops_impact,
       -- Core earnings stability score
       public.clamp_score(
               100 - ABS(public.pct_change("Basic EPS - Cont (FQ)"::NUMERIC, "Basic EPS - Cont (-4FQFQ)"::NUMERIC) -
                         public.pct_change("Basic EPS - Cont (-1FQFQ)"::NUMERIC,
                                           "Basic EPS - Cont (-4FQFQ)"::NUMERIC)) * 0.5
       )                                                                                         AS core_earnings_stability
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

CREATE OR REPLACE FUNCTION calc_cashflow_comprehensive(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                cfo_fq                  NUMERIC,
                cfo_ltm                 NUMERIC,
                cfo_fy                  NUMERIC,
                fcf_fq                  NUMERIC,
                fcf_ltm                 NUMERIC,
                fcf_fy                  NUMERIC,
                cfo_growth_yoy          NUMERIC,
                fcf_growth_yoy          NUMERIC,
                cfo_to_net_income       NUMERIC,
                fcf_margin              NUMERIC,
                fcf_yield               NUMERIC,
                cfo_positive_years      INTEGER,
                fcf_positive_years      INTEGER,
                cash_flow_quality_score NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                           AS isin,
       "CFO (FQ)"                                                       AS cfo_fq,
       "CFO (LTM)"                                                      AS cfo_ltm,
       "CFO (FY)"                                                       AS cfo_fy,
       "FCF (FQ)"                                                       AS fcf_fq,
       "FCF (LTM)"                                                      AS fcf_ltm,
       "FCF (FY)"                                                       AS fcf_fy,
       ("CFO (FY)" - "CFO (-1FY)") / NULLIF(ABS("CFO (-1FY)"), 0) * 100 AS cfo_growth_yoy,
       ("FCF (FY)" - "FCF (-1FY)") / NULLIF(ABS("FCF (-1FY)"), 0) * 100 AS fcf_growth_yoy,
       "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0)               AS cfo_to_net_income,
       "FCF (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100            AS fcf_margin,
       "FCF (LTM)" / NULLIF("Market Cap", 0) * 100                      AS fcf_yield,
       (CASE WHEN "CFO (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "CFO (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER          AS cfo_positive_years,
       (CASE WHEN "FCF (FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-1FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-2FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-3FY)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "FCF (-4FY)" > 0 THEN 1 ELSE 0 END)::INTEGER          AS fcf_positive_years,
       (CASE WHEN "CFO (LTM)" / NULLIF("Net Income - (IS) (LTM)", 0) > 1 THEN 25 ELSE 0 END +
        CASE
            WHEN "FCF (FY)" > 0 AND "FCF (-1FY)" > 0 AND "FCF (-2FY)" > 0
                AND "FCF (-3FY)" > 0 AND "FCF (-4FY)" > 0 THEN 25
            ELSE 0 END +
        CASE WHEN "CFO (LTM)" > ABS("CFI (LTM)") THEN 25 ELSE 0 END +
        CASE WHEN "FCF (LTM)" > 0 THEN 25 ELSE 0 END)::NUMERIC          AS cash_flow_quality_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION 17: MISSING FUNCTIONS (OPTIMIZED)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Beta Risk Features
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_beta_risk_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                beta_1y              NUMERIC,
                beta_5y              NUMERIC,
                beta_spread          NUMERIC,
                beta_trend           NUMERIC,
                high_beta_flag       INTEGER,
                low_beta_flag        INTEGER,
                beta_stability_score NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                        AS isin,
       "Beta (1Y)"                                   AS beta_1y,
       "Beta (5Y)"                                   AS beta_5y,
       "Beta (1Y)" - "Beta (5Y)"                     AS beta_spread,
       ("Beta (1Y)" - "Beta (5Y)") / NULLIF(ABS("Beta (5Y)"), 0) * 100
                                                     AS beta_trend,
       CASE WHEN "Beta (1Y)" > 1.5 THEN 1 ELSE 0 END AS high_beta_flag,
       CASE WHEN "Beta (1Y)" < 0.5 THEN 1 ELSE 0 END AS low_beta_flag,
       GREATEST(0, LEAST(100,
                         100 - ABS("Beta (1Y)" - "Beta (5Y)") * 50
                   ))                                AS beta_stability_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Cost Structure Features (REFACTORED)
-- Uses helper functions and corrected column references
-- Enhanced with Marketing efficiency and SG&A 5Y comparison metrics
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_cost_structure_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                     TEXT,
                cogs_to_revenue          NUMERIC,
                opex_to_revenue          NUMERIC,
                sga_to_revenue           NUMERIC,
                rnd_to_revenue           NUMERIC,
                interest_to_revenue      NUMERIC,
                sga_trend_yoy            NUMERIC,
                operating_leverage_proxy NUMERIC,
                cost_efficiency_score    NUMERIC,
                -- NEW: Marketing efficiency metrics
                marketing_to_revenue     NUMERIC,
                marketing_trend_yoy      NUMERIC,
                marketing_vs_5y_avg      NUMERIC,
                -- NEW: SG&A efficiency
                sga_vs_5y_avg            NUMERIC,
                sga_efficiency_trend     NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                          AS isin,
       public.safe_divide("Cost Of Revenues (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) *
       100                                                                             AS cogs_to_revenue,
       public.safe_divide("Total Operating Expenses (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) *
       100                                                                             AS opex_to_revenue,
       public.safe_divide("Selling General & Admin Expenses/Total (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC) * 100
                                                                                       AS sga_to_revenue,
       public.safe_divide("R&D Expenses (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) *
       100                                                                             AS rnd_to_revenue,
       public.safe_divide("Interest Expense/Total (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) *
       100                                                                             AS interest_to_revenue,
       -- SG&A trend using available FY columns
       (public.safe_divide("Selling General & Admin Expenses/Total (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC) -
        public.safe_divide("Selling General & Admin Expenses/Total (-1FY)"::NUMERIC,
                           "Total Revenues (-1FY)"::NUMERIC)) * 100
                                                                                       AS sga_trend_yoy,
       CASE
           WHEN public.calc_change_ratio("Total Revenues (FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC) > 0
               THEN public.safe_divide(
                   public.calc_change_ratio("Operating Income (FY)"::NUMERIC, "Operating Income (-1FY)"::NUMERIC),
                   public.calc_change_ratio("Total Revenues (FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC)
                    )
           END                                                                         AS operating_leverage_proxy,
       public.clamp_score(
               100 -
               public.safe_divide("Cost Of Revenues (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) * 100 * 0.5 -
               public.safe_divide("Total Operating Expenses (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) * 100 *
               0.3
       )                                                                               AS cost_efficiency_score,
       -- NEW: Marketing efficiency metrics using schema columns
       public.safe_divide("Marketing Expenses (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC) *
       100                                                                             AS marketing_to_revenue,
       public.pct_change("Marketing Expenses (FY)"::NUMERIC,
                         "Marketing Expenses (-1FY)"::NUMERIC)                         AS marketing_trend_yoy,
       public.safe_divide("Marketing Expenses (FY)"::NUMERIC,
                          "Marketing Expenses (5YAVGLTM)"::NUMERIC)                    AS marketing_vs_5y_avg,
       -- NEW: SG&A vs 5Y average
       public.safe_divide("Selling General & Admin Expenses/Total (FQ)"::NUMERIC,
                          "Selling General & Admin Expenses/Total (5YAVGFQ)"::NUMERIC) AS sga_vs_5y_avg,
       -- NEW: SG&A efficiency trend (lower ratio = better efficiency)
       (public.safe_divide("Selling General & Admin Expenses/Total (-1FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC) -
        public.safe_divide("Selling General & Admin Expenses/Total (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC)) *
       100
                                                                                       AS sga_efficiency_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Dividend Yield Comprehensive
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_dividend_yield_comprehensive(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                div_yield_ltm             NUMERIC,
                div_yield_ntm             NUMERIC,
                div_yield_ind             NUMERIC,
                div_yield_1fy_ind         NUMERIC,
                div_yield_5y_avg          NUMERIC,
                div_yield_vs_5y_avg       NUMERIC,
                div_yield_growth_expected NUMERIC,
                dividend_streak           INTEGER,
                high_yield_flag           INTEGER,
                sustainable_dividend_flag INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                AS isin,
       "Div Yield (LTM)"                                     AS div_yield_ltm,
       "Div Yield (NTM)"                                     AS div_yield_ntm,
       "Div Yield (Ind)"                                     AS div_yield_ind,
       "Div Yield (-1FYInd)"                                 AS div_yield_1fy_ind,
       "Div Yield (5YAVGLTM)"                                AS div_yield_5y_avg,
       "Div Yield (LTM)" / NULLIF("Div Yield (5YAVGLTM)", 0) AS div_yield_vs_5y_avg,
       ("Div Yield (NTM)" - "Div Yield (LTM)") / NULLIF("Div Yield (LTM)", 0) * 100
                                                             AS div_yield_growth_expected,
       "Dividend Streak"::INTEGER                            AS dividend_streak,
       CASE WHEN "Div Yield (LTM)" > 0.05 THEN 1 ELSE 0 END  AS high_yield_flag,
       CASE
           WHEN "Div Yield (LTM)" > 0
               AND "FCF (LTM)" > ABS(COALESCE("Common Dividends Paid (LTM)", 0))
               AND "Dividend Streak" >= 5
               THEN 1
           ELSE 0
           END                                               AS sustainable_dividend_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Interest Income Features
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_interest_income_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                        TEXT,
                interest_income_ltm         NUMERIC,
                interest_expense_ltm        NUMERIC,
                net_interest_income         NUMERIC,
                interest_coverage_ratio     NUMERIC,
                interest_income_to_revenue  NUMERIC,
                interest_expense_to_revenue NUMERIC,
                net_interest_margin_proxy   NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                   AS isin,
       "Interest And Investment Income (LTM)"                                   AS interest_income_ltm,
       "Interest Expense/Total (LTM)"                                           AS interest_expense_ltm,
       COALESCE("Interest And Investment Income (LTM)", 0) -
       COALESCE("Interest Expense/Total (LTM)", 0)                              AS net_interest_income,
       "EBIT (LTM)" / NULLIF("Interest Expense/Total (LTM)", 0)                 AS interest_coverage_ratio,
       "Interest And Investment Income (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100
                                                                                AS interest_income_to_revenue,
       "Interest Expense/Total (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100 AS interest_expense_to_revenue,
       (COALESCE("Interest And Investment Income (LTM)", 0) -
        COALESCE("Interest Expense/Total (LTM)", 0)) /
       NULLIF("Total Assets (LTM)", 0) * 100                                    AS net_interest_margin_proxy
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Long Term Momentum Features (REFACTORED)
-- Uses available price columns (1Y, 3Y, 5Y) and helper functions
-- Note: Price (2Y Ago) not available in schema, adjusted weights accordingly
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_long_term_momentum_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                     TEXT,
                price_momentum_1y        NUMERIC,
                price_momentum_3y        NUMERIC,
                price_momentum_5y        NUMERIC,
                long_term_trend_score    NUMERIC,
                price_vs_ema_250d        NUMERIC,
                multi_year_high_flag     INTEGER,
                secular_trend_flag       INTEGER,
                total_return_ytd         NUMERIC,
                total_return_5y          NUMERIC,
                total_return_10y         NUMERIC,
                return_cagr_3y           NUMERIC,
                return_cagr_10y          NUMERIC,
                return_vs_price_momentum NUMERIC,
                return_consistency_score NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                              AS isin,
       public.pct_change("Last Price"::NUMERIC, "Price (1Y Ago)"::NUMERIC) AS price_momentum_1y,
       public.pct_change("Last Price"::NUMERIC, "Price (3Y Ago)"::NUMERIC) AS price_momentum_3y,
       public.pct_change("Last Price"::NUMERIC, "Price (5Y Ago)"::NUMERIC) AS price_momentum_5y,
       -- Weighted trend score using available periods (1Y: 50%, 3Y: 30%, 5Y: 20%)
       (COALESCE(public.pct_change("Last Price"::NUMERIC, "Price (1Y Ago)"::NUMERIC), 0) * 0.50 +
        COALESCE(public.pct_change("Last Price"::NUMERIC, "Price (3Y Ago)"::NUMERIC), 0) * 0.30 +
        COALESCE(public.pct_change("Last Price"::NUMERIC, "Price (5Y Ago)"::NUMERIC), 0) * 0.20) / 100
                                                                           AS long_term_trend_score,
       public.pct_change("Last Price"::NUMERIC, "EMA (250D)"::NUMERIC)     AS price_vs_ema_250d,
       CASE
           WHEN public.calc_change_ratio("52W High/Adj"::NUMERIC - "Last Price", "52W High/Adj"::NUMERIC) <= 0.10
               AND public.calc_change_ratio("Last Price"::NUMERIC, "Price (3Y Ago)"::NUMERIC) > 0.5
               THEN 1
           ELSE 0
           END                                                             AS multi_year_high_flag,
       CASE
           WHEN public.calc_change_ratio("Last Price"::NUMERIC, "Price (3Y Ago)"::NUMERIC) > 0.20
               AND public.calc_change_ratio("Last Price"::NUMERIC, "Price (1Y Ago)"::NUMERIC) > 0
               AND "EMA (50D)" > "EMA (250D)"
               THEN 1
           ELSE 0
           END                                                             AS secular_trend_flag,
       "Total Return (YTD)"                                                AS total_return_ytd,
       "Total Return (5Y)"                                                 AS total_return_5y,
       "Total Return (10Y)"                                                AS total_return_10y,
       "Tot. Return %/CAGR (3Y)"                                           AS return_cagr_3y,
       "Tot. Return %/CAGR (10Y)"                                          AS return_cagr_10y,
       "Tot. Return %/CAGR (3Y)" - public.pct_change("Last Price"::NUMERIC, "Price (3Y Ago)"::NUMERIC)
                                                                           AS return_vs_price_momentum,
       public.safe_divide("Tot. Return %/CAGR (3Y)", "Volatility (1Y)")    AS return_consistency_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Revenue Estimate Consensus (REFACTORED)
-- Uses available estimate columns (Avg and Med only)
-- Note: High, Low, # columns not available in schema
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_revenue_estimate_consensus(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                       TEXT,
                revenue_est_avg_fy1e       NUMERIC,
                revenue_est_med_fy1e       NUMERIC,
                revenue_est_avg_ntm        NUMERIC,
                revenue_est_med_ntm        NUMERIC,
                revenue_avg_med_diff_pct   NUMERIC,
                revenue_consensus_strength NUMERIC,
                revenue_revision_trend     NUMERIC,
                revenue_vs_current         NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                                    AS isin,
       "Revenues - Est Avg (FY1E)"                                                               AS revenue_est_avg_fy1e,
       "Revenues - Est Med (FY1E)"                                                               AS revenue_est_med_fy1e,
       "Revenues - Est Avg (NTM)"                                                                AS revenue_est_avg_ntm,
       "Revenues - Est Med (NTM)"                                                                AS revenue_est_med_ntm,
       -- Difference between avg and median as proxy for estimate dispersion
       public.safe_divide("Revenues - Est Avg (FY1E)"::NUMERIC - "Revenues - Est Med (FY1E)",
                          "Revenues - Est Med (FY1E)"::NUMERIC) *
       100                                                                                       AS revenue_avg_med_diff_pct,
       -- Consensus strength: closer avg to median = stronger consensus
       public.clamp_score(
               100 - ABS(public.safe_divide("Revenues - Est Avg (FY1E)"::NUMERIC - "Revenues - Est Med (FY1E)",
                                            "Revenues - Est Med (FY1E)"::NUMERIC) * 100) * 2
       )                                                                                         AS revenue_consensus_strength,
       "Revenues - Est YoY % (FY1E)"                                                             AS revenue_revision_trend,
       -- Compare estimate to current revenue
       public.safe_divide("Revenues - Est Avg (FY1E)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) AS revenue_vs_current
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Revenue Features (REFACTORED)
-- Uses available revenue columns (FQ, FY, -1FY, LTM, 5YAVG)
-- Note: Quarterly historical columns (-1FQFQ to -4FQFQ) not available in schema
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_revenue_quarterly_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                        TEXT,
                -- Base revenue values
                revenue_fq                  NUMERIC,
                revenue_fy                  NUMERIC,
                revenue_ltm                 NUMERIC,
                revenue_5y_avg              NUMERIC,
                -- NEW: Quarterly historical values
                revenue_1fqfq               NUMERIC,
                revenue_2fqfq               NUMERIC,
                revenue_3fqfq               NUMERIC,
                revenue_4fqfq               NUMERIC,
                -- NEW: Extended yearly historical
                revenue_1fy                 NUMERIC,
                revenue_2fy                 NUMERIC,
                revenue_3fy                 NUMERIC,
                revenue_4fy                 NUMERIC,
                -- Growth metrics
                revenue_yoy_growth          NUMERIC,
                revenue_vs_5y_avg           NUMERIC,
                revenue_ltm_vs_fy           NUMERIC,
                revenue_fq_vs_5y_avg_fq     NUMERIC,
                -- NEW: Quarterly momentum metrics
                revenue_qoq_growth          NUMERIC,
                revenue_qoq_2q              NUMERIC,
                revenue_qoq_3q              NUMERIC,
                revenue_qoq_4q              NUMERIC,
                -- NEW: YoY quarterly comparison
                revenue_yoy_quarterly       NUMERIC,
                -- NEW: Multi-year growth
                revenue_2y_growth           NUMERIC,
                revenue_3y_growth           NUMERIC,
                revenue_4y_growth           NUMERIC,
                -- NEW: CAGR calculations
                revenue_cagr_3y             NUMERIC,
                revenue_cagr_4y             NUMERIC,
                -- NEW: Quarterly trend analysis
                revenue_4q_trend            NUMERIC,
                revenue_4q_avg              NUMERIC,
                revenue_fq_vs_4q_avg        NUMERIC,
                -- Flags and scores
                revenue_growth_flag         INTEGER,
                revenue_stability_score     NUMERIC,
                -- NEW: Additional flags
                revenue_accelerating_flag   INTEGER,
                revenue_positive_qoq_streak INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                                    AS isin,
       -- Base revenue values
       "Total Revenues (FQ)"                                                                     AS revenue_fq,
       "Total Revenues (FY)"                                                                     AS revenue_fy,
       "Total Revenues (LTM)"                                                                    AS revenue_ltm,
       "Total Revenues (5YAVGLTM)"                                                               AS revenue_5y_avg,
       -- Quarterly historical values
       "Total Revenues (-1FQFQ)"                                                                 AS revenue_1fqfq,
       "Total Revenues (-2FQFQ)"                                                                 AS revenue_2fqfq,
       "Total Revenues (-3FQFQ)"                                                                 AS revenue_3fqfq,
       "Total Revenues (-4FQFQ)"                                                                 AS revenue_4fqfq,
       -- Extended yearly historical
       "Total Revenues (-1FY)"                                                                   AS revenue_1fy,
       "Total Revenues (-2FY)"                                                                   AS revenue_2fy,
       "Total Revenues (-3FY)"                                                                   AS revenue_3fy,
       "Total Revenues (-4FY)"                                                                   AS revenue_4fy,
       -- Year-over-year growth using FY data
       public.pct_change("Total Revenues (FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC)       AS revenue_yoy_growth,
       -- Current vs 5-year average
       public.safe_divide("Total Revenues (LTM)"::NUMERIC, "Total Revenues (5YAVGLTM)"::NUMERIC) AS revenue_vs_5y_avg,
       -- LTM vs FY comparison
       public.safe_divide("Total Revenues (LTM)"::NUMERIC, "Total Revenues (FY)"::NUMERIC)       AS revenue_ltm_vs_fy,
       -- FQ vs 5-year average FQ
       public.safe_divide("Total Revenues (FQ)"::NUMERIC,
                          "Total Revenues (5YAVGFQ)"::NUMERIC)                                   AS revenue_fq_vs_5y_avg_fq,
       -- Quarterly momentum: QoQ growth rates
       public.pct_change("Total Revenues (FQ)"::NUMERIC, "Total Revenues (-1FQFQ)"::NUMERIC)     AS revenue_qoq_growth,
       public.pct_change("Total Revenues (-1FQFQ)"::NUMERIC, "Total Revenues (-2FQFQ)"::NUMERIC) AS revenue_qoq_2q,
       public.pct_change("Total Revenues (-2FQFQ)"::NUMERIC, "Total Revenues (-3FQFQ)"::NUMERIC) AS revenue_qoq_3q,
       public.pct_change("Total Revenues (-3FQFQ)"::NUMERIC, "Total Revenues (-4FQFQ)"::NUMERIC) AS revenue_qoq_4q,
       -- YoY quarterly comparison (current FQ vs same quarter last year)
       public.pct_change("Total Revenues (FQ)"::NUMERIC,
                         "Total Revenues (-4FQFQ)"::NUMERIC)                                     AS revenue_yoy_quarterly,
       -- Multi-year growth rates
       public.pct_change("Total Revenues (FY)"::NUMERIC, "Total Revenues (-2FY)"::NUMERIC)       AS revenue_2y_growth,
       public.pct_change("Total Revenues (FY)"::NUMERIC, "Total Revenues (-3FY)"::NUMERIC)       AS revenue_3y_growth,
       public.pct_change("Total Revenues (FY)"::NUMERIC, "Total Revenues (-4FY)"::NUMERIC)       AS revenue_4y_growth,
       -- CAGR calculations
       CASE
           WHEN "Total Revenues (-3FY)" > 0 AND "Total Revenues (FY)" > 0
               THEN
               (POWER(public.safe_divide("Total Revenues (FY)"::NUMERIC, "Total Revenues (-3FY)"::NUMERIC), 1.0 / 3.0) -
                1) *
               100
           END                                                                                   AS revenue_cagr_3y,
       CASE
           WHEN "Total Revenues (-4FY)" > 0 AND "Total Revenues (FY)" > 0
               THEN
               (POWER(public.safe_divide("Total Revenues (FY)"::NUMERIC, "Total Revenues (-4FY)"::NUMERIC), 1.0 / 4.0) -
                1) *
               100
           END                                                                                   AS revenue_cagr_4y,
       -- Quarterly trend: FQ vs 4 quarters ago
       public.pct_change("Total Revenues (FQ)"::NUMERIC, "Total Revenues (-4FQFQ)"::NUMERIC)     AS revenue_4q_trend,
       -- Trailing 4-quarter average
       ("Total Revenues (FQ)" + "Total Revenues (-1FQFQ)" +
        "Total Revenues (-2FQFQ)" + "Total Revenues (-3FQFQ)") / 4.0                             AS revenue_4q_avg,
       -- FQ vs trailing 4Q average
       public.safe_divide("Total Revenues (FQ)"::NUMERIC,
                          ("Total Revenues (FQ)" + "Total Revenues (-1FQFQ)" +
                           "Total Revenues (-2FQFQ)" + "Total Revenues (-3FQFQ)") /
                          4.0)                                                                   AS revenue_fq_vs_4q_avg,
       -- Growth flag: 1 if growing YoY
       CASE
           WHEN "Total Revenues (FY)" > "Total Revenues (-1FY)" THEN 1
           ELSE 0
           END                                                                                   AS revenue_growth_flag,
       -- Revenue stability: how close LTM is to 5Y average
       public.clamp_score(
               100 - ABS(public.safe_divide("Total Revenues (LTM)"::NUMERIC - "Total Revenues (5YAVGLTM)",
                                            "Total Revenues (5YAVGLTM)"::NUMERIC)) * 100
       )                                                                                         AS revenue_stability_score,
       -- Accelerating growth flag: recent growth > historical growth
       CASE
           WHEN public.pct_change("Total Revenues (FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC) >
                public.pct_change("Total Revenues (-1FY)"::NUMERIC, "Total Revenues (-2FY)"::NUMERIC)
               THEN 1
           ELSE 0
           END                                                                                   AS revenue_accelerating_flag,
       -- Positive QoQ streak count
       (CASE WHEN "Total Revenues (FQ)" > "Total Revenues (-1FQFQ)" THEN 1 ELSE 0 END +
        CASE WHEN "Total Revenues (-1FQFQ)" > "Total Revenues (-2FQFQ)" THEN 1 ELSE 0 END +
        CASE WHEN "Total Revenues (-2FQFQ)" > "Total Revenues (-3FQFQ)" THEN 1 ELSE 0 END +
        CASE WHEN "Total Revenues (-3FQFQ)" > "Total Revenues (-4FQFQ)" THEN 1 ELSE 0 END)::INTEGER
                                                                                                 AS revenue_positive_qoq_streak
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Tangible Book Features (REFACTORED)
-- Uses native TBV columns from schema for accuracy
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_tangible_book_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                -- Use native schema columns directly
                tangible_book_value_fy  NUMERIC,
                tangible_book_value_ltm NUMERIC,
                -- Calculated metrics
                tangible_book_per_share NUMERIC,
                price_to_tangible_book  NUMERIC,
                tangible_equity_ratio   NUMERIC,
                intangibles_to_equity   NUMERIC,
                goodwill_to_equity      NUMERIC,
                tangible_asset_quality  NUMERIC,
                -- NEW: TBV growth metrics
                tbv_yoy_growth          NUMERIC,
                tbv_vs_calculated       NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                                 AS isin,
       -- Use native TBV columns from schema (more accurate than calculation)
       "TBV (FY)"                                                                             AS tangible_book_value_fy,
       "TBV (LTM)"                                                                            AS tangible_book_value_ltm,
       -- Per share using native TBV
       "TBV (LTM)" / NULLIF("Shrs Out", 0)                                                    AS tangible_book_per_share,
       -- P/TBV using native column (already in schema as P/TBV (LTM))
       "P/TBV (LTM)"                                                                          AS price_to_tangible_book,
       -- Tangible equity ratio using native TBV
       "TBV (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100                                    AS tangible_equity_ratio,
       COALESCE("Gross Intangible Assets (LTM)", 0) / NULLIF("Total Equity (LTM)", 0) * 100
                                                                                              AS intangibles_to_equity,
       COALESCE("Goodwill (LTM)", 0) / NULLIF("Total Equity (LTM)", 0) * 100                  AS goodwill_to_equity,
       GREATEST(0, LEAST(100,
                         100 - (COALESCE("Goodwill (LTM)", 0) + COALESCE("Gross Intangible Assets (LTM)", 0)) /
                               NULLIF("Total Assets (LTM)", 0) * 100
                   ))                                                                         AS tangible_asset_quality,
       -- NEW: TBV growth (FY to LTM)
       public.pct_change("TBV (LTM)"::NUMERIC, "TBV (FY)"::NUMERIC)                           AS tbv_yoy_growth,
       -- Validation: compare native TBV to calculated (should be ~1.0)
       public.safe_divide("TBV (LTM)"::NUMERIC, "Total Equity (LTM)"::NUMERIC - COALESCE("Goodwill (LTM)", 0) -
                                                COALESCE("Gross Intangible Assets (LTM)", 0)) AS tbv_vs_calculated
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Inventory Temporal Features (NEW)
-- Full historical coverage for inventory trend analysis
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_inventory_temporal_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                     TEXT,
                -- Current values
                inventory_ltm            NUMERIC,
                inventory_fq             NUMERIC,
                inventory_fy             NUMERIC,
                -- Quarterly historical
                inventory_1fq            NUMERIC,
                inventory_2fq            NUMERIC,
                inventory_3fq            NUMERIC,
                inventory_4fq            NUMERIC,
                -- Yearly historical
                inventory_1fy            NUMERIC,
                inventory_2fy            NUMERIC,
                inventory_3fy            NUMERIC,
                inventory_4fy            NUMERIC,
                -- Trend metrics
                inventory_qoq_change     NUMERIC,
                inventory_yoy_change     NUMERIC,
                inventory_4q_trend       NUMERIC,
                inventory_vs_5y_avg      NUMERIC,
                -- Efficiency metrics
                inventory_days           NUMERIC,
                inventory_turnover       NUMERIC,
                inventory_to_revenue     NUMERIC,
                inventory_to_assets      NUMERIC,
                -- Quality flags
                inventory_buildup_flag   INTEGER,
                inventory_reduction_flag INTEGER,
                inventory_volatility     NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                                AS isin,
       -- Current values
       "Inventory (LTM)"                                                                     AS inventory_ltm,
       "Inventory (FQ)"                                                                      AS inventory_fq,
       "Inventory (FY)"                                                                      AS inventory_fy,
       -- Quarterly historical
       "Inventory (-1FQ)"                                                                    AS inventory_1fq,
       "Inventory (-2FQ)"                                                                    AS inventory_2fq,
       "Inventory (-3FQ)"                                                                    AS inventory_3fq,
       "Inventory (-4FQ)"                                                                    AS inventory_4fq,
       -- Yearly historical
       "Inventory (-1FY)"                                                                    AS inventory_1fy,
       "Inventory (-2FY)"                                                                    AS inventory_2fy,
       "Inventory (-3FY)"                                                                    AS inventory_3fy,
       "Inventory (-4FY)"                                                                    AS inventory_4fy,
       -- Trend metrics
       public.pct_change("Inventory (FQ)"::NUMERIC, "Inventory (-1FQ)"::NUMERIC)             AS inventory_qoq_change,
       public.pct_change("Inventory (FY)"::NUMERIC, "Inventory (-1FY)"::NUMERIC)             AS inventory_yoy_change,
       public.pct_change("Inventory (FQ)"::NUMERIC, "Inventory (-4FQ)"::NUMERIC)             AS inventory_4q_trend,
       public.safe_divide("Inventory (FQ)"::NUMERIC, "Inventory (5YAVGFQ)"::NUMERIC)         AS inventory_vs_5y_avg,
       -- Efficiency metrics
       "Inventory (LTM)" / NULLIF("Cost Of Revenues (LTM)" / 365.0, 0)                       AS inventory_days,
       "Cost Of Revenues (LTM)" / NULLIF("Inventory (LTM)", 0)                               AS inventory_turnover,
       public.safe_divide("Inventory (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) * 100 AS inventory_to_revenue,
       public.safe_divide("Inventory (LTM)"::NUMERIC, "Total Assets (LTM)"::NUMERIC) * 100   AS inventory_to_assets,
       -- Inventory buildup flag (rising faster than revenue)
       CASE
           WHEN public.pct_change("Inventory (FQ)"::NUMERIC, "Inventory (-4FQ)"::NUMERIC) >
                public.pct_change("Total Revenues (FQ)"::NUMERIC, "Total Revenues (-4FQFQ)"::NUMERIC) + 10
               THEN 1
           ELSE 0 END                                                                        AS inventory_buildup_flag,
       -- Inventory reduction flag (declining)
       CASE
           WHEN "Inventory (FQ)" < "Inventory (-1FQ)"
               AND "Inventory (-1FQ)" < "Inventory (-2FQ)"
               THEN 1
           ELSE 0 END                                                                        AS inventory_reduction_flag,
       -- Volatility (coefficient of variation)
       (ABS("Inventory (FQ)" - "Inventory (-1FQ)") +
        ABS("Inventory (-1FQ)" - "Inventory (-2FQ)") +
        ABS("Inventory (-2FQ)" - "Inventory (-3FQ)") +
        ABS("Inventory (-3FQ)" - "Inventory (-4FQ)")) /
       NULLIF(ABS(("Inventory (FQ)" + "Inventory (-1FQ)" + "Inventory (-2FQ)" +
                   "Inventory (-3FQ)" + "Inventory (-4FQ)") / 5.0), 0)                       AS inventory_volatility
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Goodwill Temporal Features (NEW)
-- M&A activity tracking through goodwill changes
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_goodwill_temporal_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                       TEXT,
                -- Current values
                goodwill_fq                NUMERIC,
                goodwill_ltm               NUMERIC,
                goodwill_fy                NUMERIC,
                -- Quarterly historical
                goodwill_1fq               NUMERIC,
                goodwill_2fq               NUMERIC,
                goodwill_3fq               NUMERIC,
                goodwill_4fq               NUMERIC,
                -- Yearly historical
                goodwill_1fy               NUMERIC,
                goodwill_2fy               NUMERIC,
                goodwill_3fy               NUMERIC,
                goodwill_4fy               NUMERIC,
                -- Trend metrics
                goodwill_qoq_change        NUMERIC,
                goodwill_yoy_change        NUMERIC,
                goodwill_3y_growth         NUMERIC,
                goodwill_vs_5y_avg         NUMERIC,
                -- M&A activity indicators
                recent_acquisition_flag    INTEGER,
                goodwill_accumulation_rate NUMERIC,
                goodwill_to_assets_trend   NUMERIC,
                -- Risk metrics
                impairment_risk_score      NUMERIC,
                goodwill_concentration     NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                             AS isin,
       -- Current values
       "Goodwill (FQ)"                                                                    AS goodwill_fq,
       "Goodwill (LTM)"                                                                   AS goodwill_ltm,
       "Goodwill (FY)"                                                                    AS goodwill_fy,
       -- Quarterly historical
       "Goodwill (-1FQ)"                                                                  AS goodwill_1fq,
       "Goodwill (-2FQ)"                                                                  AS goodwill_2fq,
       "Goodwill (-3FQ)"                                                                  AS goodwill_3fq,
       "Goodwill (-4FQ)"                                                                  AS goodwill_4fq,
       -- Yearly historical
       "Goodwill (-1FY)"                                                                  AS goodwill_1fy,
       "Goodwill (-2FY)"                                                                  AS goodwill_2fy,
       "Goodwill (-3FY)"                                                                  AS goodwill_3fy,
       "Goodwill (-4FY)"                                                                  AS goodwill_4fy,
       -- Trend metrics
       public.pct_change("Goodwill (FQ)"::NUMERIC, "Goodwill (-1FQ)"::NUMERIC)            AS goodwill_qoq_change,
       public.pct_change("Goodwill (FY)"::NUMERIC, "Goodwill (-1FY)"::NUMERIC)            AS goodwill_yoy_change,
       public.pct_change("Goodwill (FY)"::NUMERIC, "Goodwill (-3FY)"::NUMERIC)            AS goodwill_3y_growth,
       public.safe_divide("Goodwill (FQ)"::NUMERIC, "Goodwill (5YAVGFQ)"::NUMERIC)        AS goodwill_vs_5y_avg,
       -- Recent acquisition flag (goodwill increased significantly)
       CASE
           WHEN public.pct_change("Goodwill (FQ)"::NUMERIC, "Goodwill (-1FQ)"::NUMERIC) > 20
               THEN 1
           ELSE 0 END                                                                     AS recent_acquisition_flag,
       -- Goodwill accumulation rate (avg annual increase)
       CASE
           WHEN "Goodwill (-3FY)" > 0
               THEN (POWER(public.safe_divide("Goodwill (FY)"::NUMERIC, "Goodwill (-3FY)"::NUMERIC), 1.0 / 3.0) - 1) *
                    100
           END                                                                            AS goodwill_accumulation_rate,
       -- Goodwill to assets trend (increasing concentration risk)
       (public.safe_divide("Goodwill (FY)"::NUMERIC, "Total Assets (FY)"::NUMERIC) -
        public.safe_divide("Goodwill (-1FY)"::NUMERIC, "Total Assets (-1FY)"::NUMERIC)) *
       100                                                                                AS goodwill_to_assets_trend,
       -- Impairment risk score (high goodwill + declining earnings = risk)
       CASE
           WHEN "Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0) > 0.25
               AND "Net Income - (IS) (FY)" < "Net Income - (IS) (-1FY)"
               THEN public.clamp_score(
                   ("Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0)) * 200 +
                   ABS(public.pct_change("Net Income - (IS) (FY)"::NUMERIC, "Net Income - (IS) (-1FY)"::NUMERIC)) * 0.5
                    )
           ELSE public.clamp_score(
                   ("Goodwill (LTM)" / NULLIF("Total Assets (LTM)", 0)) * 100
                )
           END                                                                            AS impairment_risk_score,
       -- Goodwill concentration (relative to equity)
       public.safe_divide("Goodwill (LTM)"::NUMERIC, "Total Equity (LTM)"::NUMERIC) * 100 AS goodwill_concentration
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- R&D Investment Temporal Features (NEW)
-- Innovation investment trends and efficiency
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_rnd_temporal_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                    TEXT,
                -- Current values
                rnd_ltm                 NUMERIC,
                rnd_fq                  NUMERIC,
                rnd_fy                  NUMERIC,
                -- Quarterly historical
                rnd_1fqfq               NUMERIC,
                rnd_2fqfq               NUMERIC,
                rnd_3fqfq               NUMERIC,
                rnd_4fqfq               NUMERIC,
                -- Yearly historical
                rnd_1fy                 NUMERIC,
                rnd_2fy                 NUMERIC,
                rnd_3fy                 NUMERIC,
                rnd_4fy                 NUMERIC,
                -- Intensity metrics
                rnd_intensity_ltm       NUMERIC,
                rnd_intensity_fy        NUMERIC,
                rnd_intensity_trend     NUMERIC,
                -- Growth metrics
                rnd_qoq_growth          NUMERIC,
                rnd_yoy_growth          NUMERIC,
                rnd_cagr_3y             NUMERIC,
                -- Efficiency metrics
                rnd_per_employee        NUMERIC,
                rnd_to_gross_profit     NUMERIC,
                rnd_roi_proxy           NUMERIC,
                -- Investment flags
                rnd_increasing_flag     INTEGER,
                rnd_cut_flag            INTEGER,
                high_rnd_intensity_flag INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                                                 AS isin,
       -- Current values
       "R&D Expenses (LTM)"                                                                   AS rnd_ltm,
       "R&D Expenses (FQ)"                                                                    AS rnd_fq,
       "R&D Expenses (FY)"                                                                    AS rnd_fy,
       -- Quarterly historical
       "R&D Expenses (-1FQFQ)"                                                                AS rnd_1fqfq,
       "R&D Expenses (-2FQFQ)"                                                                AS rnd_2fqfq,
       "R&D Expenses (-3FQFQ)"                                                                AS rnd_3fqfq,
       "R&D Expenses (-4FQFQ)"                                                                AS rnd_4fqfq,
       -- Yearly historical
       "R&D Expenses (-1FY)"                                                                  AS rnd_1fy,
       "R&D Expenses (-2FY)"                                                                  AS rnd_2fy,
       "R&D Expenses (-3FY)"                                                                  AS rnd_3fy,
       "R&D Expenses (-4FY)"                                                                  AS rnd_4fy,
       -- Intensity metrics (R&D / Revenue)
       public.safe_divide("R&D Expenses (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) *
       100                                                                                    AS rnd_intensity_ltm,
       public.safe_divide("R&D Expenses (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC) * 100 AS rnd_intensity_fy,
       -- Intensity trend (increasing R&D commitment)
       (public.safe_divide("R&D Expenses (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC) -
        public.safe_divide("R&D Expenses (-1FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC)) *
       100                                                                                    AS rnd_intensity_trend,
       -- Growth metrics
       public.pct_change("R&D Expenses (FQ)"::NUMERIC, "R&D Expenses (-1FQFQ)"::NUMERIC)      AS rnd_qoq_growth,
       public.pct_change("R&D Expenses (FY)"::NUMERIC, "R&D Expenses (-1FY)"::NUMERIC)        AS rnd_yoy_growth,
       CASE
           WHEN "R&D Expenses (-3FY)" > 0 AND "R&D Expenses (FY)" > 0
               THEN
               (POWER(public.safe_divide("R&D Expenses (FY)"::NUMERIC, "R&D Expenses (-3FY)"::NUMERIC), 1.0 / 3.0) -
                1) *
               100
           END                                                                                AS rnd_cagr_3y,
       -- Efficiency metrics
       public.safe_divide("R&D Expenses (FY)"::NUMERIC, "Full Time Employees (FY)"::NUMERIC)  AS rnd_per_employee,
       public.safe_divide("R&D Expenses (LTM)"::NUMERIC, "Gross Profit (LTM)"::NUMERIC) *
       100                                                                                    AS rnd_to_gross_profit,
       -- R&D ROI proxy: revenue growth relative to R&D spend
       CASE
           WHEN "R&D Expenses (-1FY)" > 0
               THEN public.safe_divide(
                   public.pct_change("Total Revenues (FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC),
                   public.safe_divide("R&D Expenses (-1FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC) * 100
                    )
           END                                                                                AS rnd_roi_proxy,
       -- R&D increasing flag (4 consecutive quarterly increases)
       CASE
           WHEN "R&D Expenses (FQ)" > "R&D Expenses (-1FQFQ)"
               AND "R&D Expenses (-1FQFQ)" > "R&D Expenses (-2FQFQ)"
               AND "R&D Expenses (-2FQFQ)" > "R&D Expenses (-3FQFQ)"
               THEN 1
           ELSE 0 END                                                                         AS rnd_increasing_flag,
       -- R&D cut flag (significant decline may signal distress)
       CASE
           WHEN public.pct_change("R&D Expenses (FY)"::NUMERIC, "R&D Expenses (-1FY)"::NUMERIC) < -15
               THEN 1
           ELSE 0 END                                                                         AS rnd_cut_flag,
       -- High R&D intensity flag (tech/pharma typical >10%)
       CASE
           WHEN public.safe_divide("R&D Expenses (LTM)"::NUMERIC, "Total Revenues (LTM)"::NUMERIC) > 0.10
               THEN 1
           ELSE 0 END                                                                         AS high_rnd_intensity_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Unusual Items Features (REFACTORED)
-- Uses "Other Unusual Items/Total (LTM)" - only LTM period available
-- Combines with other non-recurring items for comprehensive view
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_unusual_items_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                other_unusual_items_ltm   NUMERIC,
                impairment_goodwill_ltm   NUMERIC,
                asset_writedown_ltm       NUMERIC,
                restructuring_charges_ltm NUMERIC,
                total_unusual_items       NUMERIC,
                unusual_items_to_revenue  NUMERIC,
                unusual_items_to_ebitda   NUMERIC,
                has_unusual_items_flag    INTEGER,
                earnings_quality_impact   NUMERIC
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                     AS isin,
       "Other Unusual Items/Total (LTM)"          AS other_unusual_items_ltm,
       "Impairment of Goodwill (LTM)"             AS impairment_goodwill_ltm,
       "Asset Writedown (LTM)"                    AS asset_writedown_ltm,
       "Restructuring Charges (LTM)"              AS restructuring_charges_ltm,
       -- Total unusual/non-recurring items
       COALESCE("Other Unusual Items/Total (LTM)", 0) +
       COALESCE("Impairment of Goodwill (LTM)", 0) +
       COALESCE("Asset Writedown (LTM)", 0) +
       COALESCE("Restructuring Charges (LTM)", 0) AS total_unusual_items,
       -- Unusual items as % of revenue
       public.safe_divide(
               ABS(COALESCE("Other Unusual Items/Total (LTM)", 0) +
                   COALESCE("Impairment of Goodwill (LTM)", 0) +
                   COALESCE("Asset Writedown (LTM)", 0) +
                   COALESCE("Restructuring Charges (LTM)", 0)),
               "Total Revenues (LTM)"::NUMERIC
       ) * 100                                    AS unusual_items_to_revenue,
       -- Unusual items as % of EBITDA
       public.safe_divide(
               ABS(COALESCE("Other Unusual Items/Total (LTM)", 0) +
                   COALESCE("Impairment of Goodwill (LTM)", 0) +
                   COALESCE("Asset Writedown (LTM)", 0) +
                   COALESCE("Restructuring Charges (LTM)", 0)),
               ABS("EBITDA (LTM)")::NUMERIC
       ) * 100                                    AS unusual_items_to_ebitda,
       -- Flag if any unusual items present
       CASE
           WHEN ABS(COALESCE("Other Unusual Items/Total (LTM)", 0)) +
                ABS(COALESCE("Impairment of Goodwill (LTM)", 0)) +
                ABS(COALESCE("Asset Writedown (LTM)", 0)) +
                ABS(COALESCE("Restructuring Charges (LTM)", 0)) > 0
               THEN 1
           ELSE 0 END                             AS has_unusual_items_flag,
       -- Earnings quality impact (higher = better quality, less impacted by unusual items)
       public.clamp_score(
               100 - public.safe_divide(
                             ABS(COALESCE("Other Unusual Items/Total (LTM)", 0) +
                                 COALESCE("Impairment of Goodwill (LTM)", 0) +
                                 COALESCE("Asset Writedown (LTM)", 0) +
                                 COALESCE("Restructuring Charges (LTM)", 0)),
                             ABS("Net Income - (IS) (LTM)")::NUMERIC
                     ) * 100
       )                                          AS earnings_quality_impact
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- Working Capital Deep Features
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_working_capital_deep_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                working_capital_ltm  NUMERIC,
                working_capital_fq   NUMERIC,
                working_capital_fy   NUMERIC,
                wc_to_revenue        NUMERIC,
                wc_to_assets         NUMERIC,
                wc_change_qoq        NUMERIC,
                wc_change_yoy        NUMERIC,
                days_working_capital NUMERIC,
                wc_efficiency_score  NUMERIC,
                negative_wc_flag     INTEGER,
                wc_improvement_flag  INTEGER
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"                                                              AS isin,
       "Working Capital (LTM)"                                             AS working_capital_ltm,
       "Working Capital (FQ)"                                              AS working_capital_fq,
       "Working Capital (FY)"                                              AS working_capital_fy,
       "Working Capital (LTM)" / NULLIF("Total Revenues (LTM)", 0) * 100   AS wc_to_revenue,
       "Working Capital (LTM)" / NULLIF("Total Assets (LTM)", 0) * 100     AS wc_to_assets,
       ("Working Capital (FQ)" - "Working Capital (FY)") /
       NULLIF(ABS("Working Capital (FY)"), 0) * 100                        AS wc_change_qoq,
       ("Working Capital (FY)" - "Working Capital (-1FY)") /
       NULLIF(ABS("Working Capital (-1FY)"), 0) * 100                      AS wc_change_yoy,
       "Working Capital (LTM)" / NULLIF("Total Revenues (LTM)" / 365.0, 0) AS days_working_capital,
       GREATEST(0, LEAST(100,
                         50 + (CASE WHEN "Working Capital (LTM)" > 0 THEN 25 ELSE -25 END) +
                         (CASE WHEN "Current Ratio (LTM)" > 1.5 THEN 15 ELSE 0 END) +
                         (CASE WHEN ("Working Capital (FQ)" - "Working Capital (FY)") > 0 THEN 10 ELSE -10 END)
                   ))                                                      AS wc_efficiency_score,
       CASE WHEN "Working Capital (LTM)" < 0 THEN 1 ELSE 0 END             AS negative_wc_flag,
       CASE
           WHEN "Working Capital (FQ)" > "Working Capital (FY)"
               AND "Working Capital (FY)" > "Working Capital (-1FY)"
               THEN 1
           ELSE 0
           END                                                             AS wc_improvement_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION: VOLATILITY SURFACE FEATURES (NEW)
-- =============================================================================
CREATE OR REPLACE FUNCTION calc_volatility_surface_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                      TEXT,
                vol_1m                    NUMERIC,
                vol_3m                    NUMERIC,
                vol_6m                    NUMERIC,
                vol_1y                    NUMERIC,
                vol_term_spread_short     NUMERIC,
                vol_term_spread_long      NUMERIC,
                vol_ratio_3m_1y           NUMERIC,
                vol_hump                  NUMERIC,
                beta_1y                   NUMERIC,
                beta_2y                   NUMERIC,
                beta_5y                   NUMERIC,
                beta_term_structure       NUMERIC,
                beta_convexity            NUMERIC,
                realized_vs_implied_proxy NUMERIC
            )
    STABLE PARALLEL SAFE
AS
$$
SELECT "ISIN",
       "Volatility (1M)",
       "Volatility (3M)",
       "Volatility (6M)",
       "Volatility (1Y)",
       "Volatility (3M)" - "Volatility (1M)"                             AS vol_term_spread_short,
       "Volatility (1Y)" - "Volatility (6M)"                             AS vol_term_spread_long,
       public.safe_divide("Volatility (3M)", "Volatility (1Y)")          AS vol_ratio_3m_1y,
       "Volatility (6M)" - ("Volatility (3M)" + "Volatility (1Y)") / 2.0 AS vol_hump,
       "Beta (1Y)",
       "Beta (2Y)",
       "Beta (5Y)",
       public.calc_change_ratio("Beta (1Y)", "Beta (5Y)")                AS beta_term_structure,
       "Beta (2Y)" - ("Beta (1Y)" + "Beta (5Y)") / 2.0                   AS beta_convexity,
       public.safe_divide("Volatility (1M)", "Volatility (1Y)")          AS realized_vs_implied_proxy
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION: FORWARD CONSENSUS FEATURES (NEW)
-- =============================================================================
CREATE OR REPLACE FUNCTION calc_forward_consensus_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                         TEXT,
                pe_ntm                       NUMERIC,
                pe_est_fy1                   NUMERIC,
                pe_forward_discount          NUMERIC,
                eps_gaap_vs_norm_ntm         NUMERIC,
                eps_gaap_vs_norm_fy1e        NUMERIC,
                forward_adjustment_trend     NUMERIC,
                ebitda_est_ntm               NUMERIC,
                ebitda_est_fy1e              NUMERIC,
                ev_ebitda_est_fy1            NUMERIC,
                ebitda_forward_growth        NUMERIC,
                earnings_revision_divergence NUMERIC,
                forward_pe_vs_sector_proxy   NUMERIC
            )
    STABLE PARALLEL SAFE
AS
$$
SELECT "ISIN",
       "P/E (NTM)",
       "P/E (EST FY1)",
       public.calc_change_ratio("P/E (NTM)", "P/E (LTM)")                       AS pe_forward_discount,
       "EPS GAAP - Est Avg (NTM)" - "EPS Norm - Est Avg (NTM)"                  AS eps_gaap_vs_norm_ntm,
       "EPS GAAP - Est Avg (FY1E)" - "EPS Norm - Est Avg (FY1E)"                AS eps_gaap_vs_norm_fy1e,
       ("EPS GAAP - Est Avg (FY1E)" - "EPS Norm - Est Avg (FY1E)") -
       ("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)")                             AS forward_adjustment_trend,
       "EBITDA - Est Avg (NTM)",
       "EBITDA - Est Avg (FY1E)",
       "EV/EBITDA (EST FY1)",
       public.calc_change_ratio("EBITDA - Est Avg (FY1E)", "EBITDA (LTM)")      AS ebitda_forward_growth,
       ("EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)") -
       ("EPS Est Avg Rev % (FY1E - 1M)" - "EPS GAAP Est Avg Rev % (FY1E - 1M)") AS earnings_revision_divergence,
       public.calc_change_ratio("P/E (NTM)", "P/E (3YAVGLTM)")                  AS forward_pe_vs_sector_proxy
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION: PRICE TARGET ACHIEVEMENT FEATURES (NEW)
-- =============================================================================
CREATE OR REPLACE FUNCTION calc_price_target_achievement_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                       TEXT,
                pt_achievement_1y          NUMERIC,
                pt_accuracy_1y             NUMERIC,
                pt_optimism_bias           NUMERIC,
                pt_range_hit_rate          NUMERIC,
                pt_median_vs_mean_spread   NUMERIC,
                pt_high_low_convergence_1y NUMERIC,
                analyst_count_stability    NUMERIC
            )
    STABLE PARALLEL SAFE
AS
$$
SELECT "ISIN",
       CASE
           WHEN "Price Target (1Y Ago)" > 0 AND "Last Price" >= "Price Target (1Y Ago)" THEN 1.0
           WHEN "Price Target (1Y Ago)" > 0 THEN
               public.safe_divide("Last Price", "Price Target (1Y Ago)")
           END                                                                               AS pt_achievement_1y,
       ABS("Last Price" - "Price Target (1Y Ago)") / NULLIF(ABS("Price Target (1Y Ago)"), 0) AS pt_accuracy_1y,
       ("Price Target (1Y Ago)" - "Last Price") / NULLIF(ABS("Price Target (1Y Ago)"), 0)    AS pt_optimism_bias,
       CASE
           WHEN "Last Price" BETWEEN "Price Target - Low (1Y Ago)" AND "Price Target - High (1Y Ago)"
               THEN 1.0
           ELSE 0.0
           END                                                                               AS pt_range_hit_rate,
       ("Price Target" - "Price Target - Median") /
       NULLIF("Price Target - Median", 0)                                                    AS pt_median_vs_mean_spread,
       (("Price Target - High" - "Price Target - Low") / NULLIF("Price Target - Median", 0)) -
       (("Price Target - High (1Y Ago)" - "Price Target - Low (1Y Ago)") /
        NULLIF("Price Target - Median (1Y Ago)", 0))                                         AS pt_high_low_convergence_1y,
       public.safe_divide("Price Target - #",
                          ("Price Target - # (1Y Ago)" + "Price Target - # (6M Ago)" + "Price Target - # (3M Ago)") /
                          3.0)                                                               AS analyst_count_stability
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION: DIVIDEND HISTORY FEATURES (NEW)
-- =============================================================================
CREATE OR REPLACE FUNCTION calc_dividend_history_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                     TEXT,
                div_yield_2fy            NUMERIC,
                div_yield_3fy            NUMERIC,
                div_yield_4fy            NUMERIC,
                div_yield_5fy            NUMERIC,
                div_yield_trend_3y       NUMERIC,
                div_yield_volatility     NUMERIC,
                div_yield_declining_flag INTEGER,
                div_yield_mean_5y        NUMERIC,
                div_yield_vs_5y_mean     NUMERIC
            )
    STABLE PARALLEL SAFE
AS
$$
SELECT "ISIN",
       "Div Yield (-2FYInd)",
       "Div Yield (-3FYInd)",
       "Div Yield (-4FYInd)",
       "Div Yield (-5FYInd)",
       ("Div Yield (Ind)" - "Div Yield (-3FYInd)") / 3.0 AS div_yield_trend_3y,
       GREATEST("Div Yield (Ind)", "Div Yield (-1FYInd)", "Div Yield (-2FYInd)",
                "Div Yield (-3FYInd)", "Div Yield (-4FYInd)") -
       LEAST("Div Yield (Ind)", "Div Yield (-1FYInd)", "Div Yield (-2FYInd)",
             "Div Yield (-3FYInd)",
             "Div Yield (-4FYInd)")                      AS div_yield_volatility,
       CASE
           WHEN "Div Yield (Ind)" < "Div Yield (-1FYInd)"
               AND "Div Yield (-1FYInd)" < "Div Yield (-2FYInd)"
               AND "Div Yield (-2FYInd)" < "Div Yield (-3FYInd)" THEN 1
           ELSE 0 END                                    AS div_yield_declining_flag,
       (COALESCE("Div Yield (Ind)", 0) + COALESCE("Div Yield (-1FYInd)", 0) +
        COALESCE("Div Yield (-2FYInd)", 0) + COALESCE("Div Yield (-3FYInd)", 0) +
        COALESCE("Div Yield (-4FYInd)", 0)) /
       NULLIF((CASE WHEN "Div Yield (Ind)" IS NOT NULL THEN 1 ELSE 0 END +
               CASE WHEN "Div Yield (-1FYInd)" IS NOT NULL THEN 1 ELSE 0 END +
               CASE WHEN "Div Yield (-2FYInd)" IS NOT NULL THEN 1 ELSE 0 END +
               CASE WHEN "Div Yield (-3FYInd)" IS NOT NULL THEN 1 ELSE 0 END +
               CASE WHEN "Div Yield (-4FYInd)" IS NOT NULL THEN 1 ELSE 0 END)::NUMERIC,
              0)                                         AS div_yield_mean_5y,
       public.calc_change_ratio("Div Yield (Ind)",
                                (COALESCE("Div Yield (Ind)", 0) + COALESCE("Div Yield (-1FYInd)", 0) +
                                 COALESCE("Div Yield (-2FYInd)", 0) + COALESCE("Div Yield (-3FYInd)", 0) +
                                 COALESCE("Div Yield (-4FYInd)", 0)) /
                                NULLIF((CASE WHEN "Div Yield (Ind)" IS NOT NULL THEN 1 ELSE 0 END +
                                        CASE WHEN "Div Yield (-1FYInd)" IS NOT NULL THEN 1 ELSE 0 END +
                                        CASE WHEN "Div Yield (-2FYInd)" IS NOT NULL THEN 1 ELSE 0 END +
                                        CASE WHEN "Div Yield (-3FYInd)" IS NOT NULL THEN 1 ELSE 0 END +
                                        CASE WHEN "Div Yield (-4FYInd)" IS NOT NULL THEN 1 ELSE 0 END)::NUMERIC,
                                       0))               AS div_yield_vs_5y_mean
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION: SIZE & LIQUIDITY FEATURES (NEW)
-- =============================================================================
CREATE OR REPLACE FUNCTION calc_size_liquidity_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                 TEXT,
                market_cap           NUMERIC,
                market_cap_country_r NUMERIC,
                log_market_cap       NUMERIC,
                volume_shrs          NUMERIC,
                relative_volume      NUMERIC,
                shares_outstanding   NUMERIC,
                daily_turnover_ratio NUMERIC,
                size_class           TEXT,
                style_class          TEXT,
                liquidity_score      NUMERIC
            )
    STABLE PARALLEL SAFE
AS
$$
SELECT "ISIN",
       "Market Cap",
       "Market Cap (Country R)",
       LN(GREATEST("Market Cap", 1))                                                           AS log_market_cap,
       "Volume (Shrs)",
       "Rel. Volume",
       "Shrs Out",
       public.safe_divide("Volume (Shrs)", "Shrs Out")                                         AS daily_turnover_ratio,
       "Size Class",
       "Style Class",
       "Volume (Shrs)" * COALESCE("Rel. Volume", 1) / NULLIF(LN(GREATEST("Market Cap", 1)), 0) AS liquidity_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION: INVESTMENT INCOME TEMPORAL FEATURES (NEW)
-- =============================================================================
CREATE OR REPLACE FUNCTION calc_investment_income_temporal(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                         TEXT,
                inv_income_ltm               NUMERIC,
                inv_income_fq                NUMERIC,
                inv_income_fy                NUMERIC,
                inv_income_qoq_growth        NUMERIC,
                inv_income_yoy_growth        NUMERIC,
                inv_income_to_revenue        NUMERIC,
                inv_income_trend_3y          NUMERIC,
                inv_income_positive_quarters INTEGER,
                financial_company_proxy      INTEGER
            )
    STABLE PARALLEL SAFE
AS
$$
SELECT "ISIN",
       "Interest And Investment Income (LTM)",
       "Interest And Investment Income (FQ)",
       "Interest And Investment Income (FY)",
       public.calc_change_ratio("Interest And Investment Income (FQ)",
                                "Interest And Investment Income (-1FQFQ)")                AS inv_income_qoq_growth,
       public.calc_change_ratio("Interest And Investment Income (FY)",
                                "Interest And Investment Income (-1FY)")                  AS inv_income_yoy_growth,
       public.safe_divide("Interest And Investment Income (LTM)", "Total Revenues (LTM)") AS inv_income_to_revenue,
       CASE
           WHEN "Interest And Investment Income (-3FY)" > 0 AND "Interest And Investment Income (FY)" > 0
               THEN (POWER(public.safe_divide("Interest And Investment Income (FY)",
                                              "Interest And Investment Income (-3FY)"), 1.0 / 3.0) - 1) * 100
           END                                                                            AS inv_income_trend_3y,
       (CASE WHEN "Interest And Investment Income (FQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Interest And Investment Income (-1FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Interest And Investment Income (-2FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Interest And Investment Income (-3FQFQ)" > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN "Interest And Investment Income (-4FQFQ)" > 0 THEN 1
            ELSE 0 END)::INTEGER                                                          AS inv_income_positive_quarters,
       CASE
           WHEN public.safe_divide("Interest And Investment Income (LTM)", "Total Revenues (LTM)") > 0.2
               THEN 1
           ELSE 0 END                                                                     AS financial_company_proxy
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION: TAX RATE FEATURES (NEW - Enhancement 4)
-- =============================================================================
CREATE OR REPLACE FUNCTION calc_tax_rate_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                   TEXT,
                effective_tax_rate_ltm NUMERIC,
                effective_tax_rate_fy  NUMERIC,
                tax_rate_yoy_change    NUMERIC,
                tax_rate_qoq_change    NUMERIC,
                tax_rate_stability     NUMERIC,
                low_tax_flag           INTEGER,
                tax_rate_trend_4q      NUMERIC
            )
    STABLE PARALLEL SAFE
AS
$$
SELECT "ISIN",
       "Effective Tax Rate - (Ratio) (LTM)",
       "Effective Tax Rate - (Ratio) (FY)",
       "Effective Tax Rate - (Ratio) (FY)" - "Effective Tax Rate - (Ratio) (-1FY)" AS tax_rate_yoy_change,
       "Effective Tax Rate - (Ratio) (FQ)" -
       "Effective Tax Rate - (Ratio) (-1FQFQ)"                                     AS tax_rate_qoq_change,
       -- Stability: range across available quarterly periods (lower = more stable)
       GREATEST(
               COALESCE("Effective Tax Rate - (Ratio) (FQ)", 0),
               COALESCE("Effective Tax Rate - (Ratio) (-1FQFQ)", 0),
               COALESCE("Effective Tax Rate - (Ratio) (-2FQFQ)", 0),
               COALESCE("Effective Tax Rate - (Ratio) (-3FQFQ)", 0)
       ) - LEAST(
               COALESCE("Effective Tax Rate - (Ratio) (FQ)", 0),
               COALESCE("Effective Tax Rate - (Ratio) (-1FQFQ)", 0),
               COALESCE("Effective Tax Rate - (Ratio) (-2FQFQ)", 0),
               COALESCE("Effective Tax Rate - (Ratio) (-3FQFQ)", 0)
           )                                                                       AS tax_rate_stability,
       CASE WHEN "Effective Tax Rate - (Ratio) (LTM)" < 0.10 THEN 1 ELSE 0 END     AS low_tax_flag,
       -- Trend across 4 quarters (FQ vs avg of prior 3)
       "Effective Tax Rate - (Ratio) (FQ)" -
       (COALESCE("Effective Tax Rate - (Ratio) (-1FQFQ)", 0) +
        COALESCE("Effective Tax Rate - (Ratio) (-2FQFQ)", 0) +
        COALESCE("Effective Tax Rate - (Ratio) (-3FQFQ)", 0)) /
       NULLIF((CASE WHEN "Effective Tax Rate - (Ratio) (-1FQFQ)" IS NOT NULL THEN 1 ELSE 0 END +
               CASE WHEN "Effective Tax Rate - (Ratio) (-2FQFQ)" IS NOT NULL THEN 1 ELSE 0 END +
               CASE WHEN "Effective Tax Rate - (Ratio) (-3FQFQ)" IS NOT NULL THEN 1 ELSE 0 END)::NUMERIC,
              0)                                                                   AS tax_rate_trend_4q
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION: OPEX TEMPORAL FEATURES (NEW - Enhancement 5)
-- =============================================================================
CREATE OR REPLACE FUNCTION calc_opex_temporal_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                     TEXT,
                opex_fq                  NUMERIC,
                opex_ltm                 NUMERIC,
                opex_fy                  NUMERIC,
                opex_qoq_growth          NUMERIC,
                opex_yoy_growth          NUMERIC,
                opex_vs_revenue_trend    NUMERIC,
                sga_qoq_growth           NUMERIC,
                sga_yoy_growth           NUMERIC,
                operating_leverage_score NUMERIC
            )
    STABLE PARALLEL SAFE
AS
$$
SELECT "ISIN",
       "Total Operating Expenses (FQ)",
       "Total Operating Expenses (LTM)",
       "Total Operating Expenses (FY)",
       public.calc_change_ratio("Total Operating Expenses (FQ)",
                                "Total Operating Expenses (-1FQFQ)")             AS opex_qoq_growth,
       public.calc_change_ratio("Total Operating Expenses (FY)",
                                "Total Operating Expenses (-1FY)")               AS opex_yoy_growth,
       -- Change in opex/revenue ratio (FY vs -1FY)
       (public.safe_divide("Total Operating Expenses (FY)"::NUMERIC, "Total Revenues (FY)"::NUMERIC) -
        public.safe_divide("Total Operating Expenses (-1FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC)) *
       100                                                                       AS opex_vs_revenue_trend,
       public.calc_change_ratio("Selling General & Admin Expenses/Total (FQ)",
                                "Selling General & Admin Expenses/Total (-1FY)") AS sga_qoq_growth,
       public.calc_change_ratio("Selling General & Admin Expenses/Total (FY)",
                                "Selling General & Admin Expenses/Total (-1FY)") AS sga_yoy_growth,
       -- Operating leverage: revenue growth minus opex growth
       public.calc_change_ratio("Total Revenues (FY)"::NUMERIC, "Total Revenues (-1FY)"::NUMERIC) -
       public.calc_change_ratio("Total Operating Expenses (FY)"::NUMERIC,
                                "Total Operating Expenses (-1FY)"::NUMERIC)      AS operating_leverage_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION: FCF ESTIMATE FEATURES (NEW - Enhancement 9)
-- =============================================================================
CREATE OR REPLACE FUNCTION calc_fcf_estimate_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin             TEXT,
                fcf_est_avg_fy1e NUMERIC,
                fcf_est_avg_fy2e NUMERIC,
                fcf_est_avg_fy3e NUMERIC,
                fcf_est_avg_fy4e NUMERIC,
                fcf_est_avg_fy5e NUMERIC,
                fcf_est_cagr_5y  NUMERIC,
                fcf_est_trend    NUMERIC
            )
    STABLE PARALLEL SAFE
AS
$$
SELECT "ISIN",
       "FCF - Est Avg (FY1E)",
       "FCF - Est Avg (FY2E)",
       "FCF - Est Avg (FY3E)",
       "FCF - Est Avg (FY4E)",
       "FCF - Est Avg (FY5E)",
       -- Implied 5Y CAGR from FY1E to FY5E
       CASE
           WHEN "FCF - Est Avg (FY1E)" > 0 AND "FCF - Est Avg (FY5E)" > 0
               THEN (POWER(public.safe_divide("FCF - Est Avg (FY5E)",
                                              "FCF - Est Avg (FY1E)"), 0.25) - 1) * 100
           END                                                                  AS fcf_est_cagr_5y,
       -- Linear trend: (FY5E - FY1E) / FY1E
       public.calc_change_ratio("FCF - Est Avg (FY5E)", "FCF - Est Avg (FY1E)") AS fcf_est_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION: ASSET SALE FEATURES (NEW - Enhancement 8)
-- =============================================================================
CREATE OR REPLACE FUNCTION calc_asset_sale_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                            TEXT,
                gain_loss_on_sale_of_assets_ltm NUMERIC,
                asset_sale_frequency            INTEGER,
                asset_sale_trend                NUMERIC
            )
    STABLE PARALLEL SAFE
AS
$$
SELECT "ISIN",
       "Gain (Loss) On Sale Of Assets (LTM)",
       -- Count of non-zero periods across available quarters and years
       (CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (FQ)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-1FQFQ)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-2FQFQ)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-3FQFQ)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-4FQFQ)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (FY)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-1FY)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-2FY)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-3FY)", 0)) > 0 THEN 1 ELSE 0 END +
        CASE
            WHEN ABS(COALESCE("Gain (Loss) On Sale Of Assets (-4FY)", 0)) > 0 THEN 1
            ELSE 0 END)::INTEGER                                     AS asset_sale_frequency,
       -- Trend: FQ vs average of prior quarters
       "Gain (Loss) On Sale Of Assets (FQ)" -
       (COALESCE("Gain (Loss) On Sale Of Assets (-1FQFQ)", 0) +
        COALESCE("Gain (Loss) On Sale Of Assets (-2FQFQ)", 0) +
        COALESCE("Gain (Loss) On Sale Of Assets (-3FQFQ)", 0) +
        COALESCE("Gain (Loss) On Sale Of Assets (-4FQFQ)", 0)) / 4.0 AS asset_sale_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- SECTION: SHARE DILUTION TRACKING (NEW - Enhancement 12)
-- =============================================================================
CREATE OR REPLACE FUNCTION calc_share_dilution_tracking(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin                  TEXT,
                shrs_out_1fy          NUMERIC,
                shares_yoy_change_pct NUMERIC,
                net_buyback_flag      INTEGER
            )
    STABLE PARALLEL SAFE
AS
$$
SELECT "ISIN",
       "Shrs Out (-1FY)",
       public.calc_change_ratio("Shrs Out", "Shrs Out (-1FY)")    AS shares_yoy_change_pct,
       CASE WHEN "Shrs Out" < "Shrs Out (-1FY)" THEN 1 ELSE 0 END AS net_buyback_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- -----------------------------------------------------------------------------
-- All Enhanced Features (Aggregation Function)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION calc_all_enhanced_features(p_isin TEXT DEFAULT NULL)
    RETURNS TABLE
            (
                isin           TEXT,
                feature_count  INTEGER,
                reference_date TIMESTAMP
            )
    STABLE
    PARALLEL SAFE
AS
$$
SELECT "ISIN"            AS isin,
       (SELECT COUNT(*)::INTEGER as count
        FROM information_schema.routines
        WHERE routine_name LIKE 'calc_%'
          AND routine_schema = 'public')
                         AS feature_count,
       CURRENT_TIMESTAMP AS reference_date
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$ LANGUAGE SQL;

-- =============================================================================
-- FEATURE VIEWS - REFACTORED TO USE vw_identifier_columns DIRECTLY
-- =============================================================================
-- All identifier columns (ISIN, Ticker, Name, Region, Country, Trading Country,
-- Exchange, Sector, Industry, CATEGORICAL columns, DATE columns) are now
-- inherited from vw_identifier_columns via id.* instead of hardcoded selection.
-- =============================================================================

-- =============================================================================
-- 1. vw_features_valuation_ratios (REFACTORED)
-- Source functions: calc_valuation_features, calc_tangible_book_features,
--                   calc_valuation_timeseries_features, calc_extended_valuation_timeseries
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_valuation_ratios AS
SELECT id.*,

       -- =========================================================================
       -- VALUATION FEATURES (calc_valuation_features)
       -- Source columns: P/E (LTM), P/B (LTM), EV/EBITDA (LTM), EV/Sales (LTM),
       --                 Div Yield (LTM), Net EPS - Basic (FY), Net EPS - Basic (-3FY)
       -- =========================================================================
       vf.p_e_ratio,
       vf.p_b_ratio,
       vf.ev_ebitda_ratio,
       vf.ev_sales_ratio,
       vf.dividend_yield,
       vf.peg_ratio,

       -- =========================================================================
       -- VALUATION TIMESERIES (calc_valuation_timeseries_features)
       -- Source columns: EV/Sales (LTM), EV/Sales (-1FYLTM), EV/EBITDA (LTM),
       --                 EV/EBITDA (-1FYLTM), P/E (LTM), P/E (-1FYLTM), etc.
       -- =========================================================================
       vts.ev_sales_trend_1y,
       vts.ev_ebitda_momentum,
       vts.p_e_momentum_yoy,
       vts.p_e_momentum_qoq,
       vts.ev_sales_vs_3y_avg,
       vts.ev_ebitda_vs_3y_avg,
       vts.p_e_vs_3y_avg,
       vts.ev_sales_forward_discount,
       vts.ev_ebitda_forward_discount,
       vts.p_e_forward_discount,
       vts.p_b_vs_5y_avg,

       -- =========================================================================
       -- EXTENDED VALUATION TIMESERIES (calc_extended_valuation_timeseries)
       -- Source columns: EV/Sales (-1FQLTM), P/E (5YAVGLTM), P/B (-1FY), P/E (EST FY1)
       -- =========================================================================
       evt.ev_sales_qoq_1q,
       evt.ev_sales_qoq_2q,
       evt.ev_sales_qoq_3q,
       evt.ev_sales_qoq_4q,
       evt.p_e_vs_5y_avg,
       evt.p_e_percentile_proxy,
       evt.valuation_mean_reversion,
       evt.ev_ebitda_qoq_trend,
       evt.p_b_momentum_yoy,
       evt.valuation_compression,
       evt.forward_pe_premium,

       -- =========================================================================
       -- TANGIBLE BOOK FEATURES (calc_tangible_book_features)
       -- Source columns: TBV (FY), TBV (LTM), P/TBV (LTM), Total Equity (LTM),
       --                 Goodwill (LTM), Gross Intangible Assets (LTM), Total Assets (LTM)
       -- =========================================================================
       tb.tangible_book_value_fy,
       tb.tangible_book_value_ltm,
       tb.tangible_book_per_share,
       tb.price_to_tangible_book,
       tb.tangible_equity_ratio,
       tb.intangibles_to_equity,
       tb.goodwill_to_equity,
       tb.tangible_asset_quality,
       tb.tbv_yoy_growth,
       tb.tbv_vs_calculated

FROM vw_identifier_columns id
         LEFT JOIN calc_valuation_features() vf USING (isin)
         LEFT JOIN calc_valuation_timeseries_features() vts USING (isin)
         LEFT JOIN calc_extended_valuation_timeseries() evt USING (isin)
         LEFT JOIN calc_tangible_book_features() tb USING (isin);

COMMENT ON VIEW vw_features_valuation_ratios IS
    'Valuation metrics including P/E, P/B, EV/EBITDA, tangible book value, and timeseries analysis.
    Identifier columns inherited from vw_identifier_columns.
    Source functions: calc_valuation_features, calc_valuation_timeseries_features,
    calc_extended_valuation_timeseries, calc_tangible_book_features';


-- =============================================================================
-- 2. vw_features_momentum (NEW)
-- Source functions: calc_momentum_features, calc_long_term_momentum_features
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_momentum AS
SELECT id.*,

       -- =========================================================================
       -- MOMENTUM FEATURES (calc_momentum_features)
       -- Source columns: Last Price, Price (1M/3M/6M/1Y/5D Ago), EMA (20D/50D/250D),
       --                 52W High/Adj, 52W Low/Adj, Beta (1Y/5Y), Volatility (1M/1Y)
       -- =========================================================================
       mf.price_momentum_1m,
       mf.price_momentum_3m,
       mf.price_momentum_6m,
       mf.price_momentum_1y,
       mf.price_momentum_5d,
       mf.ema_crossover_20_50,
       mf.ema_crossover_50_250,
       mf.price_vs_ema_20d,
       mf.price_vs_ema_250d,
       mf.pct_off_52w_high,
       mf.pct_above_52w_low,
       mf.range_52w_position,
       mf.beta_momentum,
       mf.volatility_regime,

       -- =========================================================================
       -- LONG TERM MOMENTUM FEATURES (calc_long_term_momentum_features)
       -- Source columns: Last Price, Price (1Y/3Y/5Y Ago), EMA (250D), 52W High/Adj
       -- =========================================================================
       ltm.price_momentum_1y AS price_momentum_1y_long,
       ltm.price_momentum_3y,
       ltm.price_momentum_5y,
       ltm.long_term_trend_score,
       ltm.price_vs_ema_250d AS price_vs_ema_250d_long,
       ltm.multi_year_high_flag,
       ltm.secular_trend_flag

FROM vw_identifier_columns id
         LEFT JOIN calc_momentum_features() mf USING (isin)
         LEFT JOIN calc_long_term_momentum_features() ltm USING (isin);

COMMENT ON VIEW vw_features_momentum IS
    'Price momentum and trend indicators across multiple timeframes.
    Source functions: calc_momentum_features, calc_long_term_momentum_features';


-- =============================================================================
-- 3. vw_features_technical_analysis (REFACTORED)
-- Source functions: calc_technical_analysis_features
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_technical_analysis AS
SELECT id.*,

       -- =========================================================================
       -- TECHNICAL ANALYSIS FEATURES (calc_technical_analysis_features)
       -- Source columns: EMA (20D/50D/100D/250D), Last Price, 52W High/Adj,
       --                 52W Low/Adj, Rel. Volume, Price Chg. % (1M),
       --                 Volatility (1M/3M/6M/1Y)
       -- =========================================================================
       ta.ema_slope_20d,
       ta.ema_trend_consistency,
       ta.price_vs_ema_100d,
       ta.near_52w_high_flag,
       ta.near_52w_low_flag,
       ta.volume_momentum_score,
       ta.breakout_signal,
       ta.high_volume_flag,
       ta.low_volume_flag,
       ta.volatility_compression,
       ta.volatility_term_structure

FROM vw_identifier_columns id
         LEFT JOIN calc_technical_analysis_features() ta USING (isin);

COMMENT ON VIEW vw_features_technical_analysis IS
    'Technical analysis indicators including EMA trends, volume signals, and volatility patterns.
    Source function: calc_technical_analysis_features';


-- =============================================================================
-- 4. vw_features_profitability (REFACTORED)
-- Source functions: calc_profitability_features, calc_margin_trends,
--                   calc_ebit_ebitda_comprehensive, calc_gross_profit_temporal
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_profitability AS
SELECT id.*,

       -- =========================================================================
       -- PROFITABILITY FEATURES (calc_profitability_features)
       -- Source columns: Return On Equity % (LTM), Return on Assets (ROA) % (LTM),
       --                 Gross Profit Margin % (LTM), Operating Income (LTM),
       --                 Total Revenues (LTM), Net Income Margin % (LTM), EBITDA (LTM),
       --                 Net Income - (IS) (LTM), Total Equity (LTM), Total Debt (LTM),
       --                 R&D Expenses (LTM), Total Assets (LTM)
       -- =========================================================================
       pf.roe,
       pf.roa,
       pf.gross_margin_pct,
       pf.operating_margin_pct,
       pf.net_margin_pct,
       pf.ebitda_margin_pct,
       pf.roic,
       pf.rnd_intensity,
       pf.equity_multiplier,

       -- =========================================================================
       -- MARGIN TRENDS (calc_margin_trends)
       -- Source columns: Gross Profit Margin % (LTM/FY), Operating Income (LTM/FY),
       --                 Total Revenues (LTM/FY), Net Income Margin % (LTM/FY),
       --                 EBITDA (LTM/FY)
       -- =========================================================================
       mt.gross_margin_trend_yoy,
       mt.operating_margin_trend,
       mt.net_margin_trend_yoy,
       mt.ebitda_margin_trend,
       mt.margin_expansion_flag,
       mt.margin_stability_score,

       -- =========================================================================
       -- EBIT/EBITDA COMPREHENSIVE (calc_ebit_ebitda_comprehensive)
       -- Source columns: EBIT (FQ/LTM/FY/-1FY/-2FY/-3FY/-4FY/-1FQFQ/-2FQFQ/-3FQFQ/-4FQFQ/5YAVGFQ/5YAVGLTM),
       --                 EBITDA (same periods), EBIT/Adj., EBITDA/Adj., Total Revenues (LTM)
       -- =========================================================================
       eec.ebit_fq,
       eec.ebit_ltm,
       eec.ebit_fy,
       eec.ebit_1fy,
       eec.ebit_2fy,
       eec.ebit_3fy,
       eec.ebit_4fy,
       eec.ebitda_fq,
       eec.ebitda_ltm,
       eec.ebitda_fy,
       eec.ebitda_1fy,
       eec.ebitda_2fy,
       eec.ebitda_3fy,
       eec.ebitda_4fy,
       eec.ebit_5yavgfq,
       eec.ebit_5yavgltm,
       eec.ebitda_5yavgfq,
       eec.ebitda_5yavgltm,
       eec.ebit_adj_fq,
       eec.ebit_adj_ltm,
       eec.ebit_adj_fy,
       eec.ebitda_adj_fq,
       eec.ebitda_adj_ltm,
       eec.ebitda_adj_fy,
       eec.ebit_growth_yoy,
       eec.ebitda_growth_yoy,
       eec.ebit_margin_ltm,
       eec.ebitda_margin_ltm,
       eec.ebit_positive_years,
       eec.ebitda_positive_years,
       eec.ebit_qoq_growth,
       eec.ebitda_qoq_growth,
       eec.ebit_cagr_3y,
       eec.ebitda_cagr_3y,
       eec.ebit_vs_5y_avg,
       eec.ebitda_vs_5y_avg,

       -- =========================================================================
       -- GROSS PROFIT TEMPORAL (calc_gross_profit_temporal)
       -- Source columns: Gross Profit (FQ/FY/LTM/-1FQFQ/-2FQFQ/-3FQFQ/-4FQFQ/-1FY/-2FY/-3FY/-4FY),
       --                 Total Revenues (FQ/5YAVGFQ), Gross Profit Margin % (LTM/FY)
       -- =========================================================================
       gpt.gp_fq,
       gpt.gp_fy,
       gpt.gp_ltm,
       gpt.gp_1fqfq,
       gpt.gp_2fqfq,
       gpt.gp_3fqfq,
       gpt.gp_4fqfq,
       gpt.gp_1fy,
       gpt.gp_2fy,
       gpt.gp_3fy,
       gpt.gp_4fy,
       gpt.gp_qoq_growth,
       gpt.gp_yoy_growth,
       gpt.gp_margin_fq,
       gpt.gp_margin_trend,
       gpt.gp_positive_quarters,
       gpt.gp_margin_expansion

FROM vw_identifier_columns id
         LEFT JOIN calc_profitability_features() pf USING (isin)
         LEFT JOIN calc_margin_trends() mt USING (isin)
         LEFT JOIN calc_ebit_ebitda_comprehensive() eec USING (isin)
         LEFT JOIN calc_gross_profit_temporal() gpt USING (isin);

COMMENT ON VIEW vw_features_profitability IS
    'Profitability metrics including ROE, ROA, margins, EBIT/EBITDA comprehensive analysis.
    Source functions: calc_profitability_features, calc_margin_trends,
    calc_ebit_ebitda_comprehensive, calc_gross_profit_temporal';


-- =============================================================================
-- 5. vw_features_earnings (NEW)
-- Source functions: calc_earnings_features, calc_eps_trajectory_features,
--                   calc_eps_comprehensive, calc_eps_continuing_features,
--                   calc_gaap_adjusted_analytics, calc_gaap_revision_features
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_earnings AS
SELECT id.*,

       -- =========================================================================
       -- EARNINGS FEATURES (calc_earnings_features)
       -- Source columns: EPS/Adj. (LTM), EPS Norm - Est Avg (FY1E), Total Revenues (LTM),
       --                 Revenues - Est Avg (FY1E), Net EPS - Basic (LTM/FQ/FY/-1FY/-4FQFQ),
       --                 EPS GAAP - Est Avg (FY1E), EBITDA/Adj. (LTM), EBITDA (LTM)
       -- =========================================================================
       ef.eps_surprise_pct,
       ef.revenue_surprise_pct,
       ef.eps_adjustment_ratio,
       ef.gaap_adj_eps_gap_pct,
       ef.ebitda_adjustment_ratio,
       ef.eps_quarterly_trend,
       ef.eps_yoy_growth,

       -- =========================================================================
       -- EPS TRAJECTORY FEATURES (calc_eps_trajectory_features)
       -- Source columns: Net EPS - Basic (FQ/-1FQFQ/-2FQFQ/-3FQFQ/-4FQFQ/FY/-1FY/-2FY/-3FY/-4FY/-5FY)
       -- =========================================================================
       etf.eps_qoq_growth,
       etf.eps_yoy_quarterly,
       etf.eps_positive_streak,
       etf.eps_cagr_3y,
       etf.eps_cagr_5y,
       etf.eps_growth_accel,
       etf.eps_vs_5y_avg,
       etf.eps_improvement_count,
       etf.eps_trajectory_score,
       etf.eps_stability,

       -- =========================================================================
       -- EPS COMPREHENSIVE (calc_eps_comprehensive)
       -- Source columns: Net EPS - Basic (FQ/LTM/FY/-1FY/-2FY/-3FY/-4FY/-5FY),
       --                 EPS/Adj. (LTM), EPS Norm - Est Avg (FY1E)
       -- =========================================================================
       ec.eps_basic_fq,
       ec.eps_basic_ltm,
       ec.eps_basic_fy,
       ec.eps_adj_ltm,
       ec.eps_norm_est_fy1e,
       ec.eps_growth_yoy       AS eps_growth_yoy_comp,
       ec.eps_cagr_3y          AS eps_cagr_3y_comp,
       ec.eps_adjustment_ratio AS eps_adjustment_ratio_comp,
       ec.eps_positive_years,
       ec.eps_trajectory_score AS eps_trajectory_score_comp,

       -- =========================================================================
       -- EPS CONTINUING FEATURES (calc_eps_continuing_features)
       -- Source columns: Basic EPS - Cont (LTM/FQ/FY/-1FQFQ/-2FQFQ/-3FQFQ/-4FQFQ/-1FY/-2FY/-3FY/-4FY),
       --                 Net EPS - Basic (LTM)
       -- =========================================================================
       ecf.eps_cont_ltm,
       ecf.eps_cont_fq,
       ecf.eps_cont_fy,
       ecf.eps_cont_1fqfq,
       ecf.eps_cont_2fqfq,
       ecf.eps_cont_3fqfq,
       ecf.eps_cont_4fqfq,
       ecf.eps_cont_1fy,
       ecf.eps_cont_2fy,
       ecf.eps_cont_3fy,
       ecf.eps_cont_4fy,
       ecf.eps_cont_qoq_growth,
       ecf.eps_cont_yoy_growth,
       ecf.eps_cont_cagr_3y,
       ecf.eps_cont_vs_total_eps,
       ecf.eps_cont_positive_streak,
       ecf.eps_cont_trajectory_score,
       ecf.discontinued_ops_impact,
       ecf.core_earnings_stability,

       -- =========================================================================
       -- GAAP ADJUSTED ANALYTICS (calc_gaap_adjusted_analytics)
       -- Source columns: EPS/Adj., Net EPS - Basic, Net Income/Adj., Net Income - (IS),
       --                 EBITDA/Adj., EBITDA, EBIT/Adj., EBIT (multiple periods),
       --                 EPS Norm - Est Avg (FY1E), EPS GAAP - Est Avg (FY1E)
       -- =========================================================================
       gaa.eps_adjustment_spread_ltm,
       gaa.eps_adjustment_spread_fy,
       gaa.eps_adjustment_pct,
       gaa.net_income_adjustment_ratio_ltm,
       gaa.net_income_adjustment_ratio_fy,
       gaa.net_income_adjustment_pct,
       gaa.ebitda_adjustment_pct_ltm,
       gaa.ebitda_adjustment_pct_fy,
       gaa.ebit_adjustment_pct_ltm,
       gaa.ebit_adjustment_pct_fy,
       gaa.earnings_quality_score,
       gaa.earnings_quality_warning,
       gaa.forward_eps_gaap_adj_spread,

       -- =========================================================================
       -- GAAP REVISION FEATURES (calc_gaap_revision_features)
       -- Source columns: EPS GAAP Est Avg Rev % (FY1E - 1M/3M/6M/1Y),
       --                 EPS Est Avg Rev % (FY1E - 1M/3M)
       -- =========================================================================
       grf.gaap_revision_momentum,
       grf.gaap_revision_1m,
       grf.gaap_revision_3m,
       grf.gaap_revision_6m,
       grf.gaap_revision_1y,
       grf.gaap_vs_norm_revision_spread,
       grf.gaap_revision_acceleration,
       grf.gaap_positive_revision_flag,
       grf.revision_quality_divergence

FROM vw_identifier_columns id
         LEFT JOIN calc_earnings_features() ef USING (isin)
         LEFT JOIN calc_eps_trajectory_features() etf USING (isin)
         LEFT JOIN calc_eps_comprehensive() ec USING (isin)
         LEFT JOIN calc_eps_continuing_features() ecf USING (isin)
         LEFT JOIN calc_gaap_adjusted_analytics() gaa USING (isin)
         LEFT JOIN calc_gaap_revision_features() grf USING (isin);

COMMENT ON VIEW vw_features_earnings IS
    'Earnings metrics including EPS analysis, GAAP adjustments, and revision trends.
    Source functions: calc_earnings_features, calc_eps_trajectory_features, calc_eps_comprehensive,
    calc_eps_continuing_features, calc_gaap_adjusted_analytics, calc_gaap_revision_features';


-- =============================================================================
-- 6. vw_features_growth (REFACTORED)
-- Source functions: calc_growth_features, calc_revenue_forecast_features,
--                   calc_revenue_quarterly_features, calc_total_revenues_temporal,
--                   calc_revenue_estimate_consensus
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_growth AS
SELECT id.*,

       -- =========================================================================
       -- GROWTH FEATURES (calc_growth_features)
       -- Source columns: Total Revenues (FY/-1FY/LTM/5YAVGLTM), EBITDA (FY/-1FY),
       --                 Operating Income (LTM/FY), FCF (LTM/FY),
       --                 Total Revenues/CAGR (5Y FY), Revenues - Est YoY % (FY1E)
       -- =========================================================================
       gf.revenue_growth_yoy,
       gf.ebitda_growth_yoy,
       gf.operating_income_growth,
       gf.fcf_growth,
       gf.revenue_cagr_5y,
       gf.forward_revenue_growth,
       gf.revenue_vs_5y_avg,

       -- =========================================================================
       -- REVENUE FORECAST FEATURES (calc_revenue_forecast_features)
       -- Source columns: Revenues - Est Avg/Med (FY1E/NTM), Total Revenues (LTM/FY),
       --                 EBITDA - Est Avg (FY1E), Enterprise Value, EPS Norm - Est # (FY1E),
       --                 EBIT - Est Med (FY1E/NTM), Total Revenues/CAGR (5Y FY)
       -- =========================================================================
       rff.revenue_est_spread,
       rff.revenue_beat_potential,
       rff.revenue_est_revision_trend,
       rff.ebitda_est_vs_actual,
       rff.forward_revenue_multiple,
       rff.revenue_estimate_count,
       rff.revenue_guidance_gap,
       rff.consensus_revenue_growth,
       rff.ebit_estimate_spread,
       rff.forward_ebitda_margin,
       rff.revenue_acceleration,
       rff.estimate_confidence_score,

       -- =========================================================================
       -- REVENUE ESTIMATE CONSENSUS (calc_revenue_estimate_consensus)
       -- Source columns: Revenues - Est Avg (FY1E/NTM), Revenues - Est Med (FY1E/NTM),
       --                 Revenues - Est YoY % (FY1E), Total Revenues (LTM)
       -- =========================================================================
       rec.revenue_est_avg_fy1e,
       rec.revenue_est_med_fy1e,
       rec.revenue_est_avg_ntm,
       rec.revenue_est_med_ntm,
       rec.revenue_avg_med_diff_pct,
       rec.revenue_consensus_strength,
       rec.revenue_revision_trend AS revenue_revision_trend_rec,
       rec.revenue_vs_current,

       -- =========================================================================
       -- REVENUE QUARTERLY FEATURES (calc_revenue_quarterly_features)
       -- Source columns: Total Revenues (FQ/FY/LTM/5YAVGLTM/-1FQFQ/-2FQFQ/-3FQFQ/-4FQFQ/-1FY/-2FY/-3FY/-4FY)
       -- =========================================================================
       rqf.revenue_fq,
       rqf.revenue_fy,
       rqf.revenue_ltm,
       rqf.revenue_5y_avg,
       rqf.revenue_1fqfq,
       rqf.revenue_2fqfq,
       rqf.revenue_3fqfq,
       rqf.revenue_4fqfq,
       rqf.revenue_1fy,
       rqf.revenue_2fy,
       rqf.revenue_3fy,
       rqf.revenue_4fy,
       rqf.revenue_qoq_growth,
       rqf.revenue_qoq_2q,
       rqf.revenue_qoq_3q,
       rqf.revenue_qoq_4q,
       rqf.revenue_yoy_quarterly,
       rqf.revenue_2y_growth,
       rqf.revenue_3y_growth,
       rqf.revenue_4y_growth,
       rqf.revenue_cagr_3y,
       rqf.revenue_cagr_4y,
       rqf.revenue_4q_trend,
       rqf.revenue_4q_avg,
       rqf.revenue_fq_vs_4q_avg,
       rqf.revenue_growth_flag,
       rqf.revenue_stability_score,
       rqf.revenue_accelerating_flag,
       rqf.revenue_positive_qoq_streak,

       -- =========================================================================
       -- TOTAL REVENUES TEMPORAL (calc_total_revenues_temporal)
       -- Source columns: Total Revenues (5YAVGFQ/5YAVGLTM/FQ/LTM)
       -- =========================================================================
       trt.revenue_5yavgfq,
       trt.revenue_5yavgltm,
       trt.revenue_vs_5y_avg_fq,
       trt.revenue_vs_5y_avg_ltm,
       trt.revenue_fq_vs_avg,
       trt.revenue_momentum

FROM vw_identifier_columns id
         LEFT JOIN calc_growth_features() gf USING (isin)
         LEFT JOIN calc_revenue_forecast_features() rff USING (isin)
         LEFT JOIN calc_revenue_estimate_consensus() rec USING (isin)
         LEFT JOIN calc_revenue_quarterly_features() rqf USING (isin)
         LEFT JOIN calc_total_revenues_temporal() trt USING (isin);

COMMENT ON VIEW vw_features_growth IS
    'Growth metrics including revenue, EBITDA, FCF growth rates and forecasts.
    Source functions: calc_growth_features, calc_revenue_forecast_features,
    calc_revenue_estimate_consensus, calc_revenue_quarterly_features, calc_total_revenues_temporal';


-- =============================================================================
-- 7. vw_features_quality_risk (REFACTORED)
-- Source functions: calc_quality_features, calc_beta_risk_features,
--                   calc_financial_distress_features, calc_accounting_quality_features,
--                   calc_quality_features_comprehensive
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_quality_risk AS
SELECT id.*,

       -- =========================================================================
       -- QUALITY FEATURES (calc_quality_features)
       -- Source columns: Impairment of Goodwill (LTM), Asset Writedown (LTM),
       --                 Restructuring Charges (LTM), Goodwill (LTM), Total Assets (LTM),
       --                 Gross Intangible Assets (LTM), EBITDA (LTM), Altman Z-Score (LTM/FY),
       --                 Current Ratio (LTM), Total Current Assets (LTM), Inventory (LTM),
       --                 Total Current Liabilities (LTM)
       -- =========================================================================
       qf.has_goodwill_impairment,
       qf.has_asset_writedown,
       qf.has_restructuring,
       qf.goodwill_to_assets_pct,
       qf.intangible_intensity,
       qf.exceptional_items_to_ebitda,
       qf.altman_z_score,
       qf.altman_z_trend,
       qf.current_ratio,
       qf.quick_ratio,

       -- =========================================================================
       -- BETA RISK FEATURES (calc_beta_risk_features)
       -- Source columns: Beta (1Y), Beta (5Y)
       -- =========================================================================
       br.beta_1y,
       br.beta_5y,
       br.beta_spread,
       br.beta_trend,
       br.high_beta_flag,
       br.low_beta_flag,
       br.beta_stability_score,

       -- =========================================================================
       -- FINANCIAL DISTRESS FEATURES (calc_financial_distress_features)
       -- Source columns: Altman Z-Score (LTM), Current Ratio (LTM), Working Capital (FQ/FY),
       --                 Cash And Equivalents (FQ), Total Operating Expenses (LTM),
       --                 Retained Earnings (FQ/FY)
       -- =========================================================================
       fdf.distress_risk_score,
       fdf.liquidity_stress_score,
       fdf.working_capital_trend,
       fdf.cash_runway_months,
       fdf.combined_distress_score,
       fdf.wc_deteriorating_flag,
       fdf.retained_earnings_growth,
       fdf.accumulated_deficit_flag,
       fdf.adequate_cash_buffer,

       -- =========================================================================
       -- ACCOUNTING QUALITY FEATURES (calc_accounting_quality_features)
       -- Source columns: Goodwill (LTM/-1FY), Restructuring Charges (LTM), Total Assets (LTM),
       --                 Impairment of Goodwill (FQ), Asset Writedown (FQ), Restructuring Charges (FQ),
       --                 Merger & Restructuring Charges (LTM), Market Cap,
       --                 Interest And Investment Income (LTM), Net Income - (IS) (LTM),
       --                 Gain (Loss) On Sale Of Assets (LTM)
       -- =========================================================================
       aqf.goodwill_change_rate,
       aqf.restructuring_intensity,
       aqf.exceptional_items_frequency,
       aqf.merger_impact_ratio,
       aqf.non_operating_income_share,
       aqf.asset_sale_boost,
       aqf.accounting_quality_score,

       -- =========================================================================
       -- QUALITY FEATURES COMPREHENSIVE (calc_quality_features_comprehensive)
       -- Source columns: Impairment of Goodwill (LTM/FY/-1FY/-2FY/-3FY/-4FY),
       --                 Asset Writedown (LTM/FY/-1FY/-2FY/-3FY/-4FY),
       --                 Restructuring Charges (LTM/FY/-1FY/-2FY/-3FY/-4FY), EBITDA (LTM)
       -- =========================================================================
       qfc.goodwill_impairment_ltm,
       qfc.asset_writedown_ltm,
       qfc.restructuring_ltm,
       qfc.has_goodwill_impairment_ltm,
       qfc.goodwill_impairment_frequency,
       qfc.asset_writedown_frequency,
       qfc.restructuring_frequency,
       qfc.exceptional_items_total_ltm,
       qfc.exceptional_items_to_ebitda AS exceptional_items_to_ebitda_comp,
       qfc.quality_issues_count_5y,
       qfc.accounting_quality_score    AS accounting_quality_score_comp

FROM vw_identifier_columns id
         LEFT JOIN calc_quality_features() qf USING (isin)
         LEFT JOIN calc_beta_risk_features() br USING (isin)
         LEFT JOIN calc_financial_distress_features() fdf USING (isin)
         LEFT JOIN calc_accounting_quality_features() aqf USING (isin)
         LEFT JOIN calc_quality_features_comprehensive() qfc USING (isin);

COMMENT ON VIEW vw_features_quality_risk IS
    'Quality and risk metrics including accounting quality, financial distress, and beta analysis.
    Source functions: calc_quality_features, calc_beta_risk_features, calc_financial_distress_features,
    calc_accounting_quality_features, calc_quality_features_comprehensive';


-- =============================================================================
-- 8. vw_features_leverage_liquidity (REFACTORED)
-- Source functions: calc_leverage_features, calc_efficiency_ratios,
--                   calc_balance_sheet_dynamics, calc_working_capital_temporal,
--                   calc_total_debt_temporal, calc_working_capital_deep_features
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_leverage_liquidity AS
SELECT id.*,

       -- =========================================================================
       -- LEVERAGE FEATURES (calc_leverage_features)
       -- Source columns: Total Debt (LTM), Total Equity (LTM), Total Assets (LTM),
       --                 EBIT (LTM), Interest Expense/Total (LTM), Current Ratio (LTM),
       --                 Cash And Equivalents (LTM), Total Current Liabilities (LTM),
       --                 Working Capital (LTM)
       -- =========================================================================
       lf.debt_to_equity,
       lf.debt_to_assets,
       lf.equity_ratio,
       lf.interest_coverage,
       lf.current_ratio,
       lf.cash_ratio,
       lf.working_capital_ratio,

       -- =========================================================================
       -- EFFICIENCY RATIOS (calc_efficiency_ratios)
       -- Source columns: Total Revenues (LTM/FY), Total Assets (LTM),
       --                 Cost Of Revenues (LTM), Inventory (LTM),
       --                 Accounts Receivable/Total (FY), Working Capital (LTM)
       -- =========================================================================
       er.asset_turnover,
       er.inventory_turnover,
       er.receivables_days,
       er.working_capital_turns,

       -- =========================================================================
       -- BALANCE SHEET DYNAMICS (calc_balance_sheet_dynamics)
       -- Source columns: Cash And Equivalents (LTM/FQ/FY/5YAVGFQ), Total Assets (LTM),
       --                 Inventory (FQ/FY/5YAVGFQ), Accounts Receivable/Total (FY/-1FY/5YAVGFQ),
       --                 Working Capital (FQ/5YAVGFY), Retained Earnings (FQ/5YAVGFQ),
       --                 Gross Intangible Assets (FY/5YAVGFQ), Goodwill (LTM),
       --                 Total Equity (LTM), Current Ratio (LTM), Total Debt (LTM), EBITDA (LTM)
       -- =========================================================================
       bsd.cash_to_assets_pct,
       bsd.cash_change_qoq,
       bsd.cash_vs_5y_avg,
       bsd.inventory_change_yoy,
       bsd.inventory_vs_5y_avg,
       bsd.receivables_change_yoy,
       bsd.receivables_vs_5y_avg,
       bsd.working_capital_vs_5y_avg,
       bsd.retained_earnings_vs_5y,
       bsd.intangibles_growth_flag,
       bsd.asset_quality_score,
       bsd.balance_sheet_strength,
       bsd.debt_maturity_risk,

       -- =========================================================================
       -- WORKING CAPITAL TEMPORAL (calc_working_capital_temporal)
       -- Source columns: Working Capital (FQ/FY/LTM/5YAVGFY/-1FQ/-2FQ/-3FQ/-4FQ/-1FY/-2FY/-3FY/-4FY)
       -- =========================================================================
       wct.wc_fq,
       wct.wc_fy,
       wct.wc_ltm,
       wct.wc_5yavgfy,
       wct.wc_1fq,
       wct.wc_2fq,
       wct.wc_3fq,
       wct.wc_4fq,
       wct.wc_1fy,
       wct.wc_2fy,
       wct.wc_3fy,
       wct.wc_4fy,
       wct.wc_qoq_change,
       wct.wc_yoy_change,
       wct.wc_4q_trend,
       wct.wc_vs_5y_avg,
       wct.wc_positive_quarters,
       wct.wc_improving_flag,
       wct.wc_volatility,

       -- =========================================================================
       -- TOTAL DEBT TEMPORAL (calc_total_debt_temporal)
       -- Source columns: Total Debt (FQ/FY/LTM/-1FQ/-2FQ/-3FQ/-4FQ/-1FY/-2FY/-3FY/-4FY),
       --                 Total Equity (FY)
       -- =========================================================================
       tdt.debt_fq,
       tdt.debt_fy,
       tdt.debt_ltm,
       tdt.debt_1fq,
       tdt.debt_2fq,
       tdt.debt_3fq,
       tdt.debt_4fq,
       tdt.debt_1fy,
       tdt.debt_2fy,
       tdt.debt_3fy,
       tdt.debt_4fy,
       tdt.debt_qoq_change,
       tdt.debt_yoy_change,
       tdt.debt_4q_trend,
       tdt.debt_3y_cagr,
       tdt.debt_deleveraging,
       tdt.debt_to_equity_trend,

       -- =========================================================================
       -- WORKING CAPITAL DEEP FEATURES (calc_working_capital_deep_features)
       -- Source columns: Working Capital (LTM/FQ/FY/-1FY), Total Revenues (LTM),
       --                 Total Assets (LTM), Current Ratio (LTM)
       -- =========================================================================
       wcd.working_capital_ltm AS wc_ltm_deep,
       wcd.working_capital_fq  AS wc_fq_deep,
       wcd.working_capital_fy  AS wc_fy_deep,
       wcd.wc_to_revenue,
       wcd.wc_to_assets,
       wcd.wc_change_qoq       AS wc_change_qoq_deep,
       wcd.wc_change_yoy       AS wc_change_yoy_deep,
       wcd.days_working_capital,
       wcd.wc_efficiency_score,
       wcd.negative_wc_flag,
       wcd.wc_improvement_flag AS wc_improvement_flag_deep

FROM vw_identifier_columns id
         LEFT JOIN calc_leverage_features() lf USING (isin)
         LEFT JOIN calc_efficiency_ratios() er USING (isin)
         LEFT JOIN calc_balance_sheet_dynamics() bsd USING (isin)
         LEFT JOIN calc_working_capital_temporal() wct USING (isin)
         LEFT JOIN calc_total_debt_temporal() tdt USING (isin)
         LEFT JOIN calc_working_capital_deep_features() wcd USING (isin);

COMMENT ON VIEW vw_features_leverage_liquidity IS
    'Leverage and liquidity metrics including debt ratios, working capital, and balance sheet dynamics.
    Source functions: calc_leverage_features, calc_efficiency_ratios, calc_balance_sheet_dynamics,
    calc_working_capital_temporal, calc_total_debt_temporal, calc_working_capital_deep_features';


-- =============================================================================
-- 9. vw_features_analyst_sentiment (REFACTORED)
-- Source functions: calc_sentiment_features, calc_price_target_dynamics
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_analyst_sentiment AS
SELECT id.*,

       -- =========================================================================
       -- SENTIMENT FEATURES (calc_sentiment_features)
       -- Source columns: # Strong Buys Ratings, # Buys Ratings, # Hold Ratings,
       --                 # Sell Ratings, # Strong Sell Ratings, Price Target - Median,
       --                 Last Price, Price Target - High, Price Target - Low,
       --                 Price Target, Price Target (1M/3M Ago), EPS Est Avg Rev % (FY1E - 1W/1M/3M/6M/1Y),
       --                 Analyst Rating, Price Target - #, Market Cap
       -- =========================================================================
       sf.analyst_bullish_pct,
       sf.analyst_bearish_pct,
       sf.analyst_neutral_pct,
       sf.analyst_conviction,
       sf.upside_potential,
       sf.price_target_spread_pct,
       sf.price_target_revision_1m,
       sf.price_target_revision_3m,
       sf.eps_revision_momentum,
       sf.analyst_rating_normalized,
       sf.analyst_coverage_quality,

       -- =========================================================================
       -- PRICE TARGET DYNAMICS (calc_price_target_dynamics)
       -- Source columns: Price Target, Price Target (1W/1M/3M/6M/1Y Ago),
       --                 Price Target - Median, Price Target - Median (1M/3M Ago),
       --                 Price Target - High (3M Ago), Price Target - Low (3M Ago),
       --                 Price Target - #, Price Target - # (1M/3M/6M/1Y Ago),
       --                 Last Price, Price (3M Ago)
       -- =========================================================================
       ptd.pt_momentum_1w,
       ptd.pt_momentum_1m,
       ptd.pt_momentum_3m,
       ptd.pt_momentum_6m,
       ptd.pt_momentum_1y,
       ptd.pt_median_momentum_1m,
       ptd.pt_median_momentum_3m,
       ptd.pt_acceleration_short,
       ptd.pt_acceleration_long,
       ptd.pt_consensus_convergence,
       ptd.analyst_coverage_change_1m,
       ptd.analyst_coverage_change_3m,
       ptd.analyst_coverage_change_1y,
       ptd.pt_vs_price_momentum,
       ptd.analyst_coverage_trend

FROM vw_identifier_columns id
         LEFT JOIN calc_sentiment_features() sf USING (isin)
         LEFT JOIN calc_price_target_dynamics() ptd USING (isin);

COMMENT ON VIEW vw_features_analyst_sentiment IS
    'Analyst sentiment metrics including ratings distribution and price target dynamics.
    Source functions: calc_sentiment_features, calc_price_target_dynamics';


-- =============================================================================
-- 10. vw_features_dividends (NEW)
-- Source functions: calc_dividend_features, calc_dividend_timing,
--                   calc_dividend_yield_comprehensive
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_dividends AS
SELECT id.*,

       -- =========================================================================
       -- DIVIDEND FEATURES (calc_dividend_features)
       -- Source columns: Dividend Streak, Div Yield (LTM/NTM), Common Dividends Paid (LTM),
       --                 Net Income/Adj. (LTM), FCF (LTM), Buyback Yield (LTM)
       -- =========================================================================
       df.dividend_streak,
       df.dividend_yield_ltm,
       df.dividend_yield_ntm,
       df.dividend_payout_ratio,
       df.fcf_dividend_coverage,
       df.buyback_yield,
       df.total_shareholder_yield,
       df.dividend_growth_expectation,

       -- =========================================================================
       -- DIVIDEND TIMING (calc_dividend_timing)
       -- Source columns: Dividend Record (Ex Date/Payable Date/Announce Date/Frequency),
       --                 Dividend Streak, Div Yield (Ind/-1FYInd/LTM/5YAVGLTM)
       -- =========================================================================
       dt.days_since_ex_date,
       dt.days_to_payment,
       dt.dividend_announced_flag,
       dt.ex_date_approaching_flag,
       dt.dividend_frequency_score,
       dt.dividend_consistency,
       dt.recent_dividend_change,
       dt.dividend_yield_vs_5y_avg,

       -- =========================================================================
       -- DIVIDEND YIELD COMPREHENSIVE (calc_dividend_yield_comprehensive)
       -- Source columns: Div Yield (LTM/NTM/Ind/-1FYInd/5YAVGLTM), Dividend Streak,
       --                 FCF (LTM), Common Dividends Paid (LTM)
       -- =========================================================================
       dyc.div_yield_ltm,
       dyc.div_yield_ntm,
       dyc.div_yield_ind,
       dyc.div_yield_1fy_ind,
       dyc.div_yield_5y_avg,
       dyc.div_yield_vs_5y_avg,
       dyc.div_yield_growth_expected,
       dyc.dividend_streak AS dividend_streak_comp,
       dyc.high_yield_flag,
       dyc.sustainable_dividend_flag

FROM vw_identifier_columns id
         LEFT JOIN calc_dividend_features() df USING (isin)
         LEFT JOIN calc_dividend_timing() dt USING (isin)
         LEFT JOIN calc_dividend_yield_comprehensive() dyc USING (isin);

COMMENT ON VIEW vw_features_dividends IS
    'Dividend metrics including yield, payout ratios, timing, and sustainability.
    Source functions: calc_dividend_features, calc_dividend_timing, calc_dividend_yield_comprehensive';


-- =============================================================================
-- 11. vw_features_employment (NEW)
-- Source functions: calc_employment_features, calc_employment_dynamics
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_employment AS
SELECT id.*,

       -- =========================================================================
       -- EMPLOYMENT FEATURES (calc_employment_features)
       -- Source columns: Full Time Employees (FY/-1FY/-3FY), Total Revenues (FY),
       --                 Normalized Net Income (FY), EBITDA (FY), Total Assets (FY),
       --                 Avg Employees (5YAVGFY)
       -- =========================================================================
       ef.revenue_per_employee,
       ef.profit_per_employee,
       ef.ebitda_per_employee,
       ef.assets_per_employee,
       ef.fte_growth_1y_pct,
       ef.fte_growth_3y_pct,
       ef.workforce_stability,

       -- =========================================================================
       -- EMPLOYMENT DYNAMICS (calc_employment_dynamics)
       -- Source columns: Full Time Employees (FY/-1FY/-2FY/-3FY),
       --                 Total Revenues (FY/-1FY)
       -- =========================================================================
       ed.fte_growth_2y_pct,
       ed.fte_acceleration,
       ed.workforce_volatility,
       ed.hiring_intensity,
       ed.productivity_trend,
       ed.headcount_vs_revenue,
       ed.workforce_efficiency_gain,
       ed.layoff_risk_flag,
       ed.rapid_hiring_flag,
       ed.sustainable_growth_flag

FROM vw_identifier_columns id
         LEFT JOIN calc_employment_features() ef USING (isin)
         LEFT JOIN calc_employment_dynamics() ed USING (isin);

COMMENT ON VIEW vw_features_employment IS
    'Employment metrics including productivity, workforce trends, and efficiency.
    Source functions: calc_employment_features, calc_employment_dynamics';


-- =============================================================================
-- 12. vw_features_cashflow (NEW)
-- Source functions: calc_cashflow_features, calc_enhanced_cashflow_features,
--                   calc_cashflow_temporal_features, calc_cashflow_comprehensive
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_cashflow AS
SELECT id.*,

       -- =========================================================================
       -- CASHFLOW FEATURES (calc_cashflow_features)
       -- Source columns: CFO (LTM/-1FY), FCF (LTM/FQ/-1FQFQ/-2FQFQ/-3FQFQ/-4FQFQ),
       --                 Net Income - (IS) (LTM), Total Revenues (LTM), CFI (LTM),
       --                 Cash Acquisitions (FQ/-1FQFQ/-2FQFQ/-3FQFQ)
       -- =========================================================================
       cf.cfo_to_net_income,
       cf.fcf_to_net_income,
       cf.fcf_margin,
       cf.cfo_growth_yoy,
       cf.fcf_positive_ratio,
       cf.acquisition_intensity,
       cf.self_funding_ratio,

       -- =========================================================================
       -- ENHANCED CASHFLOW FEATURES (calc_enhanced_cashflow_features)
       -- Source columns: FCF (FY/-1FY/-2FY/-3FY/-4FY/LTM/FQ/-4FQFQ),
       --                 Capital Expenditure (FQ/-1FQFQ/-2FQFQ/-3FQFQ/-4FQFQ/5YAVGFQ/FY/-1FY/-2FY/-3FY/LTM),
       --                 CFO (LTM), CFI (LTM), CFF (LTM), Net Income - (IS) (LTM),
       --                 Cash Acquisitions (FQ/-1FQFQ/-2FQFQ/-3FQFQ/FY/-1FY/-2FY/-3FY/5YAVGFQ/LTM),
       --                 Total Assets (LTM), Total Revenues (FY/-1FY)
       -- =========================================================================
       ecf.fcf_positive_years,
       ecf.fcf_always_positive,
       ecf.capex_vs_5y_avg,
       ecf.underinvestment_flag,
       ecf.cfo_share_of_cf,
       ecf.cfi_share_of_cf,
       ecf.cff_share_of_cf,
       ecf.self_funding_flag,
       ecf.acquisition_to_fcf,
       ecf.sustainable_ma_flag,
       ecf.fcf_4q_improvement,
       ecf.cash_flow_quality_score,
       ecf.capex_yoy_growth,
       ecf.capex_qoq_growth,
       ecf.capex_3y_trend,
       ecf.capex_volatility,
       ecf.capex_acceleration,
       ecf.capex_cut_flag,
       ecf.overinvestment_flag,
       ecf.acquisitions_yoy_growth,
       ecf.acquisitions_vs_5y_avg,
       ecf.acquisitions_ltm_total,
       ecf.ma_intensity_score,
       ecf.serial_acquirer_flag,
       ecf.acquisition_pause_flag,
       ecf.total_investment_to_cfo,
       ecf.organic_vs_inorganic,
       ecf.investment_efficiency,

       -- =========================================================================
       -- CASHFLOW TEMPORAL FEATURES (calc_cashflow_temporal_features)
       -- Source columns: CFO (FQ/-1FQFQ/-2FQFQ/-3FQFQ/-4FQFQ/LTM),
       --                 CFI (FQ/-1FQFQ/-2FQFQ/-3FQFQ/-4FQFQ/LTM),
       --                 CFF (FQ/-1FQFQ/-2FQFQ/-3FQFQ/-4FQFQ/LTM),
       --                 FCF (FQ/-4FQFQ/LTM), Cash And Equivalents (FQ)
       -- =========================================================================
       ctf.cfo_quarterly_trend,
       ctf.cfo_yoy_quarterly,
       ctf.cfi_quarterly_trend,
       ctf.cff_quarterly_trend,
       ctf.fcf_quarterly_trend,
       ctf.cfo_positive_quarters,
       ctf.cfi_negative_quarters,
       ctf.cff_pattern_score,
       ctf.cash_burn_rate,
       ctf.cf_volatility_score,
       ctf.operating_cf_momentum,
       ctf.financing_dependency,

       -- =========================================================================
       -- CASHFLOW COMPREHENSIVE (calc_cashflow_comprehensive)
       -- Source columns: CFO (FQ/LTM/FY/-1FY/-2FY/-3FY/-4FY), FCF (FQ/LTM/FY/-1FY/-2FY/-3FY/-4FY),
       --                 Net Income - (IS) (LTM), Total Revenues (LTM), Market Cap, CFI (LTM)
       -- =========================================================================
       cc.cfo_fq,
       cc.cfo_ltm,
       cc.cfo_fy,
       cc.fcf_fq,
       cc.fcf_ltm,
       cc.fcf_fy,
       cc.cfo_growth_yoy          AS cfo_growth_yoy_comp,
       cc.fcf_growth_yoy,
       cc.cfo_to_net_income       AS cfo_to_net_income_comp,
       cc.fcf_margin              AS fcf_margin_comp,
       cc.fcf_yield,
       cc.cfo_positive_years,
       cc.fcf_positive_years      AS fcf_positive_years_comp,
       cc.cash_flow_quality_score AS cash_flow_quality_score_comp,

       -- =========================================================================
       -- FCF GROWTH ESTIMATES (calc_fcf_growth_estimates) â€” NEW
       -- Source columns: FCF - Est Avg (FY1E/FY2E/FY3E/FY4E/FY5E),
       --                 FCF (LTM/FY/-1FY), Total Revenues (LTM), Market Cap,
       --                 CFO (LTM), Capital Expenditure (LTM)
       -- =========================================================================
       fge.fcf_est_fy1,
       fge.fcf_est_fy2,
       fge.fcf_est_fy3,
       fge.fcf_est_fy4,
       fge.fcf_est_fy5,
       fge.fcf_est_growth_fy1_vs_ltm,
       fge.fcf_est_growth_fy2_vs_fy1,
       fge.fcf_est_growth_fy3_vs_fy2,
       fge.fcf_est_growth_fy4_vs_fy3,
       fge.fcf_est_growth_fy5_vs_fy4,
       fge.fcf_est_cagr_3y,
       fge.fcf_est_cagr_5y,
       fge.fcf_est_margin_fy1,
       fge.fcf_est_yield_fy1,
       fge.fcf_est_growth_acceleration,
       fge.fcf_est_growth_deceleration,
       fge.fcf_est_trajectory_score,
       fge.fcf_est_always_positive,
       fge.fcf_est_vs_historical,
       fge.fcf_est_capex_implied_ratio

FROM vw_identifier_columns id
         LEFT JOIN calc_cashflow_features() cf USING (isin)
         LEFT JOIN calc_enhanced_cashflow_features() ecf USING (isin)
         LEFT JOIN calc_cashflow_temporal_features() ctf USING (isin)
         LEFT JOIN calc_cashflow_comprehensive() cc USING (isin)
         LEFT JOIN calc_fcf_growth_estimates() fge USING (isin);

COMMENT ON VIEW vw_features_cashflow IS
    'Cash flow metrics including CFO, FCF, CapEx analysis, and cash flow quality.
    Source functions: calc_cashflow_features, calc_enhanced_cashflow_features,
    calc_cashflow_temporal_features, calc_cashflow_comprehensive';


-- =============================================================================
-- 13. vw_features_temporal (NEW)
-- Source functions: calc_temporal_features, calc_fiscal_calendar_features
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_temporal AS
SELECT id.*,

       -- =========================================================================
       -- TEMPORAL FEATURES (calc_temporal_features)
       -- Source columns: Fiscal Quarter, Fiscal Month, Fiscal Year, Next Earnings,
       --                 Income Statement Report Date, Reporting Lag
       -- =========================================================================
       tf.fiscal_quarter,
       tf.fiscal_month,
       tf.fiscal_year,
       tf.days_to_earnings,
       tf.earnings_report_recency,
       tf.reporting_lag,
       tf.fiscal_year_progress,

       -- =========================================================================
       -- FISCAL CALENDAR FEATURES (calc_fiscal_calendar_features)
       -- Source columns: Income Statement Report Date, FY End Date, Next Earnings
       -- =========================================================================
       fcf.days_since_last_report,
       fcf.days_to_fy_end,
       fcf.is_quarter_end_month,
       fcf.is_fy_end_month,
       fcf.earnings_season_flag,
       fcf.pre_earnings_window,
       fcf.post_earnings_window,
       fcf.reporting_freshness_score,
       fcf.fiscal_quarter_progress

FROM vw_identifier_columns id
         LEFT JOIN calc_temporal_features() tf USING (isin)
         LEFT JOIN calc_fiscal_calendar_features() fcf USING (isin);

COMMENT ON VIEW vw_features_temporal IS
    'Temporal and fiscal calendar features for earnings timing and seasonality.
    Source functions: calc_temporal_features, calc_fiscal_calendar_features';


-- =============================================================================
-- 14. vw_features_balance_sheet (NEW)
-- Source functions: calc_total_assets_temporal, calc_inventory_temporal_features,
--                   calc_goodwill_temporal_features
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_balance_sheet AS
SELECT id.*,

       -- =========================================================================
       -- TOTAL ASSETS TEMPORAL (calc_total_assets_temporal)
       -- Source columns: Total Assets (FQ/-1FQ/-2FQ/-3FQ/-4FQ/FY/-1FY/-2FY/-3FY/-4FY/LTM)
       -- =========================================================================
       tat.assets_fq,
       tat.assets_fy,
       tat.assets_ltm,
       tat.assets_1fq,
       tat.assets_2fq,
       tat.assets_3fq,
       tat.assets_4fq,
       tat.assets_1fy,
       tat.assets_2fy,
       tat.assets_3fy,
       tat.assets_4fy,
       tat.assets_qoq_growth,
       tat.assets_yoy_growth,
       tat.assets_3y_cagr,
       tat.asset_growth_accel,
       tat.asset_base_stable,

       -- =========================================================================
       -- INVENTORY TEMPORAL FEATURES (calc_inventory_temporal_features)
       -- Source columns: Inventory (LTM/FQ/FY/-1FQ/-2FQ/-3FQ/-4FQ/-1FY/-2FY/-3FY/-4FY/5YAVGFQ),
       --                 Cost Of Revenues (LTM), Total Revenues (LTM/FQ/-4FQFQ), Total Assets (LTM)
       -- =========================================================================
       itf.inventory_ltm,
       itf.inventory_fq,
       itf.inventory_fy,
       itf.inventory_1fq,
       itf.inventory_2fq,
       itf.inventory_3fq,
       itf.inventory_4fq,
       itf.inventory_1fy,
       itf.inventory_2fy,
       itf.inventory_3fy,
       itf.inventory_4fy,
       itf.inventory_qoq_change,
       itf.inventory_yoy_change,
       itf.inventory_4q_trend,
       itf.inventory_vs_5y_avg,
       itf.inventory_days,
       itf.inventory_turnover,
       itf.inventory_to_revenue,
       itf.inventory_to_assets,
       itf.inventory_buildup_flag,
       itf.inventory_reduction_flag,
       itf.inventory_volatility,

       -- =========================================================================
       -- GOODWILL TEMPORAL FEATURES (calc_goodwill_temporal_features)
       -- Source columns: Goodwill (FQ/-1FQ/-2FQ/-3FQ/-4FQ/LTM/FY/-1FY/-2FY/-3FY/-4FY/5YAVGFQ),
       --                 Total Assets (FY/-1FY/LTM), Net Income - (IS) (FY/-1FY), Total Equity (LTM)
       -- =========================================================================
       gtf.goodwill_fq,
       gtf.goodwill_ltm,
       gtf.goodwill_fy,
       gtf.goodwill_1fq,
       gtf.goodwill_2fq,
       gtf.goodwill_3fq,
       gtf.goodwill_4fq,
       gtf.goodwill_1fy,
       gtf.goodwill_2fy,
       gtf.goodwill_3fy,
       gtf.goodwill_4fy,
       gtf.goodwill_qoq_change,
       gtf.goodwill_yoy_change,
       gtf.goodwill_3y_growth,
       gtf.goodwill_vs_5y_avg,
       gtf.recent_acquisition_flag,
       gtf.goodwill_accumulation_rate,
       gtf.goodwill_to_assets_trend,
       gtf.impairment_risk_score,
       gtf.goodwill_concentration

FROM vw_identifier_columns id
         LEFT JOIN calc_total_assets_temporal() tat USING (isin)
         LEFT JOIN calc_inventory_temporal_features() itf USING (isin)
         LEFT JOIN calc_goodwill_temporal_features() gtf USING (isin);

COMMENT ON VIEW vw_features_balance_sheet IS
    'Balance sheet temporal analysis including assets, inventory, and goodwill trends.
    Source functions: calc_total_assets_temporal, calc_inventory_temporal_features, calc_goodwill_temporal_features';


-- =============================================================================
-- 15. vw_features_cost_structure (NEW)
-- Source functions: calc_cost_structure_features, calc_rnd_temporal_features,
--                   calc_interest_income_features
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_cost_structure AS
SELECT id.*,

       -- =========================================================================
       -- COST STRUCTURE FEATURES (calc_cost_structure_features)
       -- Source columns: Cost Of Revenues (LTM), Total Operating Expenses (LTM),
       --                 Selling General & Admin Expenses/Total (FY/-1FY/FQ/5YAVGFQ),
       --                 R&D Expenses (LTM), Interest Expense/Total (LTM),
       --                 Total Revenues (LTM/FY/-1FY), Operating Income (FY/-1FY),
       --                 Marketing Expenses (FY/-1FY/5YAVGLTM)
       -- =========================================================================
       csf.cogs_to_revenue,
       csf.opex_to_revenue,
       csf.sga_to_revenue,
       csf.rnd_to_revenue,
       csf.interest_to_revenue,
       csf.sga_trend_yoy,
       csf.operating_leverage_proxy,
       csf.cost_efficiency_score,
       csf.marketing_to_revenue,
       csf.marketing_trend_yoy,
       csf.marketing_vs_5y_avg,
       csf.sga_vs_5y_avg,
       csf.sga_efficiency_trend,

       -- =========================================================================
       -- R&D TEMPORAL FEATURES (calc_rnd_temporal_features)
       -- Source columns: R&D Expenses (LTM/FQ/FY/-1FQFQ/-2FQFQ/-3FQFQ/-4FQFQ/-1FY/-2FY/-3FY/-4FY),
       --                 Total Revenues (LTM/FY/-1FY), Full Time Employees (FY), Gross Profit (LTM)
       -- =========================================================================
       rtf.rnd_ltm,
       rtf.rnd_fq,
       rtf.rnd_fy,
       rtf.rnd_1fqfq,
       rtf.rnd_2fqfq,
       rtf.rnd_3fqfq,
       rtf.rnd_4fqfq,
       rtf.rnd_1fy,
       rtf.rnd_2fy,
       rtf.rnd_3fy,
       rtf.rnd_4fy,
       rtf.rnd_intensity_ltm,
       rtf.rnd_intensity_fy,
       rtf.rnd_intensity_trend,
       rtf.rnd_qoq_growth,
       rtf.rnd_yoy_growth,
       rtf.rnd_cagr_3y,
       rtf.rnd_per_employee,
       rtf.rnd_to_gross_profit,
       rtf.rnd_roi_proxy,
       rtf.rnd_increasing_flag,
       rtf.rnd_cut_flag,
       rtf.high_rnd_intensity_flag,

       -- =========================================================================
       -- INTEREST INCOME FEATURES (calc_interest_income_features)
       -- Source columns: Interest And Investment Income (LTM), Interest Expense/Total (LTM),
       --                 EBIT (LTM), Total Revenues (LTM), Total Assets (LTM)
       -- =========================================================================
       iif.interest_income_ltm,
       iif.interest_expense_ltm,
       iif.net_interest_income,
       iif.interest_coverage_ratio,
       iif.interest_income_to_revenue,
       iif.interest_expense_to_revenue,
       iif.net_interest_margin_proxy

FROM vw_identifier_columns id
         LEFT JOIN calc_cost_structure_features() csf USING (isin)
         LEFT JOIN calc_rnd_temporal_features() rtf USING (isin)
         LEFT JOIN calc_interest_income_features() iif USING (isin);

COMMENT ON VIEW vw_features_cost_structure IS
    'Cost structure metrics including SG&A, R&D intensity, and interest analysis.
    Source functions: calc_cost_structure_features, calc_rnd_temporal_features, calc_interest_income_features';


-- =============================================================================
-- 16. vw_features_composite_scores (NEW)
-- Source functions: calc_composite_scores, calc_net_income_comprehensive
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_composite_scores AS
SELECT id.*,

       -- =========================================================================
       -- COMPOSITE SCORES (calc_composite_scores)
       -- Source columns: Return on Assets (ROA) % (LTM/FY), CFO (LTM), Net Income - (IS) (LTM),
       --                 Total Debt (LTM/FY), Total Equity (LTM/FY), Current Ratio (LTM/FY),
       --                 Shrs Out, Shrs Out (-1FY), Gross Profit Margin % (LTM/FY),
       --                 Asset Turnover (LTM/FY), Net EPS - Basic (FY/-1FY/-2FY/-3FY/-4FY/-5FY),
       --                 Return On Equity % (LTM), Last Price, Price (3M Ago), Total Revenues (FY/-1FY)
       -- =========================================================================
       cs.piotroski_f_score,
       etf.eps_trajectory_score,
       cs.dilution_score,
       cs.quality_momentum_score,

       -- =========================================================================
       -- NET INCOME COMPREHENSIVE (calc_net_income_comprehensive)
       -- Source columns: Net Income - (IS) (FQ/LTM/FY/-1FQFQ/-2FQFQ/-3FQFQ/-4FQFQ/-1FY/-2FY/-3FY/-4FY/5YAVGFQ/5YAVGLTM),
       --                 Net Income/Adj. (LTM), Normalized Net Income (LTM/5YAVGFQ/5YAVGLTM),
       --                 Net Income Margin % (LTM)
       -- =========================================================================
       nic.net_income_is_fq,
       nic.net_income_is_ltm,
       nic.net_income_is_fy,
       nic.net_income_adj_ltm,
       nic.normalized_ni_ltm,
       nic.net_income_is_1fqfq,
       nic.net_income_is_2fqfq,
       nic.net_income_is_3fqfq,
       nic.net_income_is_4fqfq,
       nic.net_income_is_1fy,
       nic.net_income_is_2fy,
       nic.net_income_is_3fy,
       nic.net_income_is_4fy,
       nic.net_income_is_5yavgfq,
       nic.net_income_is_5yavgltm,
       nic.normalized_ni_5yavgfq,
       nic.normalized_ni_5yavgltm,
       nic.net_income_growth_yoy,
       nic.net_income_margin_ltm,
       nic.ni_adjustment_ratio,
       nic.net_income_positive_years,
       nic.earnings_quality_composite,
       nic.net_income_qoq_growth,
       nic.net_income_yoy_quarterly,
       nic.net_income_vs_5y_avg,
       nic.normalized_ni_vs_5y_avg

FROM vw_identifier_columns id
         LEFT JOIN calc_composite_scores() cs USING (isin)
         LEFT JOIN calc_eps_trajectory_features() etf USING (isin)
         LEFT JOIN calc_net_income_comprehensive() nic USING (isin);

COMMENT ON VIEW vw_features_composite_scores IS
    'Composite scoring metrics including Piotroski F-Score and earnings quality.
    Source functions: calc_composite_scores, calc_net_income_comprehensive';


-- =============================================================================
-- 17. vw_features_unusual_items (NEW)
-- Source function: calc_unusual_items_features
-- =============================================================================
CREATE OR REPLACE VIEW vw_features_unusual_items AS
SELECT id.*,

       -- =========================================================================
       -- UNUSUAL ITEMS FEATURES (calc_unusual_items_features)
       -- Source columns: Other Unusual Items/Total (LTM), Impairment of Goodwill (LTM),
       --                 Asset Writedown (LTM), Restructuring Charges (LTM),
       --                 Total Revenues (LTM), EBITDA (LTM), Net Income - (IS) (LTM)
       -- =========================================================================
       uif.other_unusual_items_ltm,
       uif.impairment_goodwill_ltm,
       uif.asset_writedown_ltm,
       uif.restructuring_charges_ltm,
       uif.total_unusual_items,
       uif.unusual_items_to_revenue,
       uif.unusual_items_to_ebitda,
       uif.has_unusual_items_flag,
       uif.earnings_quality_impact

FROM vw_identifier_columns id
         LEFT JOIN calc_unusual_items_features() uif USING (isin);

COMMENT ON VIEW vw_features_unusual_items IS
    'Non-recurring and unusual items analysis for earnings quality assessment.
    Source function: calc_unusual_items_features';

-- =============================================================================
-- UNIFIED MATERIALIZED VIEW AND FEATURE REGISTRY
-- Integrates all feature calculation functions from CalcFinancialFeaturesSql.sql
-- =============================================================================

-- =============================================================================
-- SECTION 1: UNIFIED MATERIALIZED VIEW - ALL FEATURES
-- =============================================================================

DROP MATERIALIZED VIEW IF EXISTS mv_all_stock_features CASCADE;

CREATE MATERIALIZED VIEW mv_all_stock_features AS
SELECT id.*,

       -- Reference columns from equities (not in vw_identifier_columns)
       e."Market Cap"                      AS market_cap,
       e."Enterprise Value"                AS enterprise_value,
       e."Last Price"                      AS last_price,
       e."Price Target"                    AS price_target,
       e."Price Target - High"             AS price_target_high,
       e."Price Target - Low"              AS price_target_low,
       e."Price Target - Median"           AS price_target_median,
       e."Volume (Shrs)"                   AS volume_shrs,
       e."Shrs Out"                        AS shares_outstanding,

       -- =========================================================================
       -- Enhancement 1: Direct Reference Columns from equities
       -- =========================================================================
       e."Current Fiscal Quarter"          AS current_fiscal_quarter,
       e."Dividend Record (Currency)"      AS dividend_record_currency,
       e."Dividend Record (Amount)"        AS dividend_record_amount,
       e."Dividend Per Share (LTM)"        AS dividend_per_share_ltm,
       e."Market Cap (Country R)"          AS market_cap_country_r,
       e."Rel. Volume"                     AS rel_volume,
       e."1-Day %"                         AS one_day_pct,
       e."Total Return (YTD)"              AS total_return_ytd,
       e."Total Return (5Y)"               AS total_return_5y,
       e."Total Return (10Y)"              AS total_return_10y,
       e."Tot. Return %/CAGR (3Y)"         AS tot_return_pct_cagr_3y,
       e."Tot. Return %/CAGR (10Y)"        AS tot_return_pct_cagr_10y,
       e."Total Revenues/CAGR (5Y FY)"     AS total_revenues_cagr_5y_fy,
       e."Shrs Out (-1FY)"                 AS shrs_out_1fy,
       e."Analyst Rating"                  AS analyst_rating,
       e."Price Target - #"                AS price_target_count,
       e."Target % (Avg)"                  AS target_vs_price_pct,

       -- =========================================================================
       -- Enhancement 6: Analyst Rating Breakdown
       -- =========================================================================
       e."# Strong Buys Ratings"           AS num_strong_buys_ratings,
       e."# Buys Ratings"                  AS num_buys_ratings,
       e."# Hold Ratings"                  AS num_hold_ratings,
       e."# Sell Ratings"                  AS num_sell_ratings,
       e."# Strong Sell Ratings"           AS num_strong_sell_ratings,
       e."EPS Norm - Est # (FY1E)"         AS eps_norm_est_num_fy1e,

       -- =========================================================================
       -- Enhancement 7: GAAP EPS Estimate Columns
       -- =========================================================================
       e."EPS GAAP - Est Avg (NTM)"        AS eps_gaap_est_avg_ntm,
       e."EPS GAAP - Est Avg (FY1E)"       AS eps_gaap_est_avg_fy1e,
       e."EPS Norm - Est Avg (NTM)"        AS eps_norm_est_avg_ntm,
       e."EPS Norm - Est Avg (FY1E)"       AS eps_norm_est_avg_fy1e,

       -- =========================================================================
       -- SECTION 1: VALUATION RATIOS (vw_features_valuation_ratios)
       -- Source: calc_valuation_features, calc_valuation_timeseries_features,
       --         calc_extended_valuation_timeseries, calc_tangible_book_features
       -- =========================================================================
       -- calc_valuation_features
       vf.p_e_ratio,
       vf.p_b_ratio,
       vf.ev_ebitda_ratio,
       vf.ev_sales_ratio,
       vf.dividend_yield                   AS valuation_dividend_yield,
       vf.peg_ratio,

       -- calc_valuation_timeseries_features
       vts.ev_sales_trend_1y,
       vts.ev_ebitda_momentum,
       vts.p_e_momentum_yoy,
       vts.p_e_momentum_qoq,
       vts.ev_sales_vs_3y_avg,
       vts.ev_ebitda_vs_3y_avg,
       vts.p_e_vs_3y_avg,
       vts.ev_sales_forward_discount,
       vts.ev_ebitda_forward_discount,
       vts.p_e_forward_discount,
       vts.p_b_vs_5y_avg,

       -- calc_extended_valuation_timeseries
       evt.ev_sales_qoq_1q,
       evt.ev_sales_qoq_2q,
       evt.ev_sales_qoq_3q,
       evt.ev_sales_qoq_4q,
       evt.p_e_vs_5y_avg,
       evt.p_e_percentile_proxy,
       evt.valuation_mean_reversion,
       evt.ev_ebitda_qoq_trend,
       evt.p_b_momentum_yoy,
       evt.valuation_compression,
       evt.forward_pe_premium,

       -- calc_tangible_book_features
       tb.tangible_book_value_fy,
       tb.tangible_book_value_ltm,
       tb.tangible_book_per_share,
       tb.price_to_tangible_book,
       tb.tangible_equity_ratio,
       tb.intangibles_to_equity,
       tb.goodwill_to_equity,
       tb.tangible_asset_quality,
       tb.tbv_yoy_growth,
       tb.tbv_vs_calculated,

       -- =========================================================================
       -- SECTION 2: MOMENTUM (vw_features_momentum)
       -- Source: calc_momentum_features, calc_long_term_momentum_features
       -- =========================================================================
       -- calc_momentum_features
       mf.price_momentum_1m,
       mf.price_momentum_3m,
       mf.price_momentum_6m,
       mf.price_momentum_1y,
       mf.price_momentum_5d,
       mf.ema_crossover_20_50,
       mf.ema_crossover_50_250,
       mf.price_vs_ema_20d,
       mf.price_vs_ema_250d,
       mf.pct_off_52w_high,
       mf.pct_above_52w_low,
       mf.range_52w_position,
       mf.beta_momentum,
       mf.volatility_regime,

       -- calc_long_term_momentum_features
       ltm.price_momentum_3y,
       ltm.price_momentum_5y,
       ltm.long_term_trend_score,
       ltm.multi_year_high_flag,
       ltm.secular_trend_flag,

       -- =========================================================================
       -- SECTION 3: TECHNICAL ANALYSIS (vw_features_technical_analysis)
       -- Source: calc_technical_analysis_features
       -- =========================================================================
       ta.ema_slope_20d,
       ta.ema_trend_consistency,
       ta.price_vs_ema_100d,
       ta.near_52w_high_flag,
       ta.near_52w_low_flag,
       ta.volume_momentum_score,
       ta.breakout_signal,
       ta.high_volume_flag,
       ta.low_volume_flag,
       ta.volatility_compression,
       ta.volatility_term_structure,

       -- =========================================================================
       -- SECTION 4: PROFITABILITY (vw_features_profitability)
       -- Source: calc_profitability_features, calc_margin_trends,
       --         calc_ebit_ebitda_comprehensive, calc_gross_profit_temporal
       -- =========================================================================
       -- calc_profitability_features
       pf.roe,
       pf.roa,
       pf.gross_margin_pct,
       pf.operating_margin_pct,
       pf.net_margin_pct,
       pf.ebitda_margin_pct,
       pf.roic,
       pf.rnd_intensity,
       pf.equity_multiplier,

       -- calc_margin_trends
       mt.gross_margin_trend_yoy,
       mt.operating_margin_trend,
       mt.net_margin_trend_yoy,
       mt.ebitda_margin_trend,
       mt.margin_expansion_flag,
       mt.margin_stability_score,

       -- calc_ebit_ebitda_comprehensive
       eec.ebit_fq,
       eec.ebit_ltm,
       eec.ebit_fy,
       eec.ebit_1fy,
       eec.ebit_2fy,
       eec.ebit_3fy,
       eec.ebit_4fy,
       eec.ebit_1fqfq,
       eec.ebit_2fqfq,
       eec.ebit_3fqfq,
       eec.ebit_4fqfq,
       eec.ebit_5yavgfq,
       eec.ebit_5yavgltm,
       eec.ebit_adj_fq,
       eec.ebit_adj_ltm,
       eec.ebit_adj_fy,
       eec.ebitda_fq,
       eec.ebitda_ltm,
       eec.ebitda_fy,
       eec.ebitda_1fy,
       eec.ebitda_2fy,
       eec.ebitda_3fy,
       eec.ebitda_4fy,
       eec.ebitda_1fqfq,
       eec.ebitda_2fqfq,
       eec.ebitda_3fqfq,
       eec.ebitda_4fqfq,
       eec.ebitda_5yavgfq,
       eec.ebitda_5yavgltm,
       eec.ebitda_adj_fq,
       eec.ebitda_adj_ltm,
       eec.ebitda_adj_fy,
       eec.ebit_growth_yoy,
       eec.ebitda_growth_yoy,
       eec.ebit_margin_ltm,
       eec.ebitda_margin_ltm,
       eec.ebit_positive_years,
       eec.ebitda_positive_years,
       eec.ebit_qoq_growth,
       eec.ebitda_qoq_growth,
       eec.ebit_cagr_3y,
       eec.ebitda_cagr_3y,
       eec.ebit_vs_5y_avg,
       eec.ebitda_vs_5y_avg,

       -- calc_gross_profit_temporal
       gpt.gp_fq,
       gpt.gp_fy,
       gpt.gp_ltm,
       gpt.gp_1fqfq,
       gpt.gp_2fqfq,
       gpt.gp_3fqfq,
       gpt.gp_4fqfq,
       gpt.gp_1fy,
       gpt.gp_2fy,
       gpt.gp_3fy,
       gpt.gp_4fy,
       gpt.gp_qoq_growth,
       gpt.gp_yoy_growth,
       gpt.gp_margin_fq,
       gpt.gp_margin_trend,
       gpt.gp_positive_quarters,
       gpt.gp_margin_expansion,

       -- =========================================================================
       -- SECTION 5: EARNINGS (vw_features_earnings)
       -- Source: calc_earnings_features, calc_eps_trajectory_features,
       --         calc_eps_comprehensive, calc_eps_continuing_features,
       --         calc_gaap_adjusted_analytics, calc_gaap_revision_features
       -- =========================================================================
       -- calc_earnings_features
       ef.eps_surprise_pct,
       ef.revenue_surprise_pct,
       ef.eps_adjustment_ratio,
       ef.gaap_adj_eps_gap_pct,
       ef.ebitda_adjustment_ratio,
       ef.eps_quarterly_trend,
       ef.eps_yoy_growth,

       -- calc_eps_trajectory_features
       etf.eps_qoq_growth,
       etf.eps_yoy_quarterly,
       etf.eps_positive_streak,
       etf.eps_cagr_3y,
       etf.eps_cagr_5y,
       etf.eps_growth_accel,
       etf.eps_vs_5y_avg,
       etf.eps_improvement_count,
       etf.eps_trajectory_score,
       etf.eps_stability,

       -- calc_eps_comprehensive
       ec.eps_basic_fq,
       ec.eps_basic_ltm,
       ec.eps_basic_fy,
       ec.eps_adj_ltm,
       ec.eps_norm_est_fy1e,
       ec.eps_positive_years,

       -- calc_eps_continuing_features
       ecf.eps_cont_ltm,
       ecf.eps_cont_fq,
       ecf.eps_cont_fy,
       ecf.eps_cont_1fqfq,
       ecf.eps_cont_2fqfq,
       ecf.eps_cont_3fqfq,
       ecf.eps_cont_4fqfq,
       ecf.eps_cont_1fy,
       ecf.eps_cont_2fy,
       ecf.eps_cont_3fy,
       ecf.eps_cont_4fy,
       ecf.eps_cont_qoq_growth,
       ecf.eps_cont_yoy_growth,
       ecf.eps_cont_cagr_3y,
       ecf.eps_cont_vs_total_eps,
       ecf.eps_cont_positive_streak,
       ecf.eps_cont_trajectory_score,
       ecf.discontinued_ops_impact,
       ecf.core_earnings_stability,

       -- calc_gaap_adjusted_analytics
       gaa.eps_adjustment_spread_ltm,
       gaa.eps_adjustment_spread_fy,
       gaa.eps_adjustment_spread_1fy,
       gaa.eps_adjustment_spread_fq,
       gaa.eps_adjustment_spread_1fqfq,
       gaa.eps_adjustment_spread_2fqfq,
       gaa.eps_adjustment_spread_3fqfq,
       gaa.eps_adjustment_spread_4fqfq,
       gaa.eps_adjustment_spread_2fy,
       gaa.eps_adjustment_spread_3fy,
       gaa.eps_adjustment_spread_4fy,
       gaa.eps_adjustment_pct,
       gaa.net_income_adjustment_ratio_ltm,
       gaa.net_income_adjustment_ratio_fy,
       gaa.net_income_adjustment_ratio_1fy,
       gaa.net_income_adjustment_ratio_fq,
       gaa.net_income_adjustment_ratio_5yavgfq,
       gaa.net_income_adjustment_ratio_1fqfq,
       gaa.net_income_adjustment_ratio_2fqfq,
       gaa.net_income_adjustment_ratio_3fqfq,
       gaa.net_income_adjustment_ratio_4fqfq,
       gaa.net_income_adjustment_ratio_2fy,
       gaa.net_income_adjustment_ratio_3fy,
       gaa.net_income_adjustment_ratio_4fy,
       gaa.net_income_adjustment_pct,
       gaa.ebitda_adjustment_pct_ltm,
       gaa.ebitda_adjustment_pct_fy,
       gaa.ebitda_adjustment_pct_1fy,
       gaa.ebitda_adjustment_pct_fq,
       gaa.ebitda_adjustment_pct_1fqfq,
       gaa.ebitda_adjustment_pct_2fqfq,
       gaa.ebitda_adjustment_pct_3fqfq,
       gaa.ebitda_adjustment_pct_4fqfq,
       gaa.ebitda_adjustment_pct_2fy,
       gaa.ebitda_adjustment_pct_3fy,
       gaa.ebitda_adjustment_pct_4fy,
       gaa.ebit_adjustment_pct_ltm,
       gaa.ebit_adjustment_pct_fy,
       gaa.ebit_adjustment_pct_1fy,
       gaa.ebit_adjustment_pct_fq,
       gaa.ebit_adjustment_pct_1fqfq,
       gaa.ebit_adjustment_pct_2fqfq,
       gaa.ebit_adjustment_pct_3fqfq,
       gaa.ebit_adjustment_pct_4fqfq,
       gaa.ebit_adjustment_pct_2fy,
       gaa.ebit_adjustment_pct_3fy,
       gaa.ebit_adjustment_pct_4fy,
       gaa.earnings_quality_score,
       gaa.earnings_quality_warning,
       gaa.forward_eps_gaap_adj_spread,

       -- calc_gaap_revision_features
       grf.gaap_revision_momentum,
       grf.gaap_revision_1m,
       grf.gaap_revision_3m,
       grf.gaap_revision_6m,
       grf.gaap_revision_1y,
       grf.gaap_vs_norm_revision_spread,
       grf.gaap_revision_acceleration,
       grf.gaap_positive_revision_flag,
       grf.revision_quality_divergence,

       -- =========================================================================
       -- SECTION 6: GROWTH (vw_features_growth)
       -- Source: calc_growth_features, calc_revenue_forecast_features,
       --         calc_revenue_quarterly_features, calc_total_revenues_temporal
       -- =========================================================================
       -- calc_growth_features
       gf.revenue_growth_yoy,
       gf.ebitda_growth_yoy                AS growth_ebitda_growth_yoy,
       gf.operating_income_growth,
       gf.fcf_growth,
       gf.revenue_cagr_5y,
       gf.forward_revenue_growth,
       gf.revenue_vs_5y_avg,

       -- calc_revenue_forecast_features
       rff.revenue_est_spread,
       rff.revenue_beat_potential,
       rff.revenue_est_revision_trend,
       rff.ebitda_est_vs_actual,
       rff.forward_revenue_multiple,
       rff.revenue_estimate_count,
       rff.revenue_guidance_gap,
       rff.consensus_revenue_growth,
       rff.ebit_estimate_spread,
       rff.forward_ebitda_margin,
       rff.revenue_acceleration,
       rff.estimate_confidence_score,

       -- calc_revenue_quarterly_features
       rqf.revenue_fq,
       rqf.revenue_fy,
       rqf.revenue_ltm,
       rqf.revenue_5y_avg,
       rqf.revenue_1fqfq,
       rqf.revenue_2fqfq,
       rqf.revenue_3fqfq,
       rqf.revenue_4fqfq,
       rqf.revenue_1fy,
       rqf.revenue_2fy,
       rqf.revenue_3fy,
       rqf.revenue_4fy,
       rqf.revenue_qoq_growth,
       rqf.revenue_qoq_2q,
       rqf.revenue_qoq_3q,
       rqf.revenue_qoq_4q,
       rqf.revenue_yoy_quarterly,
       rqf.revenue_2y_growth,
       rqf.revenue_3y_growth,
       rqf.revenue_4y_growth,
       rqf.revenue_cagr_3y,
       rqf.revenue_cagr_4y,
       rqf.revenue_4q_trend,
       rqf.revenue_4q_avg,
       rqf.revenue_fq_vs_4q_avg,
       rqf.revenue_growth_flag,
       rqf.revenue_stability_score,
       rqf.revenue_accelerating_flag,
       rqf.revenue_positive_qoq_streak,

       -- calc_total_revenues_temporal
       trt.revenue_5yavgfq,
       trt.revenue_5yavgltm,
       trt.revenue_vs_5y_avg_fq,
       trt.revenue_vs_5y_avg_ltm,
       trt.revenue_fq_vs_avg,
       trt.revenue_momentum,

       -- =========================================================================
       -- SECTION 7: QUALITY & RISK (vw_features_quality_risk)
       -- Source: calc_quality_features, calc_beta_risk_features,
       --         calc_financial_distress_features, calc_accounting_quality_features,
       --         calc_quality_features_comprehensive
       -- =========================================================================
       -- calc_quality_features
       qf.has_goodwill_impairment,
       qf.has_asset_writedown,
       qf.has_restructuring,
       qf.goodwill_to_assets_pct,
       qf.intangible_intensity,
       qf.exceptional_items_to_ebitda,
       qf.altman_z_score,
       qf.altman_z_trend,
       qf.current_ratio,
       qf.quick_ratio,

       -- calc_beta_risk_features
       br.beta_1y,
       br.beta_5y,
       br.beta_spread,
       br.beta_trend,
       br.high_beta_flag,
       br.low_beta_flag,
       br.beta_stability_score,

       -- calc_financial_distress_features
       fdf.distress_risk_score,
       fdf.liquidity_stress_score,
       fdf.working_capital_trend,
       fdf.cash_runway_months,
       fdf.combined_distress_score,
       fdf.wc_deteriorating_flag,
       fdf.retained_earnings_growth,
       fdf.accumulated_deficit_flag,
       fdf.adequate_cash_buffer,

       -- calc_accounting_quality_features
       aqf.goodwill_change_rate,
       aqf.restructuring_intensity,
       aqf.exceptional_items_frequency,
       aqf.merger_impact_ratio,
       aqf.non_operating_income_share,
       aqf.asset_sale_boost,
       aqf.accounting_quality_score,

       -- calc_quality_features_comprehensive
       qfc.goodwill_impairment_ltm,
       qfc.asset_writedown_ltm,
       qfc.restructuring_ltm,
       qfc.has_goodwill_impairment_ltm,
       qfc.goodwill_impairment_frequency,
       qfc.asset_writedown_frequency,
       qfc.restructuring_frequency,
       qfc.exceptional_items_total_ltm,
       qfc.quality_issues_count_5y,

       -- =========================================================================
       -- SECTION 8: LEVERAGE & LIQUIDITY (vw_features_leverage_liquidity)
       -- Source: calc_leverage_features, calc_efficiency_ratios,
       --         calc_balance_sheet_dynamics, calc_working_capital_temporal,
       --         calc_total_debt_temporal, calc_working_capital_deep_features
       -- =========================================================================
       -- calc_leverage_features
       lf.debt_to_equity,
       lf.debt_to_assets,
       lf.equity_ratio,
       lf.interest_coverage,
       lf.cash_ratio,
       lf.working_capital_ratio,

       -- calc_efficiency_ratios
       er.asset_turnover,
       er.inventory_turnover,
       er.receivables_days,
       er.working_capital_turns,

       -- calc_balance_sheet_dynamics
       bsd.cash_to_assets_pct,
       bsd.cash_change_qoq,
       bsd.cash_vs_5y_avg,
       bsd.inventory_change_yoy,
       bsd.inventory_vs_5y_avg,
       bsd.receivables_change_yoy,
       bsd.receivables_vs_5y_avg,
       bsd.working_capital_vs_5y_avg,
       bsd.retained_earnings_vs_5y,
       bsd.intangibles_growth_flag,
       bsd.asset_quality_score,
       bsd.balance_sheet_strength,
       bsd.debt_maturity_risk,

       -- calc_working_capital_temporal
       wct.wc_fq,
       wct.wc_fy,
       wct.wc_ltm,
       wct.wc_5yavgfy,
       wct.wc_1fq,
       wct.wc_2fq,
       wct.wc_3fq,
       wct.wc_4fq,
       wct.wc_1fy,
       wct.wc_2fy,
       wct.wc_3fy,
       wct.wc_4fy,
       wct.wc_qoq_change,
       wct.wc_yoy_change,
       wct.wc_4q_trend,
       wct.wc_vs_5y_avg,
       wct.wc_positive_quarters,
       wct.wc_improving_flag,
       wct.wc_volatility,

       -- calc_total_debt_temporal
       tdt.debt_fq,
       tdt.debt_fy,
       tdt.debt_ltm,
       tdt.debt_1fq,
       tdt.debt_2fq,
       tdt.debt_3fq,
       tdt.debt_4fq,
       tdt.debt_1fy,
       tdt.debt_2fy,
       tdt.debt_3fy,
       tdt.debt_4fy,
       tdt.debt_qoq_change,
       tdt.debt_yoy_change,
       tdt.debt_4q_trend,
       tdt.debt_3y_cagr,
       tdt.debt_deleveraging,
       tdt.debt_to_equity_trend,

       -- calc_working_capital_deep_features
       wcd.wc_to_revenue,
       wcd.wc_to_assets,
       wcd.days_working_capital,
       wcd.wc_efficiency_score,
       wcd.negative_wc_flag,

       -- =========================================================================
       -- SECTION 9: ANALYST SENTIMENT (vw_features_analyst_sentiment)
       -- Source: calc_sentiment_features, calc_price_target_dynamics
       -- =========================================================================
       -- calc_sentiment_features
       sf.analyst_bullish_pct,
       sf.analyst_bearish_pct,
       sf.analyst_neutral_pct,
       sf.analyst_conviction,
       sf.upside_potential,
       sf.price_target_spread_pct,
       sf.price_target_revision_1m,
       sf.price_target_revision_3m,
       sf.eps_revision_momentum,
       sf.analyst_rating_normalized,
       sf.analyst_coverage_quality,

       -- calc_price_target_dynamics
       ptd.pt_momentum_1w,
       ptd.pt_momentum_1m,
       ptd.pt_momentum_3m,
       ptd.pt_momentum_6m,
       ptd.pt_momentum_1y,
       ptd.pt_median_momentum_1m,
       ptd.pt_median_momentum_3m,
       ptd.pt_acceleration_short,
       ptd.pt_acceleration_long,
       ptd.pt_consensus_convergence,
       ptd.analyst_coverage_change_1m,
       ptd.analyst_coverage_change_3m,
       ptd.analyst_coverage_change_1y,
       ptd.pt_vs_price_momentum,
       ptd.analyst_coverage_trend,

       -- =========================================================================
       -- SECTION 10: DIVIDENDS (vw_features_dividends)
       -- Source: calc_dividend_features, calc_dividend_timing,
       --         calc_dividend_yield_comprehensive
       -- =========================================================================
       -- calc_dividend_features
       df.dividend_streak,
       df.dividend_yield_ltm,
       df.dividend_yield_ntm,
       df.dividend_payout_ratio,
       df.fcf_dividend_coverage,
       df.buyback_yield,
       df.total_shareholder_yield,
       df.dividend_growth_expectation,

       -- calc_dividend_timing
       dt.days_since_ex_date,
       dt.days_to_payment,
       dt.dividend_announced_flag,
       dt.ex_date_approaching_flag,
       dt.dividend_frequency_score,
       dt.dividend_consistency,
       dt.recent_dividend_change,
       dt.dividend_yield_vs_5y_avg,

       -- calc_dividend_yield_comprehensive
       dyc.div_yield_ltm,
       dyc.div_yield_ntm,
       dyc.div_yield_ind,
       dyc.div_yield_1fy_ind,
       dyc.div_yield_5y_avg,
       dyc.div_yield_vs_5y_avg,
       dyc.div_yield_growth_expected,
       dyc.high_yield_flag,
       dyc.sustainable_dividend_flag,

       -- =========================================================================
       -- SECTION 11: EMPLOYMENT (vw_features_employment)
       -- Source: calc_employment_features, calc_employment_dynamics
       -- =========================================================================
       -- calc_employment_features
       emf.revenue_per_employee,
       emf.profit_per_employee,
       emf.ebitda_per_employee,
       emf.assets_per_employee,
       emf.fte_growth_1y_pct,
       emf.fte_growth_3y_pct,
       emf.workforce_stability,

       -- calc_employment_dynamics
       ed.fte_growth_2y_pct,
       ed.fte_acceleration,
       ed.workforce_volatility,
       ed.hiring_intensity,
       ed.productivity_trend,
       ed.headcount_vs_revenue,
       ed.workforce_efficiency_gain,
       ed.layoff_risk_flag,
       ed.rapid_hiring_flag,
       ed.sustainable_growth_flag,

       -- =========================================================================
       -- SECTION 12: CASH FLOW (vw_features_cashflow)
       -- Source: calc_cashflow_features, calc_enhanced_cashflow_features,
       --         calc_cashflow_temporal_features, calc_cashflow_comprehensive
       -- =========================================================================
       -- calc_cashflow_features
       cf.cfo_to_net_income,
       cf.fcf_to_net_income,
       cf.fcf_margin,
       cf.cfo_growth_yoy,
       cf.fcf_positive_ratio,
       cf.acquisition_intensity,
       cf.self_funding_ratio,

       -- calc_enhanced_cashflow_features
       ecff.fcf_positive_years,
       ecff.fcf_always_positive,
       ecff.capex_vs_5y_avg,
       ecff.underinvestment_flag,
       ecff.cfo_share_of_cf,
       ecff.cfi_share_of_cf,
       ecff.cff_share_of_cf,
       ecff.self_funding_flag,
       ecff.acquisition_to_fcf,
       ecff.sustainable_ma_flag,
       ecff.fcf_4q_improvement,
       ecff.cash_flow_quality_score,
       ecff.capex_yoy_growth,
       ecff.capex_qoq_growth,
       ecff.capex_3y_trend,
       ecff.capex_volatility,
       ecff.capex_acceleration,
       ecff.capex_cut_flag,
       ecff.overinvestment_flag,
       ecff.acquisitions_yoy_growth,
       ecff.acquisitions_vs_5y_avg,
       ecff.acquisitions_ltm_total,
       ecff.ma_intensity_score,
       ecff.serial_acquirer_flag,
       ecff.acquisition_pause_flag,
       ecff.total_investment_to_cfo,
       ecff.organic_vs_inorganic,
       ecff.investment_efficiency,

       -- calc_cashflow_temporal_features
       ctf.cfo_quarterly_trend,
       ctf.cfo_yoy_quarterly,
       ctf.cfi_quarterly_trend,
       ctf.cff_quarterly_trend,
       ctf.fcf_quarterly_trend,
       ctf.cfo_positive_quarters,
       ctf.cfi_negative_quarters,
       ctf.cff_pattern_score,
       ctf.cash_burn_rate,
       ctf.cf_volatility_score,
       ctf.operating_cf_momentum,
       ctf.financing_dependency,

       -- calc_cashflow_comprehensive
       cc.cfo_fq,
       cc.cfo_ltm,
       cc.cfo_fy,
       cc.fcf_fq,
       cc.fcf_ltm,
       cc.fcf_fy,
       cc.fcf_growth_yoy,
       cc.fcf_yield,
       cc.cfo_positive_years,
       cc.fcf_positive_years               AS fcf_positive_years_comp,

       -- =========================================================================
       -- SECTION 13: TEMPORAL (vw_features_temporal)
       -- Source: calc_temporal_features, calc_fiscal_calendar_features
       -- =========================================================================
       -- calc_temporal_features
       tf.fiscal_quarter,
       tf.fiscal_month,
       tf.fiscal_year,
       tf.days_to_earnings,
       tf.earnings_report_recency,
       tf.reporting_lag,
       tf.fiscal_year_progress,

       -- calc_fiscal_calendar_features
       fcf.days_since_last_report,
       fcf.days_to_fy_end,
       fcf.is_quarter_end_month,
       fcf.is_fy_end_month,
       fcf.earnings_season_flag,
       fcf.pre_earnings_window,
       fcf.post_earnings_window,
       fcf.reporting_freshness_score,
       fcf.fiscal_quarter_progress,

       -- =========================================================================
       -- SECTION 14: BALANCE SHEET (vw_features_balance_sheet)
       -- Source: calc_total_assets_temporal, calc_inventory_temporal_features,
       --         calc_goodwill_temporal_features
       -- =========================================================================
       -- calc_total_assets_temporal
       tat.assets_fq,
       tat.assets_fy,
       tat.assets_ltm,
       tat.assets_1fq,
       tat.assets_2fq,
       tat.assets_3fq,
       tat.assets_4fq,
       tat.assets_1fy,
       tat.assets_2fy,
       tat.assets_3fy,
       tat.assets_4fy,
       tat.assets_qoq_growth,
       tat.assets_yoy_growth,
       tat.assets_3y_cagr,
       tat.asset_growth_accel,
       tat.asset_base_stable,

       -- calc_inventory_temporal_features
       itf.inventory_ltm,
       itf.inventory_fq,
       itf.inventory_fy,
       itf.inventory_1fq,
       itf.inventory_2fq,
       itf.inventory_3fq,
       itf.inventory_4fq,
       itf.inventory_1fy,
       itf.inventory_2fy,
       itf.inventory_3fy,
       itf.inventory_4fy,
       itf.inventory_qoq_change,
       itf.inventory_yoy_change,
       itf.inventory_4q_trend,
       itf.inventory_vs_5y_avg             AS inventory_vs_5y_avg_itf,
       itf.inventory_days,
       itf.inventory_turnover              AS inventory_turnover_itf,
       itf.inventory_to_revenue,
       itf.inventory_to_assets,
       itf.inventory_buildup_flag,
       itf.inventory_reduction_flag,
       itf.inventory_volatility,

       -- calc_goodwill_temporal_features
       gtf.goodwill_fq,
       gtf.goodwill_ltm,
       gtf.goodwill_fy,
       gtf.goodwill_1fq,
       gtf.goodwill_2fq,
       gtf.goodwill_3fq,
       gtf.goodwill_4fq,
       gtf.goodwill_1fy,
       gtf.goodwill_2fy,
       gtf.goodwill_3fy,
       gtf.goodwill_4fy,
       gtf.goodwill_qoq_change,
       gtf.goodwill_yoy_change,
       gtf.goodwill_3y_growth,
       gtf.goodwill_vs_5y_avg,
       gtf.recent_acquisition_flag,
       gtf.goodwill_accumulation_rate,
       gtf.goodwill_to_assets_trend,
       gtf.impairment_risk_score,
       gtf.goodwill_concentration,

       -- =========================================================================
       -- SECTION 15: COST STRUCTURE (vw_features_cost_structure)
       -- Source: calc_cost_structure_features, calc_rnd_temporal_features,
       --         calc_interest_income_features
       -- =========================================================================
       -- calc_cost_structure_features
       csf.cogs_to_revenue,
       csf.opex_to_revenue,
       csf.sga_to_revenue,
       csf.rnd_to_revenue,
       csf.interest_to_revenue,
       csf.sga_trend_yoy,
       csf.operating_leverage_proxy,
       csf.cost_efficiency_score,
       csf.marketing_to_revenue,
       csf.marketing_trend_yoy,
       csf.marketing_vs_5y_avg,
       csf.sga_vs_5y_avg,
       csf.sga_efficiency_trend,

       -- calc_rnd_temporal_features
       rtf.rnd_ltm,
       rtf.rnd_fq,
       rtf.rnd_fy,
       rtf.rnd_1fqfq,
       rtf.rnd_2fqfq,
       rtf.rnd_3fqfq,
       rtf.rnd_4fqfq,
       rtf.rnd_1fy,
       rtf.rnd_2fy,
       rtf.rnd_3fy,
       rtf.rnd_4fy,
       rtf.rnd_intensity_ltm,
       rtf.rnd_intensity_fy,
       rtf.rnd_intensity_trend,
       rtf.rnd_qoq_growth,
       rtf.rnd_yoy_growth,
       rtf.rnd_cagr_3y,
       rtf.rnd_per_employee,
       rtf.rnd_to_gross_profit,
       rtf.rnd_roi_proxy,
       rtf.rnd_increasing_flag,
       rtf.rnd_cut_flag,
       rtf.high_rnd_intensity_flag,

       -- calc_interest_income_features
       iif.interest_income_ltm,
       iif.interest_expense_ltm,
       iif.net_interest_income,
       iif.interest_coverage_ratio,
       iif.interest_income_to_revenue,
       iif.interest_expense_to_revenue,
       iif.net_interest_margin_proxy,

       -- =========================================================================
       -- SECTION 16: COMPOSITE SCORES (vw_features_composite_scores)
       -- Source: calc_composite_scores, calc_net_income_comprehensive
       -- =========================================================================
       -- calc_composite_scores
       cs.piotroski_f_score,
       etf.eps_trajectory_score            AS composite_eps_trajectory_score,
       cs.dilution_score,
       cs.quality_momentum_score,

       -- calc_net_income_comprehensive
       nic.net_income_is_fq,
       nic.net_income_is_ltm,
       nic.net_income_is_fy,
       nic.net_income_adj_ltm,
       nic.normalized_ni_ltm,
       nic.net_income_is_1fqfq,
       nic.net_income_is_2fqfq,
       nic.net_income_is_3fqfq,
       nic.net_income_is_4fqfq,
       nic.net_income_is_1fy,
       nic.net_income_is_2fy,
       nic.net_income_is_3fy,
       nic.net_income_is_4fy,
       nic.net_income_is_5yavgfq,
       nic.net_income_is_5yavgltm,
       nic.normalized_ni_5yavgfq,
       nic.normalized_ni_5yavgltm,
       nic.net_income_growth_yoy,
       nic.net_income_margin_ltm,
       nic.ni_adjustment_ratio,
       nic.net_income_positive_years,
       nic.earnings_quality_composite,
       nic.net_income_qoq_growth,
       nic.net_income_yoy_quarterly,
       nic.net_income_vs_5y_avg,
       nic.normalized_ni_vs_5y_avg,

       -- =========================================================================
       -- SECTION 17: UNUSUAL ITEMS (vw_features_unusual_items)
       -- Source: calc_unusual_items_features
       -- =========================================================================
       uif.other_unusual_items_ltm,
       uif.impairment_goodwill_ltm,
       uif.asset_writedown_ltm             AS unusual_asset_writedown_ltm,
       uif.restructuring_charges_ltm,
       uif.total_unusual_items,
       uif.unusual_items_to_revenue,
       uif.unusual_items_to_ebitda,
       uif.has_unusual_items_flag,
       uif.earnings_quality_impact,

       -- =========================================================================
       -- SECTION 18: VOLATILITY SURFACE (Enhancement 2 + 3)
       -- Source: calc_volatility_surface_features
       -- =========================================================================
       vsf.vol_1m                          AS volatility_1m,
       vsf.vol_3m                          AS volatility_3m,
       vsf.vol_6m                          AS volatility_6m,
       vsf.vol_1y                          AS volatility_1y,
       vsf.vol_term_spread_short           AS volatility_trend_short,
       vsf.vol_term_spread_long            AS volatility_trend_long,
       vsf.vol_ratio_3m_1y,
       vsf.vol_hump,
       vsf.beta_2y,
       vsf.beta_term_structure,
       vsf.beta_convexity,
       vsf.realized_vs_implied_proxy,
       vsf.beta_1y - vsf.beta_2y           AS beta_short_term_shift,

       -- =========================================================================
       -- SECTION 19: TAX RATE FEATURES (Enhancement 4)
       -- Source: calc_tax_rate_features
       -- =========================================================================
       txf.effective_tax_rate_ltm,
       txf.effective_tax_rate_fy,
       txf.tax_rate_yoy_change,
       txf.tax_rate_qoq_change,
       txf.tax_rate_stability,
       txf.low_tax_flag,
       txf.tax_rate_trend_4q,

       -- =========================================================================
       -- SECTION 20: OPEX TEMPORAL (Enhancement 5)
       -- Source: calc_opex_temporal_features
       -- =========================================================================
       otf.opex_fq,
       otf.opex_ltm,
       otf.opex_fy,
       otf.opex_qoq_growth,
       otf.opex_yoy_growth,
       otf.opex_vs_revenue_trend,
       otf.sga_qoq_growth,
       otf.sga_yoy_growth,
       otf.operating_leverage_score,

       -- =========================================================================
       -- SECTION 21: ASSET SALE FEATURES (Enhancement 8)
       -- Source: calc_asset_sale_features
       -- =========================================================================
       asf.gain_loss_on_sale_of_assets_ltm AS asset_sale_gain_loss_ltm,
       asf.asset_sale_frequency,
       asf.asset_sale_trend,

       -- =========================================================================
       -- SECTION 22: FCF ESTIMATE CURVE (Enhancement 9)
       -- Source: calc_fcf_estimate_features
       -- =========================================================================
       fcfe.fcf_est_avg_fy1e,
       fcfe.fcf_est_avg_fy2e,
       fcfe.fcf_est_avg_fy3e,
       fcfe.fcf_est_avg_fy4e,
       fcfe.fcf_est_avg_fy5e,
       fcfe.fcf_est_cagr_5y,
       fcfe.fcf_est_trend,

       -- =========================================================================
       -- SECTION 23: DIVIDEND HISTORY (Enhancement 10)
       -- Source: calc_dividend_history_features
       -- =========================================================================
       dhf.div_yield_2fy                   AS div_yield_2fyind,
       dhf.div_yield_3fy                   AS div_yield_3fyind,
       dhf.div_yield_4fy                   AS div_yield_4fyind,
       dhf.div_yield_5fy                   AS div_yield_5fyind,
       dhf.div_yield_trend_3y              AS div_yield_5y_trend,
       dhf.div_yield_volatility            AS div_yield_stability,

       -- =========================================================================
       -- SECTION 24: INVESTMENT INCOME TEMPORAL (Enhancement 11)
       -- Source: calc_investment_income_temporal
       -- =========================================================================
       iit.inv_income_fq                   AS interest_income_fq,
       iit.inv_income_fy                   AS interest_income_fy,
       iit.inv_income_qoq_growth           AS interest_income_qoq_growth,
       iit.inv_income_yoy_growth           AS interest_income_yoy_growth,
       iit.inv_income_to_revenue           AS interest_income_to_revenue_trend,

       -- =========================================================================
       -- SECTION 25: SHARE DILUTION TRACKING (Enhancement 12)
       -- Source: calc_share_dilution_tracking
       -- =========================================================================
       sdt.shares_yoy_change_pct,
       sdt.net_buyback_flag,

       -- =========================================================================
       -- SECTION 26: FORWARD CONSENSUS (Enhancement 7 supplement)
       -- Source: calc_forward_consensus_features
       -- =========================================================================
       fcnf.pe_ntm,
       fcnf.pe_est_fy1,
       fcnf.pe_forward_discount,
       fcnf.eps_gaap_vs_norm_ntm,
       fcnf.eps_gaap_vs_norm_fy1e,
       fcnf.forward_adjustment_trend,
       fcnf.ebitda_est_ntm,
       fcnf.ebitda_est_fy1e,
       fcnf.ev_ebitda_est_fy1,
       fcnf.ebitda_forward_growth,
       fcnf.earnings_revision_divergence,
       fcnf.forward_pe_vs_sector_proxy,

       -- =========================================================================
       -- METADATA: Timestamp for refresh tracking
       -- =========================================================================
       CURRENT_TIMESTAMP                   AS feature_calculated_at

FROM vw_identifier_columns id
-- Base equities for reference columns
         JOIN postgres.public.equities e ON id.isin = e."ISIN"

-- Section 1: Valuation Ratios
         LEFT JOIN calc_valuation_features() vf ON id.isin = vf.isin
         LEFT JOIN calc_valuation_timeseries_features() vts ON id.isin = vts.isin
         LEFT JOIN calc_extended_valuation_timeseries() evt ON id.isin = evt.isin
         LEFT JOIN calc_tangible_book_features() tb ON id.isin = tb.isin

-- Section 2: Momentum
         LEFT JOIN calc_momentum_features() mf ON id.isin = mf.isin
         LEFT JOIN calc_long_term_momentum_features() ltm ON id.isin = ltm.isin

-- Section 3: Technical Analysis
         LEFT JOIN calc_technical_analysis_features() ta ON id.isin = ta.isin

-- Section 4: Profitability
         LEFT JOIN calc_profitability_features() pf ON id.isin = pf.isin
         LEFT JOIN calc_margin_trends() mt ON id.isin = mt.isin
         LEFT JOIN calc_ebit_ebitda_comprehensive() eec ON id.isin = eec.isin
         LEFT JOIN calc_gross_profit_temporal() gpt ON id.isin = gpt.isin

-- Section 5: Earnings
         LEFT JOIN calc_earnings_features() ef ON id.isin = ef.isin
         LEFT JOIN calc_eps_trajectory_features() etf ON id.isin = etf.isin
         LEFT JOIN calc_eps_comprehensive() ec ON id.isin = ec.isin
         LEFT JOIN calc_eps_continuing_features() ecf ON id.isin = ecf.isin
         LEFT JOIN calc_gaap_adjusted_analytics() gaa ON id.isin = gaa.isin
         LEFT JOIN calc_gaap_revision_features() grf ON id.isin = grf.isin

-- Section 6: Growth
         LEFT JOIN calc_growth_features() gf ON id.isin = gf.isin
         LEFT JOIN calc_revenue_forecast_features() rff ON id.isin = rff.isin
         LEFT JOIN calc_revenue_estimate_consensus() rec ON id.isin = rec.isin
         LEFT JOIN calc_revenue_quarterly_features() rqf ON id.isin = rqf.isin
         LEFT JOIN calc_total_revenues_temporal() trt ON id.isin = trt.isin

-- Section 7: Quality & Risk
         LEFT JOIN calc_quality_features() qf ON id.isin = qf.isin
         LEFT JOIN calc_beta_risk_features() br ON id.isin = br.isin
         LEFT JOIN calc_financial_distress_features() fdf ON id.isin = fdf.isin
         LEFT JOIN calc_accounting_quality_features() aqf ON id.isin = aqf.isin
         LEFT JOIN calc_quality_features_comprehensive() qfc ON id.isin = qfc.isin

-- Section 8: Leverage & Liquidity
         LEFT JOIN calc_leverage_features() lf ON id.isin = lf.isin
         LEFT JOIN calc_efficiency_ratios() er ON id.isin = er.isin
         LEFT JOIN calc_balance_sheet_dynamics() bsd ON id.isin = bsd.isin
         LEFT JOIN calc_working_capital_temporal() wct ON id.isin = wct.isin
         LEFT JOIN calc_total_debt_temporal() tdt ON id.isin = tdt.isin
         LEFT JOIN calc_working_capital_deep_features() wcd ON id.isin = wcd.isin

-- Section 9: Analyst Sentiment
         LEFT JOIN calc_sentiment_features() sf ON id.isin = sf.isin
         LEFT JOIN calc_price_target_dynamics() ptd ON id.isin = ptd.isin

-- Section 10: Dividends
         LEFT JOIN calc_dividend_features() df ON id.isin = df.isin
         LEFT JOIN calc_dividend_timing() dt ON id.isin = dt.isin
         LEFT JOIN calc_dividend_yield_comprehensive() dyc ON id.isin = dyc.isin

-- Section 11: Employment
         LEFT JOIN calc_employment_features() emf ON id.isin = emf.isin
         LEFT JOIN calc_employment_dynamics() ed ON id.isin = ed.isin

-- Section 12: Cash Flow
         LEFT JOIN calc_cashflow_features() cf ON id.isin = cf.isin
         LEFT JOIN calc_enhanced_cashflow_features() ecff ON id.isin = ecff.isin
         LEFT JOIN calc_cashflow_temporal_features() ctf ON id.isin = ctf.isin
         LEFT JOIN calc_cashflow_comprehensive() cc ON id.isin = cc.isin

-- Section 13: Temporal
         LEFT JOIN calc_temporal_features() tf ON id.isin = tf.isin
         LEFT JOIN calc_fiscal_calendar_features() fcf ON id.isin = fcf.isin

-- Section 14: Balance Sheet
         LEFT JOIN calc_total_assets_temporal() tat ON id.isin = tat.isin
         LEFT JOIN calc_inventory_temporal_features() itf ON id.isin = itf.isin
         LEFT JOIN calc_goodwill_temporal_features() gtf ON id.isin = gtf.isin

-- Section 15: Cost Structure
         LEFT JOIN calc_cost_structure_features() csf ON id.isin = csf.isin
         LEFT JOIN calc_rnd_temporal_features() rtf ON id.isin = rtf.isin
         LEFT JOIN calc_interest_income_features() iif ON id.isin = iif.isin

-- Section 16: Composite Scores
         LEFT JOIN calc_composite_scores() cs ON id.isin = cs.isin
         LEFT JOIN calc_net_income_comprehensive() nic ON id.isin = nic.isin

-- Section 17: Unusual Items
         LEFT JOIN calc_unusual_items_features() uif ON id.isin = uif.isin

-- Section 18: Volatility Surface (Enhancement 2 + 3)
         LEFT JOIN calc_volatility_surface_features() vsf ON id.isin = vsf.isin

-- Section 19: Tax Rate Features (Enhancement 4)
         LEFT JOIN calc_tax_rate_features() txf ON id.isin = txf.isin

-- Section 20: OpEx Temporal (Enhancement 5)
         LEFT JOIN calc_opex_temporal_features() otf ON id.isin = otf.isin

-- Section 21: Asset Sale Features (Enhancement 8)
         LEFT JOIN calc_asset_sale_features() asf ON id.isin = asf.isin

-- Section 22: FCF Estimate Curve (Enhancement 9)
         LEFT JOIN calc_fcf_estimate_features() fcfe ON id.isin = fcfe.isin

-- Section 23: Dividend History (Enhancement 10)
         LEFT JOIN calc_dividend_history_features() dhf ON id.isin = dhf.isin

-- Section 24: Investment Income Temporal (Enhancement 11)
         LEFT JOIN calc_investment_income_temporal() iit ON id.isin = iit.isin

-- Section 25: Share Dilution Tracking (Enhancement 12)
         LEFT JOIN calc_share_dilution_tracking() sdt ON id.isin = sdt.isin

-- Section 26: Forward Consensus (Enhancement 7 supplement)
         LEFT JOIN calc_forward_consensus_features() fcnf ON id.isin = fcnf.isin;

-- =============================================================================
-- INDEXES FOR OPTIMIZED QUERYING
-- =============================================================================
CREATE UNIQUE INDEX idx_mv_all_stock_features_isin
    ON mv_all_stock_features (isin);

CREATE INDEX idx_mv_all_stock_features_ticker
    ON mv_all_stock_features (ticker);

CREATE INDEX idx_mv_all_stock_features_sector_industry
    ON mv_all_stock_features (sector, industry);

CREATE INDEX idx_mv_all_stock_features_region_country
    ON mv_all_stock_features (region, country, trading_country);

CREATE INDEX idx_mv_all_stock_features_exchange
    ON mv_all_stock_features (exchange);

CREATE INDEX idx_mv_all_stock_features_market_cap
    ON mv_all_stock_features (market_cap DESC NULLS LAST);

-- =============================================================================
-- COMMENT ON MATERIALIZED VIEW
-- =============================================================================
COMMENT ON MATERIALIZED VIEW mv_all_stock_features IS
    'Unified materialized view containing all calculated stock features.
    Covers 26 feature categories from 63 calc_* functions:
    1. Valuation Ratios (4 functions)
    2. Momentum (2 functions)
    3. Technical Analysis (1 function)
    4. Profitability (4 functions)
    5. Earnings (6 functions)
    6. Growth (5 functions)
    7. Quality & Risk (5 functions)
    8. Leverage & Liquidity (6 functions)
    9. Analyst Sentiment (2 functions)
    10. Dividends (3 functions)
    11. Employment (2 functions)
    12. Cash Flow (4 functions)
    13. Temporal (2 functions)
    14. Balance Sheet (3 functions)
    15. Cost Structure (3 functions)
    16. Composite Scores (2 functions)
    17. Unusual Items (1 function)
    18. Volatility Surface (1 function) - Enhancement 2+3
    19. Tax Rate Features (1 function) - Enhancement 4
    20. OpEx Temporal (1 function) - Enhancement 5
    21. Asset Sale Features (1 function) - Enhancement 8
    22. FCF Estimate Curve (1 function) - Enhancement 9
    23. Dividend History (1 function) - Enhancement 10
    24. Investment Income Temporal (1 function) - Enhancement 11
    25. Share Dilution Tracking (1 function) - Enhancement 12
    26. Forward Consensus (1 function) - Enhancement 7

    Direct reference columns include: Enhancement 1 (17 cols), Enhancement 6 (6 cols), Enhancement 7 (4 cols)

    Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY mv_all_stock_features;';

-- =============================================================================
-- REFRESH FUNCTION
-- =============================================================================
CREATE OR REPLACE FUNCTION refresh_all_stock_features()
    RETURNS void
    LANGUAGE plpgsql
AS
$$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_all_stock_features;
    RAISE NOTICE 'mv_all_stock_features refreshed at %', NOW();
END;
$$;

COMMENT ON FUNCTION refresh_all_stock_features() IS
    'Refreshes the mv_all_stock_features materialized view concurrently (non-blocking).
    Call periodically after equities table updates.';

-- =============================================================================
-- SECTION 2: FEATURE REGISTRY METADATA TABLE (ENHANCED)
-- =============================================================================

-- Create a metadata table documenting available SQL feature functions
CREATE TABLE IF NOT EXISTS feature_registry_metadata
(
    function_name     VARCHAR(128) PRIMARY KEY,
    category          VARCHAR(64) NOT NULL,
    feature_count     SMALLINT,
    description       TEXT,
    python_equivalent VARCHAR(128),
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add index for category-based queries
CREATE INDEX IF NOT EXISTS idx_feature_registry_category
    ON feature_registry_metadata (category);

-- Add index for python_equivalent lookups
CREATE INDEX IF NOT EXISTS idx_feature_registry_python_equiv
    ON feature_registry_metadata (python_equivalent);

-- Ensure unique constraint exists on function_name for ON CONFLICT upsert support.
CREATE UNIQUE INDEX IF NOT EXISTS idx_uq_feature_registry_function_name
    ON feature_registry_metadata (function_name);

-- =============================================================================
-- CALCULATED FEATURES REGISTRY TABLE
-- Maps all calculated features from mv_all_stock_features to source columns
-- =============================================================================

CREATE TABLE IF NOT EXISTS calculated_features_registry
(
    feature_key        VARCHAR(128) PRIMARY KEY,
    feature_alias      VARCHAR(128) NOT NULL,
    category           VARCHAR(64)  NOT NULL,
    source_function    VARCHAR(128),
    description        TEXT,
    source_columns     TEXT[],      -- Array of source column references
    primary_source_col TEXT         -- Primary source column FK reference
        REFERENCES equities_schema_metadata (column_name),
    calculation_type   VARCHAR(32), -- 'ratio', 'difference', 'growth', 'flag', 'score', 'direct'
    data_type          VARCHAR(32),
    updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_calc_features_category ON calculated_features_registry (category);
CREATE INDEX IF NOT EXISTS idx_calc_features_source_fn ON calculated_features_registry (source_function);
CREATE INDEX IF NOT EXISTS idx_calc_features_primary_col ON calculated_features_registry (primary_source_col);

-- Ensure unique constraint exists on feature_key for ON CONFLICT upsert support.
-- CREATE TABLE IF NOT EXISTS won't add the PRIMARY KEY to a pre-existing table,
-- so we create a unique index unconditionally (IF NOT EXISTS makes it idempotent).
CREATE UNIQUE INDEX IF NOT EXISTS idx_uq_calc_features_feature_key
    ON calculated_features_registry (feature_key);

-- Wrap upsert in transaction for atomicity
BEGIN;

-- Upsert all function metadata
INSERT INTO feature_registry_metadata (function_name, category, feature_count, description, python_equivalent,
                                       updated_at)
VALUES
    -- Helper Functions
    ('safe_divide', 'Helper Functions', 1, 'Safe division avoiding division by zero', NULL, CURRENT_TIMESTAMP),
    ('pct_change', 'Helper Functions', 1, 'Percentage change calculation', NULL, CURRENT_TIMESTAMP),
    ('calc_change_ratio', 'Helper Functions', 1, 'Change ratio without percentage multiplier', NULL, CURRENT_TIMESTAMP),
    ('clamp_score', 'Helper Functions', 1, 'Score clamping between min and max values', NULL, CURRENT_TIMESTAMP),
    ('ema_crossover_signal', 'Helper Functions', 1, 'EMA crossover signal detection', NULL, CURRENT_TIMESTAMP),

    -- Valuation Functions
    ('calc_valuation_features', 'Valuation Ratios', 6, 'P/E, P/B, EV/EBITDA, EV/Sales, dividend yield, PEG ratio',
     'engineer_valuation_ratios', CURRENT_TIMESTAMP),
    ('calc_valuation_timeseries_features', 'Valuation Timeseries', 11,
     'Valuation momentum, mean reversion, forward discount', 'engineer_valuation_timeseries_features',
     CURRENT_TIMESTAMP),
    ('calc_extended_valuation_timeseries', 'Valuation Timeseries', 11,
     'QoQ multiple trends, mean reversion, P/B momentum', 'engineer_valuation_timeseries_features', CURRENT_TIMESTAMP),

    -- Technical Analysis Functions
    ('calc_momentum_features', 'Technical Analysis', 14,
     'Price momentum, EMA crossovers, 52W range, beta, volatility', 'engineer_momentum_features', CURRENT_TIMESTAMP),
    ('calc_technical_analysis_features', 'Technical Analysis', 11,
     'EMA trends, breakout signals, volume momentum, volatility compression', 'engineer_technical_analysis_features',
     CURRENT_TIMESTAMP),
    ('calc_long_term_momentum_features', 'Technical Analysis', 7,
     '3Y/5Y momentum, weighted trend score, secular trend flag', 'engineer_long_term_momentum_features',
     CURRENT_TIMESTAMP),

    -- Profitability Functions
    ('calc_profitability_features', 'Profitability', 9, 'ROE, ROA, margins, ROIC, DuPont components',
     'engineer_profitability_ratios', CURRENT_TIMESTAMP),
    ('calc_margin_trends', 'Profitability', 6, 'Gross, operating, net, EBITDA margin trends, expansion flag',
     'engineer_margin_trends', CURRENT_TIMESTAMP),
    ('calc_ebit_ebitda_comprehensive', 'Profitability', 42, 'EBIT/EBITDA for all periods, growth and margins',
     'engineer_margin_trends', CURRENT_TIMESTAMP),
    ('calc_total_revenues_temporal', 'Growth Metrics', 12,
     'Comprehensive revenue trends across FQ, LTM, FY and 5Y averages',
     'engineer_growth_metrics', CURRENT_TIMESTAMP),

    -- Quality & Risk Functions
    ('calc_quality_features', 'Quality & Risk', 10, 'Impairments, goodwill, Z-score, liquidity ratios',
     'engineer_accounting_quality_features', CURRENT_TIMESTAMP),
    ('calc_financial_distress_features', 'Financial Distress', 9, 'Distress risk score, liquidity stress, cash runway',
     'engineer_financial_distress_features', CURRENT_TIMESTAMP),
    ('calc_accounting_quality_features', 'Accounting Quality', 7,
     'Goodwill changes, restructuring, exceptional items, quality score', 'engineer_accounting_quality_features',
     CURRENT_TIMESTAMP),
    ('calc_quality_features_comprehensive', 'Accounting Quality', 11,
     'Detailed impairments, writedowns, restructuring across periods', 'engineer_accounting_quality_features',
     CURRENT_TIMESTAMP),
    ('calc_beta_risk_features', 'Quality & Risk', 7, 'Multi-period betas, trend analysis, stability score',
     'engineer_beta_risk_features', CURRENT_TIMESTAMP),
    ('calc_working_capital_temporal', 'Leverage & Liquidity', 21, 'Full historical coverage of working capital trends',
     'engineer_working_capital_deep_features', CURRENT_TIMESTAMP),
    ('calc_total_debt_temporal', 'Leverage & Liquidity', 18,
     'Leverage trend analysis with quarterly and yearly historical data',
     'engineer_leverage_ratios', CURRENT_TIMESTAMP),
    ('calc_total_assets_temporal', 'Balance Sheet', 17, 'Balance sheet dynamics with full historical asset coverage',
     'engineer_balance_sheet_trends', CURRENT_TIMESTAMP),

    -- Leverage & Liquidity Functions
    ('calc_leverage_features', 'Leverage & Liquidity', 7, 'Debt ratios, coverage, working capital ratio',
     'engineer_leverage_ratios', CURRENT_TIMESTAMP),
    ('calc_efficiency_ratios', 'Efficiency Ratios', 4, 'Asset and inventory turnover, receivables days',
     'engineer_efficiency_ratios', CURRENT_TIMESTAMP),
    ('calc_balance_sheet_dynamics', 'Balance Sheet', 13, 'Cash trends, inventory vs 5Y avg, asset quality, BS strength',
     'engineer_balance_sheet_trends', CURRENT_TIMESTAMP),
    ('calc_working_capital_deep_features', 'Leverage & Liquidity', 11,
     'Working capital ratios, trends, efficiency score', 'engineer_working_capital_deep_features', CURRENT_TIMESTAMP),

    -- Analyst Sentiment Functions
    ('calc_sentiment_features', 'Analyst Sentiment', 10, 'Ratings, price targets, revisions, coverage quality',
     'engineer_analyst_quality_features', CURRENT_TIMESTAMP),
    ('calc_price_target_dynamics', 'Price Target Dynamics', 15,
     'PT momentum (1W-1Y), consensus convergence, coverage changes', 'engineer_price_target_dynamics',
     CURRENT_TIMESTAMP),

    -- Earnings Functions
    ('calc_earnings_features', 'Earnings Quality', 7, 'Surprises, adjustments, GAAP vs non-GAAP',
     'engineer_estimated_vs_actual_analytics', CURRENT_TIMESTAMP),
    ('calc_eps_trajectory_features', 'EPS Trajectory', 10, 'EPS growth rates, CAGR, positive streak, trajectory score',
     'engineer_eps_trajectory_features', CURRENT_TIMESTAMP),
    ('calc_gaap_adjusted_analytics', 'GAAP vs Adjusted', 48,
     'Comprehensive GAAP/Adjusted analytics for EPS, Net Income, EBITDA, EBIT across all periods (LTM, FY, FQ, -1FY to -4FY, -1FQFQ to -4FQFQ, 5YAVGFQ) with quality scores',
     'engineer_gaap_vs_adjusted_analytics', CURRENT_TIMESTAMP),
    ('calc_gaap_revision_features', 'Earnings Quality', 9, 'GAAP EPS revision momentum, spread vs normalized',
     'engineer_gaap_vs_adjusted_analytics', CURRENT_TIMESTAMP),
    ('calc_eps_comprehensive', 'Earnings Quality', 10, 'Basic, Continuing, Adjusted EPS, growth, CAGR, trajectory',
     'engineer_eps_trajectory_features', CURRENT_TIMESTAMP),
    ('calc_net_income_comprehensive', 'Earnings Quality', 30, 'GAAP, Adjusted, Normalized NI for all periods',
     'engineer_gaap_vs_adjusted_analytics', CURRENT_TIMESTAMP),
    ('calc_gross_profit_temporal', 'Profitability', 20,
     'Gross profit margin trend analysis with full historical coverage',
     'engineer_margin_trends', CURRENT_TIMESTAMP),
    ('calc_unusual_items_features', 'Earnings Quality', 9, 'Unusual items totals, ratios, earnings quality impact',
     'engineer_unusual_items_features', CURRENT_TIMESTAMP),

    -- Growth Functions
    ('calc_growth_features', 'Growth Metrics', 7, 'Revenue, EBITDA, FCF growth rates', 'engineer_growth_metrics',
     CURRENT_TIMESTAMP),
    ('calc_revenue_forecast_features', 'Revenue Forecasting', 12, 'Estimate spread, beat potential, forward multiples',
     'engineer_revenue_forecast_features', CURRENT_TIMESTAMP),
    ('calc_revenue_estimate_consensus', 'Revenue Forecasting', 8,
     'Estimate skew, consensus confidence, upside to consensus', 'engineer_revenue_estimate_consensus',
     CURRENT_TIMESTAMP),
    ('calc_revenue_quarterly_features', 'Revenue Forecasting', 32,
     'Quarterly revenue trends with full historical coverage (-1FQFQ to -4FQFQ, -1FY to -4FY), QoQ/YoY momentum, CAGR, trend analysis, and growth flags',
     'engineer_revenue_quarterly_features', CURRENT_TIMESTAMP),

    -- Dividend Functions
    ('calc_dividend_features', 'Dividend Reliability', 8, 'Streak, yield, payout, coverage, shareholder yield',
     'engineer_dividend_reliability_features', CURRENT_TIMESTAMP),
    ('calc_dividend_timing', 'Dividend Reliability', 8, 'Days to ex-date/payment, consistency, yield vs 5Y avg',
     'engineer_dividend_timing_features', CURRENT_TIMESTAMP),
    ('calc_dividend_yield_comprehensive', 'Dividend Reliability', 10,
     'Comprehensive dividend yields, sustainability flag', 'engineer_dividend_reliability_features', CURRENT_TIMESTAMP),

    -- Employment Functions
    ('calc_employment_features', 'Employee Productivity', 7, 'Per-employee metrics, FTE growth',
     'engineer_employee_productivity_features', CURRENT_TIMESTAMP),
    ('calc_employment_dynamics', 'Employment Dynamics', 10, 'FTE growth, acceleration, hiring intensity',
     'engineer_employment_dynamics_features', CURRENT_TIMESTAMP),

    -- Cash Flow Functions
    ('calc_cashflow_features', 'Cash Flow', 7, 'CFO/NI, FCF margin, self-funding ratio',
     'engineer_cash_flow_quality_features', CURRENT_TIMESTAMP),
    ('calc_enhanced_cashflow_features', 'Cash flow', 12,
     'FCF consistency, CapEx efficiency, M&A sustainability', 'engineer_cash_flow_quality_features', CURRENT_TIMESTAMP),
    ('calc_cashflow_temporal_features', 'Cash Flow', 12, 'Quarterly CF trends, burn rate, volatility, momentum',
     'engineer_cashflow_temporal_features', CURRENT_TIMESTAMP),
    ('calc_cashflow_comprehensive', 'Cash Flow', 14, 'CFO, CFI, CFF, FCF for all periods, quality score',
     'engineer_cash_flow_quality_features', CURRENT_TIMESTAMP),

    -- NEW: FCF Growth Estimates
    ('calc_fcf_growth_estimates', 'Cash Flow', 20,
     'Estimated FCF growth rates from consensus forecasts (FY1E-FY5E), CAGRs, forward margins/yields, growth acceleration, trajectory quality',
     'engineer_fcf_growth_estimate_features', CURRENT_TIMESTAMP),

    -- Temporal Functions
    ('calc_temporal_features', 'Temporal Patterns', 7, 'Fiscal calendar, earnings timing', 'engineer_temporal_features',
     CURRENT_TIMESTAMP),
    ('calc_fiscal_calendar_features', 'Temporal Patterns', 9, 'Days since report, quarter/FY flags, freshness score',
     'engineer_fiscal_calendar_features', CURRENT_TIMESTAMP),

    -- Composite Functions
    ('calc_composite_scores', 'Composite Scores', 4, 'Piotroski F-Score, EPS trajectory, dilution, quality-momentum',
     'engineer_composite_scores', CURRENT_TIMESTAMP),
    ('calc_all_enhanced_features', 'Composite', 3, 'Aggregation metadata of all enhanced features',
     'engineer_all_enhanced_features', CURRENT_TIMESTAMP),

    -- Cost Structure Functions (ENHANCED with Marketing and SG&A 5Y metrics)
    ('calc_cost_structure_features', 'Efficiency Ratios', 13,
     'COGS, SG&A, R&D ratios, operating leverage, marketing efficiency, SG&A vs 5Y avg',
     'engineer_cost_structure_features', CURRENT_TIMESTAMP),

    -- Interest Income Functions
    ('calc_interest_income_features', 'Interest Income', 7, 'Net interest income, coverage ratios, income quality',
     'engineer_interest_income_features', CURRENT_TIMESTAMP),

    -- Tangible Book Functions (ENHANCED with native TBV columns)
    ('calc_tangible_book_features', 'Valuation Ratios', 10,
     'Native TBV (FY/LTM), Price-to-TBV, TBV per share, tangible equity ratio, TBV growth, validation',
     'engineer_tangible_book_features', CURRENT_TIMESTAMP),

    -- NEW Functions
    ('calc_eps_continuing_features', 'Earnings Quality', 18,
     'EPS from continuing operations analysis with discontinued ops impact, trajectory, stability',
     'engineer_eps_continuing_features', CURRENT_TIMESTAMP),

    ('calc_inventory_temporal_features', 'Balance Sheet', 21,
     'Full inventory temporal coverage with efficiency and quality metrics, buildup/reduction flags',
     'engineer_inventory_temporal_features', CURRENT_TIMESTAMP),

    ('calc_goodwill_temporal_features', 'Accounting Quality', 19,
     'M&A activity tracking through goodwill changes, impairment risk, accumulation rate',
     'engineer_goodwill_temporal_features', CURRENT_TIMESTAMP),

    ('calc_rnd_temporal_features', 'Efficiency Ratios', 21,
     'R&D investment trends, intensity, efficiency metrics, ROI proxy, investment flags',
     'engineer_rnd_temporal_features', CURRENT_TIMESTAMP),

    ('calc_volatility_surface_features', 'Technical Analysis', 15,
     'Multi-period volatility spreads, beta term structure, convexity, and realized vs implied proxy',
     'engineer_momentum_features', CURRENT_TIMESTAMP),

    ('calc_forward_consensus_features', 'Analyst Sentiment', 13,
     'Forward P/E discounts, GAAP vs non-GAAP consensus spreads, EBITDA growth expectations, and revision divergence',
     'engineer_analyst_quality_features', CURRENT_TIMESTAMP),

    ('calc_price_target_achievement_features', 'Analyst Sentiment', 8,
     'Historical price target accuracy, optimism bias, range hit rates, and analyst count stability',
     'engineer_analyst_quality_features', CURRENT_TIMESTAMP),

    ('calc_dividend_history_features', 'Dividend Reliability', 10,
     'Long-term dividend yield trends, volatility, declining flags, and mean reversion metrics',
     'engineer_dividend_reliability_features', CURRENT_TIMESTAMP),

    ('calc_size_liquidity_features', 'Technical Analysis', 11,
     'Market cap dynamics, log size, relative volume, turnover ratios, and composite liquidity scores',
     'engineer_momentum_features', CURRENT_TIMESTAMP),

    ('calc_investment_income_temporal', 'Interest Income', 10,
     'Temporal analysis of interest and investment income, growth rates, revenue contribution, and positive streaks',
     'engineer_interest_income_features', CURRENT_TIMESTAMP),

    ('calc_tax_rate_features', 'Accounting Quality', 8,
     'Effective tax rate trends, stability metrics, low tax flags, and quarterly divergence',
     'engineer_accounting_quality_features', CURRENT_TIMESTAMP),

    ('calc_opex_temporal_features', 'Efficiency Ratios', 10,
     'Operating expense growth, SG&A trends, opex-to-revenue dynamics, and operating leverage scores',
     'engineer_cost_structure_features', CURRENT_TIMESTAMP),

    ('calc_fcf_estimate_features', 'Cash Flow', 8,
     'Forward-looking FCF estimates (FY1E-FY5E), implied CAGRs, and linear growth trends',
     'engineer_fcf_growth_estimate_features', CURRENT_TIMESTAMP),

    ('calc_asset_sale_features', 'Accounting Quality', 4,
     'Gains/losses on asset sales, frequency analysis, and temporal trends',
     'engineer_accounting_quality_features', CURRENT_TIMESTAMP),

    ('calc_share_dilution_tracking', 'Quality & Risk', 4,
     'Share count changes, YoY dilution/accretion, and net buyback activity flags',
     'engineer_leverage_ratios', CURRENT_TIMESTAMP),

    -- Materialized View (UPDATED feature count)
    ('mv_all_stock_features', 'Materialized View', 887,
     'Unified materialized view containing all calculated features from 56 calc_* functions across 29 categories',
     'all_features_combined', CURRENT_TIMESTAMP)

ON CONFLICT (function_name) DO UPDATE SET category          = EXCLUDED.category,
                                          feature_count     = EXCLUDED.feature_count,
                                          description       = EXCLUDED.description,
                                          python_equivalent = EXCLUDED.python_equivalent,
                                          updated_at        = CURRENT_TIMESTAMP;

-- =============================================================================
-- INSERT CALCULATED FEATURES INTO REGISTRY
-- =============================================================================

INSERT INTO calculated_features_registry (feature_key, feature_alias, category, source_function, description,
                                          source_columns, primary_source_col, calculation_type, data_type, updated_at)
VALUES
    -- IDENTIFIERS (Direct mappings)
    ('feat_isin', 'isin', 'Identifier', NULL, 'International Securities Identification Number', ARRAY ['ISIN'], 'ISIN',
     'direct', 'TEXT', CURRENT_TIMESTAMP),
    ('feat_ticker', 'ticker', 'Identifier', NULL, 'Stock ticker symbol', ARRAY ['Ticker'], 'Ticker', 'direct', 'TEXT',
     CURRENT_TIMESTAMP),
    ('feat_name', 'name', 'Identifier', NULL, 'Company name', ARRAY ['Name'], 'Name', 'direct', 'TEXT',
     CURRENT_TIMESTAMP),
    ('feat_region', 'region', 'Identifier', NULL, 'Geographic region', ARRAY ['Region'], 'Region', 'direct', 'TEXT',
     CURRENT_TIMESTAMP),
    ('feat_country', 'country', 'Identifier', NULL, 'Country of incorporation', ARRAY ['Country'], 'Country', 'direct',
     'TEXT', CURRENT_TIMESTAMP),
    ('feat_trading_country', 'trading_country', 'Identifier', NULL, 'Primary trading country',
     ARRAY ['Trading Country'], 'Trading Country', 'direct', 'TEXT', CURRENT_TIMESTAMP),
    ('feat_exchange', 'exchange', 'Identifier', NULL, 'Stock exchange', ARRAY ['Exchange'], 'Exchange', 'direct',
     'TEXT', CURRENT_TIMESTAMP),
    ('feat_sector', 'sector', 'Identifier', NULL, 'Business sector', ARRAY ['Sector'], 'Sector', 'direct', 'TEXT',
     CURRENT_TIMESTAMP),
    ('feat_industry', 'industry', 'Identifier', NULL, 'Industry classification', ARRAY ['Industry'], 'Industry',
     'direct', 'TEXT', CURRENT_TIMESTAMP),

    -- VALUATION FEATURES
    ('feat_p_e_ratio', 'p_e_ratio', 'Valuation Ratios', 'calc_valuation_features', 'Price-to-Earnings ratio (LTM)',
     ARRAY ['P/E (LTM)'], 'P/E (LTM)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_p_b_ratio', 'p_b_ratio', 'Valuation Ratios', 'calc_valuation_features', 'Price-to-Book ratio (LTM)',
     ARRAY ['P/B (LTM)'], 'P/B (LTM)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ev_ebitda_ratio', 'ev_ebitda_ratio', 'Valuation Ratios', 'calc_valuation_features',
     'Enterprise Value to EBITDA ratio', ARRAY ['EV/EBITDA (LTM)'], 'EV/EBITDA (LTM)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_ev_sales_ratio', 'ev_sales_ratio', 'Valuation Ratios', 'calc_valuation_features',
     'Enterprise Value to Sales ratio', ARRAY ['EV/Sales (LTM)'], 'EV/Sales (LTM)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_dividend_yield', 'valuation_dividend_yield', 'Valuation Ratios', 'calc_valuation_features',
     'Dividend yield (LTM)',
     ARRAY ['Div Yield (LTM)'], 'Div Yield (LTM)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_peg_ratio', 'peg_ratio', 'Valuation Ratios', 'calc_valuation_features',
     'Price/Earnings to EPS CAGR growth proxy (3Y)',
     ARRAY ['P/E (LTM)', 'Net EPS - Basic (FY)', 'Net EPS - Basic (-3FY)'], 'P/E (LTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),

    -- VALUATION TIMESERIES
    ('feat_ev_sales_trend_1y', 'ev_sales_trend_1y', 'Valuation Timeseries', 'calc_valuation_timeseries_features',
     'EV/Sales 1-year trend', ARRAY ['EV/Sales (LTM)', 'EV/Sales (-1FYLTM)'], 'EV/Sales (LTM)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_ev_ebitda_momentum', 'ev_ebitda_momentum', 'Valuation Timeseries', 'calc_valuation_timeseries_features',
     'EV/EBITDA momentum', ARRAY ['EV/EBITDA (LTM)', 'EV/EBITDA (-1FYLTM)'], 'EV/EBITDA (LTM)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_p_e_momentum_yoy', 'p_e_momentum_yoy', 'Valuation Timeseries', 'calc_valuation_timeseries_features',
     'P/E year-over-year momentum', ARRAY ['P/E (LTM)', 'P/E (-1FYLTM)'], 'P/E (LTM)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_p_e_momentum_qoq', 'p_e_momentum_qoq', 'Valuation Timeseries', 'calc_valuation_timeseries_features',
     'P/E quarter-over-quarter momentum', ARRAY ['P/E (LTM)', 'P/E (-1FQLTM)'], 'P/E (LTM)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_ev_sales_vs_3y_avg', 'ev_sales_vs_3y_avg', 'Valuation Timeseries', 'calc_valuation_timeseries_features',
     'EV/Sales vs 3-year average', ARRAY ['EV/Sales (LTM)', 'EV/Sales (3YAVGLTM)'], 'EV/Sales (LTM)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ev_ebitda_vs_3y_avg', 'ev_ebitda_vs_3y_avg', 'Valuation Timeseries', 'calc_valuation_timeseries_features',
     'EV/EBITDA vs 3-year average', ARRAY ['EV/EBITDA (LTM)', 'EV/EBITDA (3YAVGLTM)'], 'EV/EBITDA (LTM)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),

    -- INVESTMENT INCOME TEMPORAL FEATURES
    ('feat_inv_income_ltm', 'inv_income_ltm', 'Interest Income', 'calc_investment_income_temporal',
     'Interest and Investment Income (LTM)', ARRAY ['Interest And Investment Income (LTM)'],
     'Interest And Investment Income (LTM)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inv_income_fq', 'interest_income_fq', 'Interest Income', 'calc_investment_income_temporal',
     'Interest and Investment Income (FQ)', ARRAY ['Interest And Investment Income (FQ)'],
     'Interest And Investment Income (FQ)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inv_income_fy', 'interest_income_fy', 'Interest Income', 'calc_investment_income_temporal',
     'Interest and Investment Income (FY)', ARRAY ['Interest And Investment Income (FY)'],
     'Interest And Investment Income (FY)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inv_income_qoq_growth', 'interest_income_qoq_growth', 'Interest Income', 'calc_investment_income_temporal',
     'Interest and Investment Income QoQ growth',
     ARRAY ['Interest And Investment Income (FQ)', 'Interest And Investment Income (-1FQFQ)'],
     'Interest And Investment Income (FQ)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inv_income_yoy_growth', 'interest_income_yoy_growth', 'Interest Income', 'calc_investment_income_temporal',
     'Interest and Investment Income YoY growth',
     ARRAY ['Interest And Investment Income (FY)', 'Interest And Investment Income (-1FY)'],
     'Interest And Investment Income (FY)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inv_income_to_revenue', 'interest_income_to_revenue_trend', 'Interest Income',
     'calc_investment_income_temporal',
     'Interest and Investment Income as % of revenue',
     ARRAY ['Interest And Investment Income (LTM)', 'Total Revenues (LTM)'], 'Interest And Investment Income (LTM)',
     'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inv_income_trend_3y', 'inv_income_trend_3y', 'Interest Income', 'calc_investment_income_temporal',
     '3-year CAGR of interest and investment income',
     ARRAY ['Interest And Investment Income (FY)', 'Interest And Investment Income (-3FY)'],
     'Interest And Investment Income (FY)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inv_income_positive_quarters', 'inv_income_positive_quarters', 'Interest Income',
     'calc_investment_income_temporal', 'Number of positive interest income quarters (last 5)',
     ARRAY ['Interest And Investment Income (FQ)', 'Interest And Investment Income (-1FQFQ)', 'Interest And Investment Income (-2FQFQ)', 'Interest And Investment Income (-3FQFQ)', 'Interest And Investment Income (-4FQFQ)'],
     'Interest And Investment Income (FQ)', 'score', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_financial_company_proxy', 'financial_company_proxy', 'Interest Income', 'calc_investment_income_temporal',
     'Financial company proxy flag (income > 20% of revenue)',
     ARRAY ['Interest And Investment Income (LTM)', 'Total Revenues (LTM)'], 'Interest And Investment Income (LTM)',
     'flag', 'INTEGER', CURRENT_TIMESTAMP),

    -- TAX RATE FEATURES
    ('feat_effective_tax_rate_ltm', 'effective_tax_rate_ltm', 'Accounting Quality', 'calc_tax_rate_features',
     'Effective tax rate (LTM)', ARRAY ['Effective Tax Rate - (Ratio) (LTM)'], 'Effective Tax Rate - (Ratio) (LTM)',
     'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_effective_tax_rate_fy', 'effective_tax_rate_fy', 'Accounting Quality', 'calc_tax_rate_features',
     'Effective tax rate (FY)', ARRAY ['Effective Tax Rate - (Ratio) (FY)'], 'Effective Tax Rate - (Ratio) (FY)',
     'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_tax_rate_yoy_change', 'tax_rate_yoy_change', 'Accounting Quality', 'calc_tax_rate_features',
     'YoY change in effective tax rate',
     ARRAY ['Effective Tax Rate - (Ratio) (FY)', 'Effective Tax Rate - (Ratio) (-1FY)'],
     'Effective Tax Rate - (Ratio) (FY)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_tax_rate_qoq_change', 'tax_rate_qoq_change', 'Accounting Quality', 'calc_tax_rate_features',
     'QoQ change in effective tax rate',
     ARRAY ['Effective Tax Rate - (Ratio) (FQ)', 'Effective Tax Rate - (Ratio) (-1FQFQ)'],
     'Effective Tax Rate - (Ratio) (FQ)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_tax_rate_stability', 'tax_rate_stability', 'Accounting Quality', 'calc_tax_rate_features',
     'Tax rate stability (quarterly range)',
     ARRAY ['Effective Tax Rate - (Ratio) (FQ)', 'Effective Tax Rate - (Ratio) (-1FQFQ)', 'Effective Tax Rate - (Ratio) (-2FQFQ)', 'Effective Tax Rate - (Ratio) (-3FQFQ)'],
     'Effective Tax Rate - (Ratio) (FQ)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_low_tax_flag', 'low_tax_flag', 'Accounting Quality', 'calc_tax_rate_features',
     'Low tax rate flag (<10%)', ARRAY ['Effective Tax Rate - (Ratio) (LTM)'], 'Effective Tax Rate - (Ratio) (LTM)',
     'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_tax_rate_trend_4q', 'tax_rate_trend_4q', 'Accounting Quality', 'calc_tax_rate_features',
     'Tax rate trend vs prior 3-quarter average',
     ARRAY ['Effective Tax Rate - (Ratio) (FQ)', 'Effective Tax Rate - (Ratio) (-1FQFQ)', 'Effective Tax Rate - (Ratio) (-2FQFQ)', 'Effective Tax Rate - (Ratio) (-3FQFQ)'],
     'Effective Tax Rate - (Ratio) (FQ)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),

    -- OPEX TEMPORAL FEATURES
    ('feat_opex_fq', 'opex_fq', 'Efficiency Ratios', 'calc_opex_temporal_features', 'Total Operating Expenses (FQ)',
     ARRAY ['Total Operating Expenses (FQ)'], 'Total Operating Expenses (FQ)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_opex_ltm', 'opex_ltm', 'Efficiency Ratios', 'calc_opex_temporal_features', 'Total Operating Expenses (LTM)',
     ARRAY ['Total Operating Expenses (LTM)'], 'Total Operating Expenses (LTM)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_opex_fy', 'opex_fy', 'Efficiency Ratios', 'calc_opex_temporal_features', 'Total Operating Expenses (FY)',
     ARRAY ['Total Operating Expenses (FY)'], 'Total Operating Expenses (FY)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_opex_qoq_growth', 'opex_qoq_growth', 'Efficiency Ratios', 'calc_opex_temporal_features',
     'Operating expense QoQ growth', ARRAY ['Total Operating Expenses (FQ)', 'Total Operating Expenses (-1FQFQ)'],
     'Total Operating Expenses (FQ)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_opex_yoy_growth', 'opex_yoy_growth', 'Efficiency Ratios', 'calc_opex_temporal_features',
     'Operating expense YoY growth', ARRAY ['Total Operating Expenses (FY)', 'Total Operating Expenses (-1FY)'],
     'Total Operating Expenses (FY)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_opex_vs_revenue_trend', 'opex_vs_revenue_trend', 'Efficiency Ratios', 'calc_opex_temporal_features',
     'Change in opex-to-revenue ratio (FY vs -1FY)',
     ARRAY ['Total Operating Expenses (FY)', 'Total Revenues (FY)', 'Total Operating Expenses (-1FY)', 'Total Revenues (-1FY)'],
     'Total Operating Expenses (FY)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_sga_qoq_growth', 'sga_qoq_growth', 'Efficiency Ratios', 'calc_opex_temporal_features',
     'SG&A expense QoQ growth',
     ARRAY ['Selling General & Admin Expenses/Total (FQ)', 'Selling General & Admin Expenses/Total (-1FY)'],
     'Selling General & Admin Expenses/Total (FQ)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_sga_yoy_growth', 'sga_yoy_growth', 'Efficiency Ratios', 'calc_opex_temporal_features',
     'SG&A expense YoY growth',
     ARRAY ['Selling General & Admin Expenses/Total (FY)', 'Selling General & Admin Expenses/Total (-1FY)'],
     'Selling General & Admin Expenses/Total (FY)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_operating_leverage_score', 'operating_leverage_score', 'Efficiency Ratios', 'calc_opex_temporal_features',
     'Operating leverage score (revenue growth - opex growth)',
     ARRAY ['Total Revenues (FY)', 'Total Revenues (-1FY)', 'Total Operating Expenses (FY)', 'Total Operating Expenses (-1FY)'],
     'Total Revenues (FY)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),

    -- ASSET SALE FEATURES
    ('feat_gain_loss_on_sale_of_assets_ltm', 'asset_sale_gain_loss_ltm', 'Accounting Quality',
     'calc_asset_sale_features', 'Gains/Losses on asset sales (LTM)', ARRAY ['Gain (Loss) On Sale Of Assets (LTM)'],
     'Gain (Loss) On Sale Of Assets (LTM)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_asset_sale_frequency', 'asset_sale_frequency', 'Accounting Quality', 'calc_asset_sale_features',
     'Frequency of asset sales (last 10 periods)',
     ARRAY ['Gain (Loss) On Sale Of Assets (FQ)', 'Gain (Loss) On Sale Of Assets (-1FQFQ)', 'Gain (Loss) On Sale Of Assets (-2FQFQ)', 'Gain (Loss) On Sale Of Assets (-3FQFQ)', 'Gain (Loss) On Sale Of Assets (-4FQFQ)', 'Gain (Loss) On Sale Of Assets (FY)', 'Gain (Loss) On Sale Of Assets (-1FY)', 'Gain (Loss) On Sale Of Assets (-2FY)', 'Gain (Loss) On Sale Of Assets (-3FY)', 'Gain (Loss) On Sale Of Assets (-4FY)'],
     'Gain (Loss) On Sale Of Assets (FQ)', 'score', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_asset_sale_trend', 'asset_sale_trend', 'Accounting Quality', 'calc_asset_sale_features',
     'Asset sale trend vs prior average',
     ARRAY ['Gain (Loss) On Sale Of Assets (FQ)', 'Gain (Loss) On Sale Of Assets (-1FQFQ)', 'Gain (Loss) On Sale Of Assets (-2FQFQ)', 'Gain (Loss) On Sale Of Assets (-3FQFQ)', 'Gain (Loss) On Sale Of Assets (-4FQFQ)'],
     'Gain (Loss) On Sale Of Assets (FQ)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),

    -- SHARE DILUTION TRACKING
    ('feat_shrs_out_1fy', 'shrs_out_1fy', 'Quality & Risk', 'calc_share_dilution_tracking',
     'Shares outstanding (1 year ago)', ARRAY ['Shrs Out (-1FY)'], 'Shrs Out (-1FY)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_shares_yoy_change_pct', 'shares_yoy_change_pct', 'Quality & Risk', 'calc_share_dilution_tracking',
     'YoY percentage change in shares outstanding', ARRAY ['Shrs Out', 'Shrs Out (-1FY)'], 'Shrs Out', 'growth',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_buyback_flag', 'net_buyback_flag', 'Quality & Risk', 'calc_share_dilution_tracking',
     'Net buyback activity flag', ARRAY ['Shrs Out', 'Shrs Out (-1FY)'], 'Shrs Out', 'flag', 'INTEGER',
     CURRENT_TIMESTAMP),

    -- FORWARD CONSENSUS FEATURES
    ('feat_pe_ntm', 'pe_ntm', 'Analyst Sentiment', 'calc_forward_consensus_features', 'Forward P/E (NTM)',
     ARRAY ['P/E (NTM)'], 'P/E (NTM)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pe_est_fy1', 'pe_est_fy1', 'Analyst Sentiment', 'calc_forward_consensus_features', 'Estimated P/E (FY1)',
     ARRAY ['P/E (EST FY1)'], 'P/E (EST FY1)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pe_forward_discount_consensus', 'pe_forward_discount', 'Analyst Sentiment',
     'calc_forward_consensus_features',
     'Forward P/E discount (NTM vs LTM)', ARRAY ['P/E (NTM)', 'P/E (LTM)'], 'P/E (NTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_eps_gaap_vs_norm_ntm', 'eps_gaap_vs_norm_ntm', 'Analyst Sentiment', 'calc_forward_consensus_features',
     'GAAP vs Normalized EPS spread (NTM)', ARRAY ['EPS GAAP - Est Avg (NTM)', 'EPS Norm - Est Avg (NTM)'],
     'EPS GAAP - Est Avg (NTM)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_gaap_vs_norm_fy1e', 'eps_gaap_vs_norm_fy1e', 'Analyst Sentiment', 'calc_forward_consensus_features',
     'GAAP vs Normalized EPS spread (FY1E)', ARRAY ['EPS GAAP - Est Avg (FY1E)', 'EPS Norm - Est Avg (FY1E)'],
     'EPS GAAP - Est Avg (FY1E)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_forward_adjustment_trend', 'forward_adjustment_trend', 'Analyst Sentiment',
     'calc_forward_consensus_features',
     'Trend in forward EPS adjustments',
     ARRAY ['EPS GAAP - Est Avg (FY1E)', 'EPS Norm - Est Avg (FY1E)', 'EPS/Adj. (LTM)', 'Net EPS - Basic (LTM)'],
     'EPS GAAP - Est Avg (FY1E)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_est_ntm', 'ebitda_est_ntm', 'Analyst Sentiment', 'calc_forward_consensus_features',
     'Estimated EBITDA (NTM)', ARRAY ['EBITDA - Est Avg (NTM)'], 'EBITDA - Est Avg (NTM)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_ebitda_est_fy1e', 'ebitda_est_fy1e', 'Analyst Sentiment', 'calc_forward_consensus_features',
     'Estimated EBITDA (FY1E)', ARRAY ['EBITDA - Est Avg (FY1E)'], 'EBITDA - Est Avg (FY1E)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_ev_ebitda_est_fy1', 'ev_ebitda_est_fy1', 'Analyst Sentiment', 'calc_forward_consensus_features',
     'Estimated EV/EBITDA (FY1)', ARRAY ['EV/EBITDA (EST FY1)'], 'EV/EBITDA (EST FY1)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_ebitda_forward_growth', 'ebitda_forward_growth', 'Analyst Sentiment', 'calc_forward_consensus_features',
     'Forward EBITDA growth (FY1E vs LTM)', ARRAY ['EBITDA - Est Avg (FY1E)', 'EBITDA (LTM)'],
     'EBITDA - Est Avg (FY1E)',
     'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_earnings_revision_divergence', 'earnings_revision_divergence', 'Analyst Sentiment',
     'calc_forward_consensus_features', 'Divergence between GAAP and non-GAAP revisions (3M vs 1M)',
     ARRAY ['EPS Est Avg Rev % (FY1E - 3M)', 'EPS GAAP Est Avg Rev % (FY1E - 3M)', 'EPS Est Avg Rev % (FY1E - 1M)', 'EPS GAAP Est Avg Rev % (FY1E - 1M)'],
     'EPS Est Avg Rev % (FY1E - 3M)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_forward_pe_vs_sector_proxy', 'forward_pe_vs_sector_proxy', 'Analyst Sentiment',
     'calc_forward_consensus_features', 'Forward P/E vs 3Y average proxy', ARRAY ['P/E (NTM)', 'P/E (3YAVGLTM)'],
     'P/E (NTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),

    -- PRICE TARGET ACHIEVEMENT FEATURES
    ('feat_pt_achievement_1y', 'pt_achievement_1y', 'Analyst Sentiment', 'calc_price_target_achievement_features',
     'Price target achievement ratio (1Y)', ARRAY ['Price Target (1Y Ago)', 'Last Price'], 'Price Target (1Y Ago)',
     'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_accuracy_1y', 'pt_accuracy_1y', 'Analyst Sentiment', 'calc_price_target_achievement_features',
     'Price target accuracy (1Y)', ARRAY ['Last Price', 'Price Target (1Y Ago)'], 'Price Target (1Y Ago)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_optimism_bias', 'pt_optimism_bias', 'Analyst Sentiment', 'calc_price_target_achievement_features',
     'Analyst optimism bias (1Y)', ARRAY ['Price Target (1Y Ago)', 'Last Price'], 'Price Target (1Y Ago)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_range_hit_rate', 'pt_range_hit_rate', 'Analyst Sentiment', 'calc_price_target_achievement_features',
     'Price target range hit rate (1Y)',
     ARRAY ['Last Price', 'Price Target - Low (1Y Ago)', 'Price Target - High (1Y Ago)'], 'Last Price', 'flag',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_median_vs_mean_spread', 'pt_median_vs_mean_spread', 'Analyst Sentiment',
     'calc_price_target_achievement_features', 'Price target median vs mean spread',
     ARRAY ['Price Target', 'Price Target - Median'], 'Price Target - Median', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_high_low_convergence_1y', 'pt_high_low_convergence_1y', 'Analyst Sentiment',
     'calc_price_target_achievement_features', 'Price target high-low convergence trend (1Y)',
     ARRAY ['Price Target - High', 'Price Target - Low', 'Price Target - Median', 'Price Target - High (1Y Ago)', 'Price Target - Low (1Y Ago)', 'Price Target - Median (1Y Ago)'],
     'Price Target - Median', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_analyst_count_stability', 'analyst_count_stability', 'Analyst Sentiment',
     'calc_price_target_achievement_features', 'Analyst coverage count stability (1Y)',
     ARRAY ['Price Target - #', 'Price Target - # (1Y Ago)', 'Price Target - # (6M Ago)', 'Price Target - # (3M Ago)'],
     'Price Target - #', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_p_e_vs_3y_avg', 'p_e_vs_3y_avg', 'Valuation Timeseries', 'calc_valuation_timeseries_features',
     'P/E vs 3-year average', ARRAY ['P/E (LTM)', 'P/E (3YAVGLTM)'], 'P/E (LTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),

    -- DIVIDEND HISTORY FEATURES
    ('feat_div_yield_2fy', 'div_yield_2fyind', 'Dividend Reliability', 'calc_dividend_history_features',
     'Dividend yield (2 years ago)', ARRAY ['Div Yield (-2FYInd)'], 'Div Yield (-2FYInd)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_div_yield_3fy', 'div_yield_3fyind', 'Dividend Reliability', 'calc_dividend_history_features',
     'Dividend yield (3 years ago)', ARRAY ['Div Yield (-3FYInd)'], 'Div Yield (-3FYInd)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_div_yield_4fy', 'div_yield_4fyind', 'Dividend Reliability', 'calc_dividend_history_features',
     'Dividend yield (4 years ago)', ARRAY ['Div Yield (-4FYInd)'], 'Div Yield (-4FYInd)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_div_yield_5fy', 'div_yield_5fyind', 'Dividend Reliability', 'calc_dividend_history_features',
     'Dividend yield (5 years ago)', ARRAY ['Div Yield (-5FYInd)'], 'Div Yield (-5FYInd)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_div_yield_trend_3y', 'div_yield_5y_trend', 'Dividend Reliability', 'calc_dividend_history_features',
     '3-year trend in dividend yield', ARRAY ['Div Yield (Ind)', 'Div Yield (-3FYInd)'], 'Div Yield (Ind)', 'growth',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_div_yield_volatility', 'div_yield_stability', 'Dividend Reliability', 'calc_dividend_history_features',
     'Dividend yield volatility (5Y range)',
     ARRAY ['Div Yield (Ind)', 'Div Yield (-1FYInd)', 'Div Yield (-2FYInd)', 'Div Yield (-3FYInd)', 'Div Yield (-4FYInd)'],
     'Div Yield (Ind)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_div_yield_declining_flag', 'div_yield_declining_flag', 'Dividend Reliability',
     'calc_dividend_history_features', 'Declining dividend yield flag (3 consecutive years)',
     ARRAY ['Div Yield (Ind)', 'Div Yield (-1FYInd)', 'Div Yield (-2FYInd)', 'Div Yield (-3FYInd)'], 'Div Yield (Ind)',
     'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_div_yield_mean_5y', 'div_yield_mean_5y', 'Dividend Reliability', 'calc_dividend_history_features',
     '5-year mean dividend yield',
     ARRAY ['Div Yield (Ind)', 'Div Yield (-1FYInd)', 'Div Yield (-2FYInd)', 'Div Yield (-3FYInd)', 'Div Yield (-4FYInd)'],
     'Div Yield (Ind)', 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_div_yield_vs_5y_mean', 'div_yield_vs_5y_mean', 'Dividend Reliability', 'calc_dividend_history_features',
     'Current dividend yield vs 5-year mean ratio',
     ARRAY ['Div Yield (Ind)', 'Div Yield (-1FYInd)', 'Div Yield (-2FYInd)', 'Div Yield (-3FYInd)', 'Div Yield (-4FYInd)'],
     'Div Yield (Ind)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),

    -- SIZE & LIQUIDITY FEATURES
    ('feat_market_cap_dynamic', 'market_cap', 'Technical Analysis', 'calc_size_liquidity_features',
     'Market capitalization (dynamic)', ARRAY ['Market Cap'], 'Market Cap', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_market_cap_country_r', 'market_cap_country_r', 'Technical Analysis', 'calc_size_liquidity_features',
     'Market cap country rank proxy', ARRAY ['Market Cap (Country R)'], 'Market Cap (Country R)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_log_market_cap', 'log_market_cap', 'Technical Analysis', 'calc_size_liquidity_features',
     'Log of market capitalization', ARRAY ['Market Cap'], 'Market Cap', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_volume_shrs', 'volume_shrs', 'Technical Analysis', 'calc_size_liquidity_features',
     'Trading volume (shares)',
     ARRAY ['Volume (Shrs)'], 'Volume (Shrs)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_relative_volume_dynamic', 'relative_volume', 'Technical Analysis', 'calc_size_liquidity_features',
     'Relative trading volume', ARRAY ['Rel. Volume'], 'Rel. Volume', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_shares_outstanding', 'shares_outstanding', 'Technical Analysis', 'calc_size_liquidity_features',
     'Shares outstanding', ARRAY ['Shrs Out'], 'Shrs Out', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_daily_turnover_ratio', 'daily_turnover_ratio', 'Technical Analysis', 'calc_size_liquidity_features',
     'Daily share turnover ratio', ARRAY ['Volume (Shrs)', 'Shrs Out'], 'Volume (Shrs)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_size_class', 'size_class', 'Technical Analysis', 'calc_size_liquidity_features',
     'Company size classification', ARRAY ['Size Class'], 'Size Class', 'direct', 'TEXT', CURRENT_TIMESTAMP),
    ('feat_style_class', 'style_class', 'Technical Analysis', 'calc_size_liquidity_features',
     'Investment style classification', ARRAY ['Style Class'], 'Style Class', 'direct', 'TEXT', CURRENT_TIMESTAMP),
    ('feat_liquidity_score', 'liquidity_score', 'Technical Analysis', 'calc_size_liquidity_features',
     'Composite liquidity score', ARRAY ['Volume (Shrs)', 'Rel. Volume', 'Market Cap'], 'Volume (Shrs)', 'score',
     'NUMERIC', CURRENT_TIMESTAMP),

    -- FCF ESTIMATE FEATURES (v2)
    ('feat_fcf_est_avg_fy1e', 'fcf_est_avg_fy1e', 'Cash Flow', 'calc_fcf_estimate_features',
     'Consensus FCF estimate FY+1 (v2)', ARRAY ['FCF - Est Avg (FY1E)'], 'FCF - Est Avg (FY1E)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_fcf_est_avg_fy2e', 'fcf_est_avg_fy2e', 'Cash Flow', 'calc_fcf_estimate_features',
     'Consensus FCF estimate FY+2 (v2)', ARRAY ['FCF - Est Avg (FY2E)'], 'FCF - Est Avg (FY2E)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_fcf_est_avg_fy3e', 'fcf_est_avg_fy3e', 'Cash Flow', 'calc_fcf_estimate_features',
     'Consensus FCF estimate FY+3 (v2)', ARRAY ['FCF - Est Avg (FY3E)'], 'FCF - Est Avg (FY3E)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_fcf_est_avg_fy4e', 'fcf_est_avg_fy4e', 'Cash Flow', 'calc_fcf_estimate_features',
     'Consensus FCF estimate FY+4 (v2)', ARRAY ['FCF - Est Avg (FY4E)'], 'FCF - Est Avg (FY4E)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_fcf_est_avg_fy5e', 'fcf_est_avg_fy5e', 'Cash Flow', 'calc_fcf_estimate_features',
     'Consensus FCF estimate FY+5 (v2)', ARRAY ['FCF - Est Avg (FY5E)'], 'FCF - Est Avg (FY5E)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_fcf_est_cagr_5y_implied', 'fcf_est_cagr_5y', 'Cash Flow', 'calc_fcf_estimate_features',
     'Implied 5-year FCF CAGR (FY5E/FY1E)', ARRAY ['FCF - Est Avg (FY5E)', 'FCF - Est Avg (FY1E)'],
     'FCF - Est Avg (FY5E)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_trend_linear', 'fcf_est_trend', 'Cash Flow', 'calc_fcf_estimate_features',
     'Linear FCF estimate trend (FY5E/FY1E ratio)', ARRAY ['FCF - Est Avg (FY5E)', 'FCF - Est Avg (FY1E)'],
     'FCF - Est Avg (FY5E)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ev_sales_forward_discount', 'ev_sales_forward_discount', 'Valuation Timeseries',
     'calc_valuation_timeseries_features', 'Forward EV/Sales discount', ARRAY ['EV/Sales (NTM)', 'EV/Sales (LTM)'],
     'EV/Sales (NTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ev_ebitda_forward_discount', 'ev_ebitda_forward_discount', 'Valuation Timeseries',
     'calc_valuation_timeseries_features', 'Forward EV/EBITDA discount', ARRAY ['EV/EBITDA (NTM)', 'EV/EBITDA (LTM)'],
     'EV/EBITDA (NTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_p_e_forward_discount', 'p_e_forward_discount', 'Valuation Timeseries', 'calc_valuation_timeseries_features',
     'Forward P/E discount', ARRAY ['P/E (EST FY1)', 'P/E (LTM)'], 'P/E (EST FY1)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_p_b_vs_5y_avg', 'p_b_vs_5y_avg', 'Valuation Timeseries', 'calc_valuation_timeseries_features',
     'P/B vs 5-year average', ARRAY ['P/B (LTM)', 'P/B (5YAVG)'], 'P/B (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),

    -- EXTENDED VALUATION TIMESERIES
    ('feat_ev_sales_qoq_1q', 'ev_sales_qoq_1q', 'Valuation Timeseries', 'calc_extended_valuation_timeseries',
     'EV/Sales QoQ change (1Q)', ARRAY ['EV/Sales (LTM)', 'EV/Sales (-1FQLTM)'], 'EV/Sales (LTM)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_p_e_vs_5y_avg_ext', 'p_e_vs_5y_avg_ext', 'Valuation Timeseries', 'calc_extended_valuation_timeseries',
     'P/E vs 5-year average extended', ARRAY ['P/E (LTM)', 'P/E (5YAVGLTM)'], 'P/E (LTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_p_b_momentum_yoy', 'p_b_momentum_yoy', 'Valuation Timeseries', 'calc_extended_valuation_timeseries',
     'P/B year-over-year momentum', ARRAY ['P/B (LTM)', 'P/B (-1FY)'], 'P/B (LTM)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_forward_pe_premium', 'forward_pe_premium', 'Valuation Timeseries', 'calc_extended_valuation_timeseries',
     'Forward P/E premium percentage', ARRAY ['P/E (EST FY1)', 'P/E (LTM)'], 'P/E (EST FY1)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),

    -- MOMENTUM FEATURES
    ('feat_price_momentum_1m', 'price_momentum_1m', 'Technical Analysis', 'calc_momentum_features',
     '1-month price momentum', ARRAY ['Last Price', 'Price (1M Ago)'], 'Last Price', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_price_momentum_3m', 'price_momentum_3m', 'Technical Analysis', 'calc_momentum_features',
     '3-month price momentum', ARRAY ['Last Price', 'Price (3M Ago)'], 'Last Price', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_price_momentum_6m', 'price_momentum_6m', 'Technical Analysis', 'calc_momentum_features',
     '6-month price momentum', ARRAY ['Last Price', 'Price (6M Ago)'], 'Last Price', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_price_momentum_1y', 'price_momentum_1y', 'Technical Analysis', 'calc_momentum_features',
     '1-year price momentum', ARRAY ['Last Price', 'Price (1Y Ago)'], 'Last Price', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_price_momentum_5d', 'price_momentum_5d', 'Technical Analysis', 'calc_momentum_features',
     '5-day price momentum', ARRAY ['Last Price', 'Price (5D Ago)'], 'Last Price', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_ema_crossover_20_50', 'ema_crossover_20_50', 'Technical Analysis', 'calc_momentum_features',
     'EMA 20/50 crossover signal', ARRAY ['EMA (20D)', 'EMA (50D)'], 'EMA (20D)', 'flag', 'SMALLINT',
     CURRENT_TIMESTAMP),
    ('feat_ema_crossover_50_250', 'ema_crossover_50_250', 'Technical Analysis', 'calc_momentum_features',
     'EMA 50/250 crossover signal', ARRAY ['EMA (50D)', 'EMA (250D)'], 'EMA (50D)', 'flag', 'SMALLINT',
     CURRENT_TIMESTAMP),
    ('feat_price_vs_ema_20d', 'price_vs_ema_20d', 'Technical Analysis', 'calc_momentum_features',
     'Price vs 20-day EMA', ARRAY ['Last Price', 'EMA (20D)'], 'Last Price', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_price_vs_ema_250d', 'price_vs_ema_250d', 'Technical Analysis', 'calc_momentum_features',
     'Price vs 250-day EMA', ARRAY ['Last Price', 'EMA (250D)'], 'Last Price', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pct_off_52w_high', 'pct_off_52w_high', 'Technical Analysis', 'calc_momentum_features',
     'Percentage off 52-week high', ARRAY ['52W High/Adj', 'Last Price'], '52W High/Adj', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_pct_above_52w_low', 'pct_above_52w_low', 'Technical Analysis', 'calc_momentum_features',
     'Percentage above 52-week low', ARRAY ['Last Price', '52W Low/Adj'], '52W Low/Adj', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_range_52w_position', 'range_52w_position', 'Technical Analysis', 'calc_momentum_features',
     'Position within 52-week range (0-1)', ARRAY ['Last Price', '52W Low/Adj', '52W High/Adj'], 'Last Price', 'score',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_beta_momentum', 'beta_momentum', 'Technical Analysis', 'calc_momentum_features',
     'Beta momentum (1Y vs 5Y)', ARRAY ['Beta (1Y)', 'Beta (5Y)'], 'Beta (1Y)', 'difference', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_volatility_regime', 'volatility_regime', 'Technical Analysis', 'calc_momentum_features',
     'Volatility regime indicator', ARRAY ['Volatility (1M)', 'Volatility (1Y)'], 'Volatility (1M)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),

    -- TECHNICAL ANALYSIS FEATURES
    ('feat_ema_slope_20d', 'ema_slope_20d', 'Technical Analysis', 'calc_technical_analysis_features',
     'EMA 20-day slope', ARRAY ['EMA (20D)', 'EMA (50D)'], 'EMA (20D)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ema_trend_consistency', 'ema_trend_consistency', 'Technical Analysis', 'calc_technical_analysis_features',
     'EMA trend consistency (-1, 0, 1)', ARRAY ['EMA (20D)', 'EMA (50D)', 'EMA (100D)', 'EMA (250D)'], 'EMA (20D)',
     'flag', 'SMALLINT', CURRENT_TIMESTAMP),
    ('feat_price_vs_ema_100d', 'price_vs_ema_100d', 'Technical Analysis', 'calc_technical_analysis_features',
     'Price vs 100-day EMA percentage', ARRAY ['Last Price', 'EMA (100D)'], 'Last Price', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_near_52w_high_flag', 'near_52w_high_flag', 'Technical Analysis', 'calc_technical_analysis_features',
     'Near 52-week high flag (within 5%)', ARRAY ['52W High/Adj', 'Last Price'], 'Last Price', 'flag', 'BOOLEAN',
     CURRENT_TIMESTAMP),
    ('feat_near_52w_low_flag', 'near_52w_low_flag', 'Technical Analysis', 'calc_technical_analysis_features',
     'Near 52-week low flag (within 5%)', ARRAY ['Last Price', '52W Low/Adj'], 'Last Price', 'flag', 'BOOLEAN',
     CURRENT_TIMESTAMP),
    ('feat_volume_momentum_score', 'volume_momentum_score', 'Technical Analysis', 'calc_technical_analysis_features',
     'Volume-weighted momentum score', ARRAY ['Rel. Volume', 'Price Chg. % (1M)'], 'Rel. Volume', 'score', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_breakout_signal', 'breakout_signal', 'Technical Analysis', 'calc_technical_analysis_features',
     'Breakout signal flag', ARRAY ['EMA (20D)', 'EMA (50D)', '52W High/Adj', 'Last Price'], 'Last Price', 'flag',
     'INTEGER', CURRENT_TIMESTAMP),
    ('feat_high_volume_flag', 'high_volume_flag', 'Technical Analysis', 'calc_technical_analysis_features',
     'High relative volume flag (>1.5x)', ARRAY ['Rel. Volume'], 'Rel. Volume', 'flag', 'BOOLEAN', CURRENT_TIMESTAMP),
    ('feat_low_volume_flag', 'low_volume_flag', 'Technical Analysis', 'calc_technical_analysis_features',
     'Low relative volume flag (<0.5x)', ARRAY ['Rel. Volume'], 'Rel. Volume', 'flag', 'BOOLEAN', CURRENT_TIMESTAMP),
    ('feat_volatility_compression', 'volatility_compression', 'Technical Analysis', 'calc_technical_analysis_features',
     'Volatility compression (1Y - 1M)', ARRAY ['Volatility (1Y)', 'Volatility (1M)'], 'Volatility (1Y)', 'difference',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_volatility_term_structure', 'volatility_term_structure', 'Technical Analysis',
     'calc_technical_analysis_features', 'Volatility term structure (3M - 6M)',
     ARRAY ['Volatility (3M)', 'Volatility (6M)'], 'Volatility (3M)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),

    -- VOLATILITY SURFACE FEATURES
    ('feat_vol_1m', 'volatility_1m', 'Technical Analysis', 'calc_volatility_surface_features', '1-month volatility',
     ARRAY ['Volatility (1M)'], 'Volatility (1M)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_vol_3m', 'volatility_3m', 'Technical Analysis', 'calc_volatility_surface_features', '3-month volatility',
     ARRAY ['Volatility (3M)'], 'Volatility (3M)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_vol_6m', 'volatility_6m', 'Technical Analysis', 'calc_volatility_surface_features', '6-month volatility',
     ARRAY ['Volatility (6M)'], 'Volatility (6M)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_vol_1y', 'volatility_1y', 'Technical Analysis', 'calc_volatility_surface_features', '1-year volatility',
     ARRAY ['Volatility (1Y)'], 'Volatility (1Y)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_vol_term_spread_short', 'volatility_trend_short', 'Technical Analysis', 'calc_volatility_surface_features',
     'Short-term volatility term spread (3M - 1M)', ARRAY ['Volatility (3M)', 'Volatility (1M)'], 'Volatility (3M)',
     'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_vol_term_spread_long', 'volatility_trend_long', 'Technical Analysis', 'calc_volatility_surface_features',
     'Long-term volatility term spread (1Y - 6M)', ARRAY ['Volatility (1Y)', 'Volatility (6M)'], 'Volatility (1Y)',
     'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_vol_ratio_3m_1y', 'vol_ratio_3m_1y', 'Technical Analysis', 'calc_volatility_surface_features',
     '3M/1Y volatility ratio', ARRAY ['Volatility (3M)', 'Volatility (1Y)'], 'Volatility (3M)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_vol_hump', 'vol_hump', 'Technical Analysis', 'calc_volatility_surface_features',
     'Volatility hump (6M - average of 3M & 1Y)', ARRAY ['Volatility (6M)', 'Volatility (3M)', 'Volatility (1Y)'],
     'Volatility (6M)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_beta_1y', 'beta_1y', 'Technical Analysis', 'calc_volatility_surface_features', '1-year beta',
     ARRAY ['Beta (1Y)'], 'Beta (1Y)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_beta_2y', 'beta_2y', 'Technical Analysis', 'calc_volatility_surface_features', '2-year beta',
     ARRAY ['Beta (2Y)'], 'Beta (2Y)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_beta_5y', 'beta_5y', 'Technical Analysis', 'calc_volatility_surface_features', '5-year beta',
     ARRAY ['Beta (5Y)'], 'Beta (5Y)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_beta_term_structure', 'beta_term_structure', 'Technical Analysis', 'calc_volatility_surface_features',
     'Beta term structure (1Y vs 5Y ratio)', ARRAY ['Beta (1Y)', 'Beta (5Y)'], 'Beta (1Y)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_beta_convexity', 'beta_convexity', 'Technical Analysis', 'calc_volatility_surface_features',
     'Beta convexity (2Y - average of 1Y & 5Y)', ARRAY ['Beta (2Y)', 'Beta (1Y)', 'Beta (5Y)'], 'Beta (2Y)',
     'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_realized_vs_implied_proxy', 'realized_vs_implied_proxy', 'Technical Analysis',
     'calc_volatility_surface_features', 'Proxy for realized vs implied volatility (1M/1Y ratio)',
     ARRAY ['Volatility (1M)', 'Volatility (1Y)'], 'Volatility (1M)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),

    -- PROFITABILITY FEATURES
    ('feat_roe', 'roe', 'Profitability', 'calc_profitability_features', 'Return on Equity (LTM)',
     ARRAY ['Return On Equity % (LTM)'], 'Return On Equity % (LTM)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_roa', 'roa', 'Profitability', 'calc_profitability_features', 'Return on Assets (LTM)',
     ARRAY ['Return on Assets (ROA) % (LTM)'], 'Return on Assets (ROA) % (LTM)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_gross_margin_pct', 'gross_margin_pct', 'Profitability', 'calc_profitability_features',
     'Gross Profit Margin (LTM)', ARRAY ['Gross Profit Margin % (LTM)'], 'Gross Profit Margin % (LTM)', 'direct',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_operating_margin_pct', 'operating_margin_pct', 'Profitability', 'calc_profitability_features',
     'Operating Margin (calculated)', ARRAY ['Operating Income (LTM)', 'Total Revenues (LTM)'],
     'Operating Income (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_margin_pct', 'net_margin_pct', 'Profitability', 'calc_profitability_features', 'Net Income Margin (LTM)',
     ARRAY ['Net Income Margin % (LTM)'], 'Net Income Margin % (LTM)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_margin_pct', 'ebitda_margin_pct', 'Profitability', 'calc_profitability_features',
     'EBITDA Margin (calculated)', ARRAY ['EBITDA (LTM)', 'Total Revenues (LTM)'], 'EBITDA (LTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_roic', 'roic', 'Profitability', 'calc_profitability_features', 'Return on Invested Capital',
     ARRAY ['Net Income - (IS) (LTM)', 'Total Equity (LTM)', 'Total Debt (LTM)'], 'Net Income - (IS) (LTM)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_intensity', 'rnd_intensity', 'Profitability', 'calc_profitability_features',
     'R&D Intensity (R&D/Revenue)', ARRAY ['R&D Expenses (LTM)', 'Total Revenues (LTM)'], 'R&D Expenses (LTM)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_equity_multiplier', 'equity_multiplier', 'Profitability', 'calc_profitability_features',
     'Equity Multiplier (Assets/Equity)', ARRAY ['Total Assets (LTM)', 'Total Equity (LTM)'], 'Total Assets (LTM)',
     'ratio', 'NUMERIC', CURRENT_TIMESTAMP),

    -- MARGIN TRENDS
    ('feat_gross_margin_trend_yoy', 'gross_margin_trend_yoy', 'Profitability', 'calc_margin_trends',
     'Gross margin YoY trend', ARRAY ['Gross Profit Margin % (LTM)', 'Gross Profit Margin % (FY)'],
     'Gross Profit Margin % (LTM)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_operating_margin_trend', 'operating_margin_trend', 'Profitability', 'calc_margin_trends',
     'Operating margin trend',
     ARRAY ['Operating Income (LTM)', 'Total Revenues (LTM)', 'Operating Income (FY)', 'Total Revenues (FY)'],
     'Operating Income (LTM)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_margin_trend_yoy', 'net_margin_trend_yoy', 'Profitability', 'calc_margin_trends', 'Net margin YoY trend',
     ARRAY ['Net Income Margin % (LTM)', 'Net Income Margin % (FY)'], 'Net Income Margin % (LTM)', 'difference',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_margin_trend', 'ebitda_margin_trend', 'Profitability', 'calc_margin_trends', 'EBITDA margin trend',
     ARRAY ['EBITDA (LTM)', 'Total Revenues (LTM)', 'EBITDA (FY)', 'Total Revenues (FY)'], 'EBITDA (LTM)', 'difference',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_margin_expansion_flag', 'margin_expansion_flag', 'Profitability', 'calc_margin_trends',
     'Margin expansion flag (all margins improving)',
     ARRAY ['Gross Profit Margin % (LTM)', 'Net Income Margin % (LTM)', 'EBITDA (LTM)'], 'Gross Profit Margin % (LTM)',
     'flag', 'BOOLEAN', CURRENT_TIMESTAMP),

    -- QUALITY & RISK FEATURES
    ('feat_has_goodwill_impairment', 'has_goodwill_impairment', 'Quality & Risk', 'calc_quality_features',
     'Goodwill impairment flag', ARRAY ['Impairment of Goodwill (LTM)'], 'Impairment of Goodwill (LTM)', 'flag',
     'INTEGER', CURRENT_TIMESTAMP),
    ('feat_has_asset_writedown', 'has_asset_writedown', 'Quality & Risk', 'calc_quality_features',
     'Asset writedown flag', ARRAY ['Asset Writedown (LTM)'], 'Asset Writedown (LTM)', 'flag', 'BOOLEAN',
     CURRENT_TIMESTAMP),
    ('feat_has_restructuring', 'has_restructuring', 'Quality & Risk', 'calc_quality_features',
     'Restructuring charges flag', ARRAY ['Restructuring Charges (LTM)'], 'Restructuring Charges (LTM)', 'flag',
     'INTEGER', CURRENT_TIMESTAMP),
    ('feat_goodwill_to_assets_pct', 'goodwill_to_assets_pct', 'Quality & Risk', 'calc_quality_features',
     'Goodwill as % of assets', ARRAY ['Goodwill (LTM)', 'Total Assets (LTM)'], 'Goodwill (LTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_intangible_intensity', 'intangible_intensity', 'Quality & Risk', 'calc_quality_features',
     'Intangible assets intensity', ARRAY ['Gross Intangible Assets (LTM)', 'Total Assets (LTM)'],
     'Gross Intangible Assets (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_exceptional_items_to_ebitda', 'exceptional_items_to_ebitda', 'Quality & Risk', 'calc_quality_features',
     'Exceptional items as % of EBITDA',
     ARRAY ['Impairment of Goodwill (LTM)', 'Asset Writedown (LTM)', 'Restructuring Charges (LTM)', 'EBITDA (LTM)'],
     'EBITDA (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_altman_z_score', 'altman_z_score', 'Quality & Risk', 'calc_quality_features', 'Altman Z-Score (LTM)',
     ARRAY ['Altman Z-Score (LTM)'], 'Altman Z-Score (LTM)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_altman_z_trend', 'altman_z_trend', 'Quality & Risk', 'calc_quality_features',
     'Altman Z-Score trend (FY - LTM)', ARRAY ['Altman Z-Score (FY)', 'Altman Z-Score (LTM)'], 'Altman Z-Score (LTM)',
     'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_current_ratio', 'current_ratio', 'Quality & Risk', 'calc_quality_features', 'Current Ratio (LTM)',
     ARRAY ['Current Ratio (LTM)'], 'Current Ratio (LTM)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_quick_ratio', 'quick_ratio', 'Quality & Risk', 'calc_quality_features', 'Quick Ratio (calculated)',
     ARRAY ['Total Current Assets (LTM)', 'Inventory (LTM)', 'Total Current Liabilities (LTM)'],
     'Total Current Assets (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_beta_spread', 'beta_spread', 'Quality & Risk', 'calc_beta_risk_features', 'Beta spread (1Y - 5Y)',
     ARRAY ['Beta (1Y)', 'Beta (5Y)'], 'Beta (1Y)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_beta_trend', 'beta_trend', 'Quality & Risk', 'calc_beta_risk_features', 'Beta trend percentage',
     ARRAY ['Beta (1Y)', 'Beta (5Y)'], 'Beta (1Y)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_high_beta_flag', 'high_beta_flag', 'Quality & Risk', 'calc_beta_risk_features', 'High beta flag (>1.5)',
     ARRAY ['Beta (1Y)'], 'Beta (1Y)', 'flag', 'BOOLEAN', CURRENT_TIMESTAMP),
    ('feat_low_beta_flag', 'low_beta_flag', 'Quality & Risk', 'calc_beta_risk_features', 'Low beta flag (<0.5)',
     ARRAY ['Beta (1Y)'], 'Beta (1Y)', 'flag', 'BOOLEAN', CURRENT_TIMESTAMP),
    ('feat_beta_stability_score', 'beta_stability_score', 'Quality & Risk', 'calc_beta_risk_features',
     'Beta stability score (0-100)', ARRAY ['Beta (1Y)', 'Beta (5Y)'], 'Beta (1Y)', 'score', 'NUMERIC',
     CURRENT_TIMESTAMP),

    -- FINANCIAL DISTRESS FEATURES
    ('feat_distress_risk_score', 'distress_risk_score', 'Financial Distress', 'calc_financial_distress_features',
     'Distress risk score (0-100)', ARRAY ['Altman Z-Score (LTM)'], 'Altman Z-Score (LTM)', 'score', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_liquidity_stress_score', 'liquidity_stress_score', 'Financial Distress', 'calc_financial_distress_features',
     'Liquidity stress score', ARRAY ['Current Ratio (LTM)'], 'Current Ratio (LTM)', 'score', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_working_capital_trend', 'working_capital_trend', 'Financial Distress', 'calc_financial_distress_features',
     'Working capital trend (FQ vs FY)', ARRAY ['Working Capital (FQ)', 'Working Capital (FY)'], 'Working Capital (FQ)',
     'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cash_runway_months', 'cash_runway_months', 'Financial Distress', 'calc_financial_distress_features',
     'Cash runway in months', ARRAY ['Cash And Equivalents (FQ)', 'Total Operating Expenses (LTM)'],
     'Cash And Equivalents (FQ)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_accumulated_deficit_flag', 'accumulated_deficit_flag', 'Financial Distress',
     'calc_financial_distress_features', 'Accumulated deficit flag', ARRAY ['Retained Earnings (FQ)'],
     'Retained Earnings (FQ)', 'flag', 'BOOLEAN', CURRENT_TIMESTAMP),
    ('feat_adequate_cash_buffer', 'adequate_cash_buffer', 'Financial Distress', 'calc_financial_distress_features',
     'Adequate cash buffer flag (>6 months)', ARRAY ['Cash And Equivalents (FQ)', 'Total Operating Expenses (LTM)'],
     'Cash And Equivalents (FQ)', 'flag', 'BOOLEAN', CURRENT_TIMESTAMP),

    -- ACCOUNTING QUALITY FEATURES
    ('feat_goodwill_change_rate', 'goodwill_change_rate', 'Accounting Quality', 'calc_accounting_quality_features',
     'Goodwill YoY change rate', ARRAY ['Goodwill (LTM)', 'Goodwill (-1FY)'], 'Goodwill (LTM)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_restructuring_intensity', 'restructuring_intensity', 'Accounting Quality',
     'calc_accounting_quality_features', 'Restructuring as % of assets',
     ARRAY ['Restructuring Charges (LTM)', 'Total Assets (LTM)'], 'Restructuring Charges (LTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_exceptional_items_frequency', 'exceptional_items_frequency', 'Accounting Quality',
     'calc_accounting_quality_features', 'Count of exceptional items (FQ)',
     ARRAY ['Impairment of Goodwill (FQ)', 'Asset Writedown (FQ)', 'Restructuring Charges (FQ)'],
     'Impairment of Goodwill (FQ)', 'score', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_merger_impact_ratio', 'merger_impact_ratio', 'Accounting Quality', 'calc_accounting_quality_features',
     'M&A charges to market cap', ARRAY ['Merger & Restructuring Charges (LTM)', 'Market Cap'],
     'Merger & Restructuring Charges (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_non_operating_income_share', 'non_operating_income_share', 'Accounting Quality',
     'calc_accounting_quality_features', 'Non-operating income share',
     ARRAY ['Interest And Investment Income (LTM)', 'Net Income - (IS) (LTM)'], 'Interest And Investment Income (LTM)',
     'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_asset_sale_boost', 'asset_sale_boost', 'Accounting Quality', 'calc_accounting_quality_features',
     'Asset sale boost flag', ARRAY ['Gain (Loss) On Sale Of Assets (LTM)'], 'Gain (Loss) On Sale Of Assets (LTM)',
     'flag', 'BOOLEAN', CURRENT_TIMESTAMP),
    ('feat_accounting_quality_score', 'accounting_quality_score', 'Accounting Quality',
     'calc_accounting_quality_features', 'Accounting quality composite score (0-100)',
     ARRAY ['Impairment of Goodwill (LTM)', 'Asset Writedown (LTM)', 'Restructuring Charges (LTM)', 'Goodwill (LTM)', 'Total Assets (LTM)', 'Net Income - (IS) (LTM)'],
     'Net Income - (IS) (LTM)', 'score', 'NUMERIC', CURRENT_TIMESTAMP),

    -- LEVERAGE & LIQUIDITY FEATURES
    ('feat_debt_to_equity', 'debt_to_equity', 'Leverage & Liquidity', 'calc_leverage_features', 'Debt-to-Equity ratio',
     ARRAY ['Total Debt (LTM)', 'Total Equity (LTM)'], 'Total Debt (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_to_assets', 'debt_to_assets', 'Leverage & Liquidity', 'calc_leverage_features', 'Debt-to-Assets ratio',
     ARRAY ['Total Debt (LTM)', 'Total Assets (LTM)'], 'Total Debt (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_equity_ratio', 'equity_ratio', 'Leverage & Liquidity', 'calc_leverage_features', 'Equity ratio',
     ARRAY ['Total Equity (LTM)', 'Total Assets (LTM)'], 'Total Equity (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_interest_coverage', 'interest_coverage', 'Leverage & Liquidity', 'calc_leverage_features',
     'Interest coverage ratio', ARRAY ['EBIT (LTM)', 'Interest Expense/Total (LTM)'], 'EBIT (LTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_cash_ratio', 'cash_ratio', 'Leverage & Liquidity', 'calc_leverage_features', 'Cash ratio',
     ARRAY ['Cash And Equivalents (LTM)', 'Total Current Liabilities (LTM)'], 'Cash And Equivalents (LTM)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_working_capital_ratio', 'working_capital_ratio', 'Leverage & Liquidity', 'calc_leverage_features',
     'Working capital to assets', ARRAY ['Working Capital (LTM)', 'Total Assets (LTM)'], 'Working Capital (LTM)',
     'ratio', 'NUMERIC', CURRENT_TIMESTAMP),

    -- EFFICIENCY RATIOS
    ('feat_asset_turnover', 'asset_turnover', 'Efficiency Ratios', 'calc_efficiency_ratios', 'Asset turnover ratio',
     ARRAY ['Total Revenues (LTM)', 'Total Assets (LTM)'], 'Total Revenues (LTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_inventory_turnover', 'inventory_turnover', 'Efficiency Ratios', 'calc_efficiency_ratios',
     'Inventory turnover ratio', ARRAY ['Cost Of Revenues (LTM)', 'Inventory (LTM)'], 'Cost Of Revenues (LTM)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_receivables_days', 'receivables_days', 'Efficiency Ratios', 'calc_efficiency_ratios',
     'Days receivables outstanding', ARRAY ['Accounts Receivable/Total (FY)', 'Total Revenues (FY)'],
     'Accounts Receivable/Total (FY)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_working_capital_turns', 'working_capital_turns', 'Efficiency Ratios', 'calc_efficiency_ratios',
     'Working capital turnover', ARRAY ['Total Revenues (LTM)', 'Working Capital (LTM)'], 'Total Revenues (LTM)',
     'ratio', 'NUMERIC', CURRENT_TIMESTAMP),

    -- BALANCE SHEET DYNAMICS
    ('feat_cash_to_assets_pct', 'cash_to_assets_pct', 'Balance Sheet', 'calc_balance_sheet_dynamics',
     'Cash as % of assets', ARRAY ['Cash And Equivalents (LTM)', 'Total Assets (LTM)'], 'Cash And Equivalents (LTM)',
     'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cash_change_qoq', 'cash_change_qoq', 'Balance Sheet', 'calc_balance_sheet_dynamics', 'Cash QoQ change',
     ARRAY ['Cash And Equivalents (FQ)', 'Cash And Equivalents (FY)'], 'Cash And Equivalents (FQ)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_cash_vs_5y_avg', 'cash_vs_5y_avg', 'Balance Sheet', 'calc_balance_sheet_dynamics', 'Cash vs 5-year average',
     ARRAY ['Cash And Equivalents (FQ)', 'Cash And Equivalents (5YAVGFQ)'], 'Cash And Equivalents (FQ)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_change_yoy', 'inventory_change_yoy', 'Balance Sheet', 'calc_balance_sheet_dynamics',
     'Inventory YoY change', ARRAY ['Inventory (FY)', 'Inventory (FQ)'], 'Inventory (FY)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_inventory_vs_5y_avg', 'inventory_vs_5y_avg', 'Balance Sheet', 'calc_balance_sheet_dynamics',
     'Inventory vs 5-year average', ARRAY ['Inventory (FQ)', 'Inventory (5YAVGFQ)'], 'Inventory (FQ)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_working_capital_vs_5y_avg', 'working_capital_vs_5y_avg', 'Balance Sheet', 'calc_balance_sheet_dynamics',
     'Working capital vs 5-year average', ARRAY ['Working Capital (FQ)', 'Working Capital (5YAVGFY)'],
     'Working Capital (FQ)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_retained_earnings_vs_5y', 'retained_earnings_vs_5y', 'Balance Sheet', 'calc_balance_sheet_dynamics',
     'Retained earnings vs 5-year average', ARRAY ['Retained Earnings (FQ)', 'Retained Earnings (5YAVGFQ)'],
     'Retained Earnings (FQ)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_intangibles_growth_flag', 'intangibles_growth_flag', 'Balance Sheet', 'calc_balance_sheet_dynamics',
     'Intangibles growth flag (>50%)', ARRAY ['Gross Intangible Assets (FY)', 'Gross Intangible Assets (5YAVGFQ)'],
     'Gross Intangible Assets (FY)', 'flag', 'BOOLEAN', CURRENT_TIMESTAMP),
    ('feat_asset_quality_score', 'asset_quality_score', 'Balance Sheet', 'calc_balance_sheet_dynamics',
     'Asset quality score (0-100)', ARRAY ['Cash And Equivalents (LTM)', 'Total Assets (LTM)', 'Goodwill (LTM)'],
     'Total Assets (LTM)', 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_balance_sheet_strength', 'balance_sheet_strength', 'Balance Sheet', 'calc_balance_sheet_dynamics',
     'Balance sheet strength score (0-100)',
     ARRAY ['Cash And Equivalents (LTM)', 'Total Assets (LTM)', 'Total Equity (LTM)', 'Working Capital (LTM)', 'Current Ratio (LTM)'],
     'Total Assets (LTM)', 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_maturity_risk', 'debt_maturity_risk', 'Balance Sheet', 'calc_balance_sheet_dynamics',
     'Debt maturity risk (Debt/EBITDA)', ARRAY ['Total Debt (LTM)', 'EBITDA (LTM)'], 'Total Debt (LTM)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),

    -- ANALYST SENTIMENT FEATURES
    ('feat_analyst_bullish_pct', 'analyst_bullish_pct', 'Analyst Sentiment', 'calc_sentiment_features',
     'Analyst bullish percentage',
     ARRAY ['# Strong Buys Ratings', '# Buys Ratings', '# Hold Ratings', '# Sell Ratings', '# Strong Sell Ratings'],
     '# Strong Buys Ratings', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_analyst_bearish_pct', 'analyst_bearish_pct', 'Analyst Sentiment', 'calc_sentiment_features',
     'Analyst bearish percentage',
     ARRAY ['# Sell Ratings', '# Strong Sell Ratings', '# Hold Ratings', '# Buys Ratings', '# Strong Buys Ratings'],
     '# Sell Ratings', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_analyst_neutral_pct', 'analyst_neutral_pct', 'Analyst Sentiment', 'calc_sentiment_features',
     'Analyst neutral (hold) percentage',
     ARRAY ['# Hold Ratings', '# Strong Buys Ratings', '# Buys Ratings', '# Sell Ratings', '# Strong Sell Ratings'],
     '# Hold Ratings', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_upside_potential', 'upside_potential', 'Analyst Sentiment', 'calc_sentiment_features',
     'Price target upside potential %', ARRAY ['Price Target - Median', 'Last Price'], 'Price Target - Median', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_price_target_spread_pct', 'price_target_spread_pct', 'Analyst Sentiment', 'calc_sentiment_features',
     'Price target spread %', ARRAY ['Price Target - High', 'Price Target - Low', 'Price Target - Median'],
     'Price Target - Median', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_price_target_revision_1m', 'price_target_revision_1m', 'Analyst Sentiment', 'calc_sentiment_features',
     'Price target 1-month revision', ARRAY ['Price Target', 'Price Target (1M Ago)'], 'Price Target', 'growth',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_price_target_revision_3m', 'price_target_revision_3m', 'Analyst Sentiment', 'calc_sentiment_features',
     'Price target 3-month revision', ARRAY ['Price Target', 'Price Target (3M Ago)'], 'Price Target', 'growth',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_revision_momentum', 'eps_revision_momentum', 'Analyst Sentiment', 'calc_sentiment_features',
     'EPS revision momentum (weighted)',
     ARRAY ['EPS Est Avg Rev % (FY1E - 1W)', 'EPS Est Avg Rev % (FY1E - 1M)', 'EPS Est Avg Rev % (FY1E - 3M)', 'EPS Est Avg Rev % (FY1E - 6M)', 'EPS Est Avg Rev % (FY1E - 1Y)'],
     'EPS Est Avg Rev % (FY1E - 1M)', 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_analyst_rating_normalized', 'analyst_rating_normalized', 'Analyst Sentiment', 'calc_sentiment_features',
     'Normalized analyst rating (0-100)', ARRAY ['Analyst Rating'], 'Analyst Rating', 'score', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_analyst_coverage_quality', 'analyst_coverage_quality', 'Analyst Sentiment', 'calc_sentiment_features',
     'Analyst coverage quality', ARRAY ['Price Target - #', 'Market Cap'], 'Price Target - #', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),

    -- PRICE TARGET DYNAMICS
    ('feat_pt_momentum_1w', 'pt_momentum_1w', 'Price Target Dynamics', 'calc_price_target_dynamics',
     'Price target 1-week momentum', ARRAY ['Price Target', 'Price Target (1W Ago)'], 'Price Target', 'growth',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_momentum_1m', 'pt_momentum_1m', 'Price Target Dynamics', 'calc_price_target_dynamics',
     'Price target 1-month momentum', ARRAY ['Price Target', 'Price Target (1M Ago)'], 'Price Target', 'growth',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_momentum_3m', 'pt_momentum_3m', 'Price Target Dynamics', 'calc_price_target_dynamics',
     'Price target 3-month momentum', ARRAY ['Price Target', 'Price Target (3M Ago)'], 'Price Target', 'growth',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_momentum_6m', 'pt_momentum_6m', 'Price Target Dynamics', 'calc_price_target_dynamics',
     'Price target 6-month momentum', ARRAY ['Price Target', 'Price Target (6M Ago)'], 'Price Target', 'growth',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_momentum_1y', 'pt_momentum_1y', 'Price Target Dynamics', 'calc_price_target_dynamics',
     'Price target 1-year momentum', ARRAY ['Price Target', 'Price Target (1Y Ago)'], 'Price Target', 'growth',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_analyst_coverage_change_1m', 'analyst_coverage_change_1m', 'Price Target Dynamics',
     'calc_price_target_dynamics', 'Analyst coverage change (1M)',
     ARRAY ['Price Target - #', 'Price Target - # (1M Ago)'], 'Price Target - #', 'difference', 'INTEGER',
     CURRENT_TIMESTAMP),
    ('feat_analyst_coverage_change_3m', 'analyst_coverage_change_3m', 'Price Target Dynamics',
     'calc_price_target_dynamics', 'Analyst coverage change (3M)',
     ARRAY ['Price Target - #', 'Price Target - # (3M Ago)'], 'Price Target - #', 'difference', 'INTEGER',
     CURRENT_TIMESTAMP),
    ('feat_analyst_coverage_change_1y', 'analyst_coverage_change_1y', 'Price Target Dynamics',
     'calc_price_target_dynamics', 'Analyst coverage change (1Y)',
     ARRAY ['Price Target - #', 'Price Target - # (1Y Ago)'], 'Price Target - #', 'difference', 'INTEGER',
     CURRENT_TIMESTAMP),

    -- EARNINGS FEATURES
    ('feat_eps_surprise_pct', 'eps_surprise_pct', 'Earnings Quality', 'calc_earnings_features',
     'EPS surprise percentage', ARRAY ['EPS/Adj. (LTM)', 'EPS Norm - Est Avg (FY1E)'], 'EPS/Adj. (LTM)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_surprise_pct', 'revenue_surprise_pct', 'Earnings Quality', 'calc_earnings_features',
     'Revenue surprise percentage', ARRAY ['Total Revenues (LTM)', 'Revenues - Est Avg (FY1E)'], 'Total Revenues (LTM)',
     'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_adjustment_ratio', 'eps_adjustment_ratio', 'Earnings Quality', 'calc_earnings_features',
     'EPS adjustment ratio (Adj/Basic)', ARRAY ['EPS/Adj. (LTM)', 'Net EPS - Basic (LTM)'], 'EPS/Adj. (LTM)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gaap_adj_eps_gap_pct', 'gaap_adj_eps_gap_pct', 'Earnings Quality', 'calc_earnings_features',
     'GAAP vs adjusted EPS gap %', ARRAY ['EPS GAAP - Est Avg (FY1E)', 'EPS Norm - Est Avg (FY1E)'],
     'EPS GAAP - Est Avg (FY1E)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adjustment_ratio', 'ebitda_adjustment_ratio', 'Earnings Quality', 'calc_earnings_features',
     'EBITDA adjustment ratio', ARRAY ['EBITDA/Adj. (LTM)', 'EBITDA (LTM)'], 'EBITDA/Adj. (LTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_eps_quarterly_trend', 'eps_quarterly_trend', 'Earnings Quality', 'calc_earnings_features',
     'EPS quarterly trend', ARRAY ['Net EPS - Basic (FQ)', 'Net EPS - Basic (-4FQFQ)'], 'Net EPS - Basic (FQ)',
     'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_yoy_growth', 'eps_yoy_growth', 'Earnings Quality', 'calc_earnings_features', 'EPS YoY growth',
     ARRAY ['Net EPS - Basic (FY)', 'Net EPS - Basic (-1FY)'], 'Net EPS - Basic (FY)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),

    -- EPS TRAJECTORY FEATURES
    ('feat_eps_qoq_growth', 'eps_qoq_growth', 'EPS Trajectory', 'calc_eps_trajectory_features', 'EPS QoQ growth',
     ARRAY ['Net EPS - Basic (FQ)', 'Net EPS - Basic (-1FQFQ)'], 'Net EPS - Basic (FQ)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_eps_yoy_quarterly', 'eps_yoy_quarterly', 'EPS Trajectory', 'calc_eps_trajectory_features',
     'EPS YoY (quarterly basis)', ARRAY ['Net EPS - Basic (FQ)', 'Net EPS - Basic (-4FQFQ)'], 'Net EPS - Basic (FQ)',
     'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_positive_streak', 'eps_positive_streak', 'EPS Trajectory', 'calc_eps_trajectory_features',
     'Consecutive positive EPS quarters',
     ARRAY ['Net EPS - Basic (FQ)', 'Net EPS - Basic (-1FQFQ)', 'Net EPS - Basic (-2FQFQ)', 'Net EPS - Basic (-3FQFQ)', 'Net EPS - Basic (-4FQFQ)'],
     'Net EPS - Basic (FQ)', 'score', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_eps_cagr_3y', 'eps_cagr_3y', 'EPS Trajectory', 'calc_eps_trajectory_features', 'EPS 3-year CAGR',
     ARRAY ['Net EPS - Basic (FY)', 'Net EPS - Basic (-3FY)'], 'Net EPS - Basic (FY)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_eps_cagr_5y', 'eps_cagr_5y', 'EPS Trajectory', 'calc_eps_trajectory_features', 'EPS 5-year CAGR',
     ARRAY ['Net EPS - Basic (FY)', 'Net EPS - Basic (-5FY)'], 'Net EPS - Basic (FY)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_eps_improvement_count', 'eps_improvement_count', 'EPS Trajectory', 'calc_eps_trajectory_features',
     'Years of EPS improvement (0-5)',
     ARRAY ['Net EPS - Basic (FY)', 'Net EPS - Basic (-1FY)', 'Net EPS - Basic (-2FY)', 'Net EPS - Basic (-3FY)', 'Net EPS - Basic (-4FY)', 'Net EPS - Basic (-5FY)'],
     'Net EPS - Basic (FY)', 'score', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_eps_trajectory_score', 'eps_trajectory_score', 'EPS Trajectory', 'calc_eps_trajectory_features',
     'EPS trajectory score (0-100)',
     ARRAY ['Net EPS - Basic (FY)', 'Net EPS - Basic (-1FY)', 'Net EPS - Basic (-2FY)', 'Net EPS - Basic (-3FY)', 'Net EPS - Basic (-4FY)', 'Net EPS - Basic (-5FY)'],
     'Net EPS - Basic (FY)', 'score', 'NUMERIC', CURRENT_TIMESTAMP),

    -- GROWTH FEATURES
    ('feat_revenue_growth_yoy', 'revenue_growth_yoy', 'Growth Metrics', 'calc_growth_features', 'Revenue YoY growth',
     ARRAY ['Total Revenues (FY)', 'Total Revenues (-1FY)'], 'Total Revenues (FY)', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_ebitda_growth_yoy', 'ebitda_growth_yoy', 'Growth Metrics', 'calc_growth_features', 'EBITDA YoY growth',
     ARRAY ['EBITDA (FY)', 'EBITDA (-1FY)'], 'EBITDA (FY)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_operating_income_growth', 'operating_income_growth', 'Growth Metrics', 'calc_growth_features',
     'Operating income growth', ARRAY ['Operating Income (LTM)', 'Operating Income (FY)'], 'Operating Income (LTM)',
     'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_growth', 'fcf_growth', 'Growth Metrics', 'calc_growth_features', 'Free cash flow growth',
     ARRAY ['FCF (LTM)', 'FCF (FY)'], 'FCF (LTM)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_cagr_5y', 'revenue_cagr_5y', 'Growth Metrics', 'calc_growth_features', 'Revenue 5-year CAGR',
     ARRAY ['Total Revenues/CAGR (5Y FY)'], 'Total Revenues/CAGR (5Y FY)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_forward_revenue_growth', 'forward_revenue_growth', 'Growth Metrics', 'calc_growth_features',
     'Forward revenue growth estimate', ARRAY ['Revenues - Est YoY % (FY1E)'], 'Revenues - Est YoY % (FY1E)', 'direct',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_vs_5y_avg', 'revenue_vs_5y_avg', 'Growth Metrics', 'calc_growth_features',
     'Revenue vs 5-year average', ARRAY ['Total Revenues (LTM)', 'Total Revenues (5YAVGLTM)'], 'Total Revenues (LTM)',
     'ratio', 'NUMERIC', CURRENT_TIMESTAMP),

    -- DIVIDEND FEATURES
    ('feat_dividend_streak', 'dividend_streak', 'Dividend Reliability', 'calc_dividend_features',
     'Consecutive dividend years', ARRAY ['Dividend Streak'], 'Dividend Streak', 'direct', 'INTEGER',
     CURRENT_TIMESTAMP),
    ('feat_dividend_yield_ltm', 'dividend_yield_ltm', 'Dividend Reliability', 'calc_dividend_features',
     'Dividend yield (LTM)', ARRAY ['Div Yield (LTM)'], 'Div Yield (LTM)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_dividend_yield_ntm', 'dividend_yield_ntm', 'Dividend Reliability', 'calc_dividend_features',
     'Dividend yield (NTM)', ARRAY ['Div Yield (NTM)'], 'Div Yield (NTM)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_dividend_payout_ratio', 'dividend_payout_ratio', 'Dividend Reliability', 'calc_dividend_features',
     'Dividend payout ratio', ARRAY ['Common Dividends Paid (LTM)', 'Net Income/Adj. (LTM)'],
     'Common Dividends Paid (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_dividend_coverage', 'fcf_dividend_coverage', 'Dividend Reliability', 'calc_dividend_features',
     'FCF dividend coverage', ARRAY ['FCF (LTM)', 'Common Dividends Paid (LTM)'], 'FCF (LTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_buyback_yield', 'buyback_yield', 'Dividend Reliability', 'calc_dividend_features', 'Buyback yield (LTM)',
     ARRAY ['Buyback Yield (LTM)'], 'Buyback Yield (LTM)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_total_shareholder_yield', 'total_shareholder_yield', 'Dividend Reliability', 'calc_dividend_features',
     'Total shareholder yield', ARRAY ['Buyback Yield (LTM)', 'Div Yield (LTM)'], 'Div Yield (LTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_dividend_growth_expectation', 'dividend_growth_expectation', 'Dividend Reliability',
     'calc_dividend_features', 'Expected dividend growth', ARRAY ['Div Yield (NTM)', 'Div Yield (LTM)'],
     'Div Yield (NTM)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),

    -- EMPLOYMENT FEATURES
    ('feat_revenue_per_employee', 'revenue_per_employee', 'Employee Productivity', 'calc_employment_features',
     'Revenue per employee', ARRAY ['Total Revenues (FY)', 'Full Time Employees (FY)'], 'Total Revenues (FY)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_profit_per_employee', 'profit_per_employee', 'Employee Productivity', 'calc_employment_features',
     'Profit per employee', ARRAY ['Normalized Net Income (FY)', 'Full Time Employees (FY)'],
     'Normalized Net Income (FY)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_per_employee', 'ebitda_per_employee', 'Employee Productivity', 'calc_employment_features',
     'EBITDA per employee', ARRAY ['EBITDA (FY)', 'Full Time Employees (FY)'], 'EBITDA (FY)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_assets_per_employee', 'assets_per_employee', 'Employee Productivity', 'calc_employment_features',
     'Assets per employee', ARRAY ['Total Assets (FY)', 'Full Time Employees (FY)'], 'Total Assets (FY)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fte_growth_1y_pct', 'fte_growth_1y_pct', 'Employee Productivity', 'calc_employment_features',
     'FTE 1-year growth %', ARRAY ['Full Time Employees (FY)', 'Full Time Employees (-1FY)'],
     'Full Time Employees (FY)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fte_growth_3y_pct', 'fte_growth_3y_pct', 'Employee Productivity', 'calc_employment_features',
     'FTE 3-year growth %', ARRAY ['Full Time Employees (FY)', 'Full Time Employees (-3FY)'],
     'Full Time Employees (FY)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_workforce_stability', 'workforce_stability', 'Employee Productivity', 'calc_employment_features',
     'Workforce stability ratio', ARRAY ['Full Time Employees (FY)', 'Avg Employees (5YAVGFY)'],
     'Full Time Employees (FY)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),

    -- CASH FLOW FEATURES
    ('feat_cfo_to_net_income', 'cfo_to_net_income', 'Cash Flow', 'calc_cashflow_features', 'CFO to Net Income ratio',
     ARRAY ['CFO (LTM)', 'Net Income - (IS) (LTM)'], 'CFO (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_to_net_income', 'fcf_to_net_income', 'Cash Flow', 'calc_cashflow_features', 'FCF to Net Income ratio',
     ARRAY ['FCF (LTM)', 'Net Income - (IS) (LTM)'], 'FCF (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_margin', 'fcf_margin', 'Cash Flow', 'calc_cashflow_features', 'FCF margin',
     ARRAY ['FCF (LTM)', 'Total Revenues (LTM)'], 'FCF (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cfo_growth_yoy', 'cfo_growth_yoy', 'Cash Flow', 'calc_cashflow_features', 'CFO YoY growth',
     ARRAY ['CFO (LTM)', 'CFO (-1FY)'], 'CFO (LTM)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_positive_ratio', 'fcf_positive_ratio', 'Cash Flow', 'calc_cashflow_features',
     'FCF positive quarters ratio', ARRAY ['FCF (FQ)', 'FCF (-1FQFQ)', 'FCF (-2FQFQ)', 'FCF (-3FQFQ)', 'FCF (-4FQFQ)'],
     'FCF (FQ)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_acquisition_intensity', 'acquisition_intensity', 'Cash Flow', 'calc_cashflow_features',
     'Cash acquisition intensity',
     ARRAY ['Cash Acquisitions (FQ)', 'Cash Acquisitions (-1FQFQ)', 'Cash Acquisitions (-2FQFQ)', 'Cash Acquisitions (-3FQFQ)'],
     'Cash Acquisitions (FQ)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_self_funding_ratio', 'self_funding_ratio', 'Cash Flow', 'calc_cashflow_features',
     'Self-funding ratio (CFO/CFI)', ARRAY ['CFO (LTM)', 'CFI (LTM)'], 'CFO (LTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),

    -- FCF GROWTH ESTIMATE FEATURES (NEW)
    ('feat_fcf_est_fy1', 'fcf_est_fy1', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Consensus FCF estimate FY+1', ARRAY ['FCF - Est Avg (FY1E)'], 'FCF - Est Avg (FY1E)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_fcf_est_fy2', 'fcf_est_fy2', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Consensus FCF estimate FY+2', ARRAY ['FCF - Est Avg (FY2E)'], 'FCF - Est Avg (FY2E)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_fcf_est_fy3', 'fcf_est_fy3', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Consensus FCF estimate FY+3', ARRAY ['FCF - Est Avg (FY3E)'], 'FCF - Est Avg (FY3E)', 'direct', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_fcf_est_growth_fy1_vs_ltm', 'fcf_est_growth_fy1_vs_ltm', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Estimated FCF growth: FY1E vs LTM', ARRAY ['FCF - Est Avg (FY1E)', 'FCF (LTM)'], 'FCF - Est Avg (FY1E)',
     'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_growth_fy2_vs_fy1', 'fcf_est_growth_fy2_vs_fy1', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Estimated FCF growth: FY2E vs FY1E', ARRAY ['FCF - Est Avg (FY2E)', 'FCF - Est Avg (FY1E)'],
     'FCF - Est Avg (FY2E)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_cagr_3y', 'fcf_est_cagr_3y', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Estimated 3-year FCF CAGR (FY3E/LTM)', ARRAY ['FCF - Est Avg (FY3E)', 'FCF (LTM)'],
     'FCF - Est Avg (FY3E)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_cagr_5y', 'fcf_est_cagr_5y', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Estimated 5-year FCF CAGR (FY5E/LTM)', ARRAY ['FCF - Est Avg (FY5E)', 'FCF (LTM)'],
     'FCF - Est Avg (FY5E)', 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_margin_fy1', 'fcf_est_margin_fy1', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Forward FCF margin (FY1E / Revenue LTM)', ARRAY ['FCF - Est Avg (FY1E)', 'Total Revenues (LTM)'],
     'FCF - Est Avg (FY1E)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_yield_fy1', 'fcf_est_yield_fy1', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Forward FCF yield (FY1E / Market Cap)', ARRAY ['FCF - Est Avg (FY1E)', 'Market Cap'],
     'FCF - Est Avg (FY1E)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_growth_acceleration', 'fcf_est_growth_acceleration', 'Cash Flow', 'calc_fcf_growth_estimates',
     'FCF growth acceleration (FY2-FY1 growth minus FY1-LTM growth)',
     ARRAY ['FCF - Est Avg (FY2E)', 'FCF - Est Avg (FY1E)', 'FCF (LTM)'], 'FCF - Est Avg (FY2E)',
     'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_trajectory_score', 'fcf_est_trajectory_score', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Pct of 5 forward years with positive FCF (0-100)',
     ARRAY ['FCF - Est Avg (FY1E)', 'FCF - Est Avg (FY2E)', 'FCF - Est Avg (FY3E)', 'FCF - Est Avg (FY4E)', 'FCF - Est Avg (FY5E)'],
     'FCF - Est Avg (FY1E)', 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_always_positive', 'fcf_est_always_positive', 'Cash Flow', 'calc_fcf_growth_estimates',
     'All 5 forward FCF estimates positive flag',
     ARRAY ['FCF - Est Avg (FY1E)', 'FCF - Est Avg (FY2E)', 'FCF - Est Avg (FY3E)', 'FCF - Est Avg (FY4E)', 'FCF - Est Avg (FY5E)'],
     'FCF - Est Avg (FY1E)', 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_fcf_est_vs_historical', 'fcf_est_vs_historical', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Forward FCF growth vs last actual YoY growth',
     ARRAY ['FCF - Est Avg (FY1E)', 'FCF (LTM)', 'FCF (FY)', 'FCF (-1FY)'], 'FCF - Est Avg (FY1E)',
     'difference', 'NUMERIC', CURRENT_TIMESTAMP),

    -- COMPOSITE SCORES
    ('feat_piotroski_f_score', 'piotroski_f_score', 'Composite Scores', 'calc_composite_scores',
     'Piotroski F-Score (0-9)',
     ARRAY ['Return on Assets (ROA) % (LTM)', 'CFO (LTM)', 'Total Debt (LTM)', 'Total Equity (LTM)', 'Current Ratio (LTM)', 'Shrs Out', 'Gross Profit Margin % (LTM)', 'Asset Turnover (LTM)'],
     'Return on Assets (ROA) % (LTM)', 'score', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_dilution_score', 'dilution_score', 'Composite Scores', 'calc_composite_scores',
     'Share dilution score (0-100)', ARRAY ['Shrs Out', 'Shrs Out (-1FY)'], 'Shrs Out', 'score', 'NUMERIC',
     CURRENT_TIMESTAMP),

    -- TEMPORAL FEATURES
    ('feat_fiscal_quarter', 'fiscal_quarter', 'Temporal Patterns', 'calc_temporal_features', 'Current fiscal quarter',
     ARRAY ['Fiscal Quarter'], 'Fiscal Quarter', 'direct', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_fiscal_month', 'fiscal_month', 'Temporal Patterns', 'calc_temporal_features', 'Current fiscal month',
     ARRAY ['Fiscal Month'], 'Fiscal Month', 'direct', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_fiscal_year', 'fiscal_year', 'Temporal Patterns', 'calc_temporal_features', 'Current fiscal year',
     ARRAY ['Fiscal Year'], 'Fiscal Year', 'direct', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_days_to_earnings', 'days_to_earnings', 'Temporal Patterns', 'calc_temporal_features',
     'Days until next earnings', ARRAY ['Next Earnings'], 'Next Earnings', 'difference', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_earnings_report_recency', 'earnings_report_recency', 'Temporal Patterns', 'calc_temporal_features',
     'Days since last earnings report', ARRAY ['Income Statement Report Date'], 'Income Statement Report Date',
     'difference', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_reporting_lag', 'reporting_lag', 'Temporal Patterns', 'calc_temporal_features', 'Reporting lag days',
     ARRAY ['Reporting Lag'], 'Reporting Lag', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fiscal_year_progress', 'fiscal_year_progress', 'Temporal Patterns', 'calc_temporal_features',
     'Fiscal year progress (0-1)', ARRAY ['Fiscal Month'], 'Fiscal Month', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),

    -- MARKET DATA
    ('feat_market_cap', 'market_cap', 'Market Data', NULL, 'Market capitalization', ARRAY ['Market Cap'], 'Market Cap',
     'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_enterprise_value', 'enterprise_value', 'Market Data', NULL, 'Enterprise value', ARRAY ['Enterprise Value'],
     'Enterprise Value', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_last_price', 'last_price', 'Market Data', NULL, 'Last traded price', ARRAY ['Last Price'], 'Last Price',
     'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_target_vs_price_pct','target_vs_price_pct','Market Data',Null,'Delta between Stock Price Target and last Price',ARRAY['Target % (Avg)'], 'Target % (Avg)', 'direct', 'NUMERIC', CURRENT_TIMESTAMP),

    -- COST STRUCTURE FEATURES
    ('feat_cogs_to_revenue', 'cogs_to_revenue', 'Efficiency Ratios', 'calc_cost_structure_features',
     'COGS as % of revenue', ARRAY ['Cost Of Revenues (LTM)', 'Total Revenues (LTM)'], 'Cost Of Revenues (LTM)',
     'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_opex_to_revenue', 'opex_to_revenue', 'Efficiency Ratios', 'calc_cost_structure_features',
     'OpEx as % of revenue', ARRAY ['Total Operating Expenses (LTM)', 'Total Revenues (LTM)'],
     'Total Operating Expenses (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_sga_to_revenue', 'sga_to_revenue', 'Efficiency Ratios', 'calc_cost_structure_features',
     'SG&A as % of revenue', ARRAY ['Selling General & Admin Expenses/Total (FY)', 'Total Revenues (FY)'],
     'Selling General & Admin Expenses/Total (FY)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_to_revenue', 'rnd_to_revenue', 'Efficiency Ratios', 'calc_cost_structure_features',
     'R&D as % of revenue', ARRAY ['R&D Expenses (LTM)', 'Total Revenues (LTM)'], 'R&D Expenses (LTM)', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_interest_to_revenue', 'interest_to_revenue', 'Efficiency Ratios', 'calc_cost_structure_features',
     'Interest expense as % of revenue', ARRAY ['Interest Expense/Total (LTM)', 'Total Revenues (LTM)'],
     'Interest Expense/Total (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),

    -- LONG TERM MOMENTUM
    ('feat_price_momentum_3y', 'price_momentum_3y', 'Technical Analysis', 'calc_long_term_momentum_features',
     '3-year price momentum', ARRAY ['Last Price', 'Price (3Y Ago)'], 'Last Price', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_price_momentum_5y', 'price_momentum_5y', 'Technical Analysis', 'calc_long_term_momentum_features',
     '5-year price momentum', ARRAY ['Last Price', 'Price (5Y Ago)'], 'Last Price', 'growth', 'NUMERIC',
     CURRENT_TIMESTAMP),
    ('feat_long_term_trend_score', 'long_term_trend_score', 'Technical Analysis', 'calc_long_term_momentum_features',
     'Long-term trend score (weighted)', ARRAY ['Last Price', 'Price (1Y Ago)', 'Price (3Y Ago)', 'Price (5Y Ago)'],
     'Last Price', 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_multi_year_high_flag', 'multi_year_high_flag', 'Technical Analysis', 'calc_long_term_momentum_features',
     'Near multi-year high flag', ARRAY ['52W High/Adj', 'Last Price', 'Price (3Y Ago)'], 'Last Price', 'flag',
     'INTEGER', CURRENT_TIMESTAMP),
    ('feat_secular_trend_flag', 'secular_trend_flag', 'Technical Analysis', 'calc_long_term_momentum_features',
     'Secular uptrend flag', ARRAY ['Last Price', 'Price (3Y Ago)', 'Price (1Y Ago)', 'EMA (50D)', 'EMA (250D)'],
     'Last Price', 'flag', 'BOOLEAN', CURRENT_TIMESTAMP),

    -- TANGIBLE BOOK FEATURES
    ('feat_tangible_book_value', 'tangible_book_value', 'Valuation Ratios', 'calc_tangible_book_features',
     'Tangible book value', ARRAY ['Total Equity (LTM)', 'Goodwill (LTM)', 'Gross Intangible Assets (LTM)'],
     'Total Equity (LTM)', 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_tangible_book_per_share', 'tangible_book_per_share', 'Valuation Ratios', 'calc_tangible_book_features',
     'Tangible book value per share',
     ARRAY ['Total Equity (LTM)', 'Goodwill (LTM)', 'Gross Intangible Assets (LTM)', 'Shrs Out'], 'Shrs Out', 'ratio',
     'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_price_to_tangible_book', 'price_to_tangible_book', 'Valuation Ratios', 'calc_tangible_book_features',
     'Price to tangible book ratio',
     ARRAY ['Last Price', 'Shrs Out', 'Total Equity (LTM)', 'Goodwill (LTM)', 'Gross Intangible Assets (LTM)'],
     'Last Price', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_tangible_equity_ratio', 'tangible_equity_ratio', 'Valuation Ratios', 'calc_tangible_book_features',
     'Tangible equity ratio',
     ARRAY ['Total Equity (LTM)', 'Goodwill (LTM)', 'Gross Intangible Assets (LTM)', 'Total Assets (LTM)'],
     'Total Equity (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_intangibles_to_equity', 'intangibles_to_equity', 'Valuation Ratios', 'calc_tangible_book_features',
     'Intangibles to equity ratio', ARRAY ['Gross Intangible Assets (LTM)', 'Total Equity (LTM)'],
     'Gross Intangible Assets (LTM)', 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_to_equity', 'goodwill_to_equity', 'Valuation Ratios', 'calc_tangible_book_features',
     'Goodwill to equity ratio', ARRAY ['Goodwill (LTM)', 'Total Equity (LTM)'], 'Goodwill (LTM)', 'ratio', 'NUMERIC',
     CURRENT_TIMESTAMP),

    -- BALANCE SHEET (calc_balance_sheet_dynamics)
    ('feat_receivables_change_yoy', 'receivables_change_yoy', 'Balance Sheet', 'calc_balance_sheet_dynamics',
     'Receivables Change Yoy',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_receivables_vs_5y_avg', 'receivables_vs_5y_avg', 'Balance Sheet', 'calc_balance_sheet_dynamics',
     'Receivables Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- CASH FLOW (calc_cashflow_comprehensive)
    ('feat_cfo_fq', 'cfo_fq', 'Cash Flow', 'calc_cashflow_comprehensive', 'Cfo Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cfo_fy', 'cfo_fy', 'Cash Flow', 'calc_cashflow_comprehensive', 'Cfo Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cfo_ltm', 'cfo_ltm', 'Cash Flow', 'calc_cashflow_comprehensive', 'Cfo Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cfo_positive_years', 'cfo_positive_years', 'Cash Flow', 'calc_cashflow_comprehensive', 'Cfo Positive Years',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_fq', 'fcf_fq', 'Cash Flow', 'calc_cashflow_comprehensive', 'Fcf Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_fy', 'fcf_fy', 'Cash Flow', 'calc_cashflow_comprehensive', 'Fcf Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_growth_yoy', 'fcf_growth_yoy', 'Cash Flow', 'calc_cashflow_comprehensive', 'Fcf Growth Yoy',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_ltm', 'fcf_ltm', 'Cash Flow', 'calc_cashflow_comprehensive', 'Fcf Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_positive_years_comp', 'fcf_positive_years_comp', 'Cash Flow', 'calc_cashflow_comprehensive',
     'Fcf Positive Years Comp',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_yield', 'fcf_yield', 'Cash Flow', 'calc_cashflow_comprehensive', 'Fcf Yield',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    -- COMPOSITE SCORES (calc_composite_scores)
    ('feat_quality_momentum_score', 'quality_momentum_score', 'Composite Scores', 'calc_composite_scores',
     'Quality Momentum Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    -- EFFICIENCY RATIOS (calc_cost_structure_features)
    ('feat_cost_efficiency_score', 'cost_efficiency_score', 'Efficiency Ratios', 'calc_cost_structure_features',
     'Cost Efficiency Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_marketing_to_revenue', 'marketing_to_revenue', 'Efficiency Ratios', 'calc_cost_structure_features',
     'Marketing To Revenue',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_marketing_trend_yoy', 'marketing_trend_yoy', 'Efficiency Ratios', 'calc_cost_structure_features',
     'Marketing Trend Yoy',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_marketing_vs_5y_avg', 'marketing_vs_5y_avg', 'Efficiency Ratios', 'calc_cost_structure_features',
     'Marketing Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_operating_leverage_proxy', 'operating_leverage_proxy', 'Efficiency Ratios', 'calc_cost_structure_features',
     'Operating Leverage Proxy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_sga_efficiency_trend', 'sga_efficiency_trend', 'Efficiency Ratios', 'calc_cost_structure_features',
     'Sga Efficiency Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_sga_trend_yoy', 'sga_trend_yoy', 'Efficiency Ratios', 'calc_cost_structure_features', 'Sga Trend Yoy',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_sga_vs_5y_avg', 'sga_vs_5y_avg', 'Efficiency Ratios', 'calc_cost_structure_features', 'Sga Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- CASH FLOW (calc_cashflow_temporal_features)
    ('feat_cash_burn_rate', 'cash_burn_rate', 'Cash Flow', 'calc_cashflow_temporal_features', 'Cash Burn Rate',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cf_volatility_score', 'cf_volatility_score', 'Cash Flow', 'calc_cashflow_temporal_features',
     'Cf Volatility Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cff_pattern_score', 'cff_pattern_score', 'Cash Flow', 'calc_cashflow_temporal_features', 'Cff Pattern Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cff_quarterly_trend', 'cff_quarterly_trend', 'Cash Flow', 'calc_cashflow_temporal_features',
     'Cff Quarterly Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cfi_negative_quarters', 'cfi_negative_quarters', 'Cash Flow', 'calc_cashflow_temporal_features',
     'Cfi Negative Quarters',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cfi_quarterly_trend', 'cfi_quarterly_trend', 'Cash Flow', 'calc_cashflow_temporal_features',
     'Cfi Quarterly Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cfo_positive_quarters', 'cfo_positive_quarters', 'Cash Flow', 'calc_cashflow_temporal_features',
     'Cfo Positive Quarters',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cfo_quarterly_trend', 'cfo_quarterly_trend', 'Cash Flow', 'calc_cashflow_temporal_features',
     'Cfo Quarterly Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cfo_yoy_quarterly', 'cfo_yoy_quarterly', 'Cash Flow', 'calc_cashflow_temporal_features', 'Cfo Yoy Quarterly',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_quarterly_trend', 'fcf_quarterly_trend', 'Cash Flow', 'calc_cashflow_temporal_features',
     'Fcf Quarterly Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_financing_dependency', 'financing_dependency', 'Cash Flow', 'calc_cashflow_temporal_features',
     'Financing Dependency',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_operating_cf_momentum', 'operating_cf_momentum', 'Cash Flow', 'calc_cashflow_temporal_features',
     'Operating Cf Momentum',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    -- DIVIDEND RELIABILITY (calc_dividend_timing)
    ('feat_days_since_ex_date', 'days_since_ex_date', 'Dividend Reliability', 'calc_dividend_timing',
     'Days Since Ex Date',
     NULL, NULL, 'difference', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_days_to_payment', 'days_to_payment', 'Dividend Reliability', 'calc_dividend_timing', 'Days To Payment',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_dividend_announced_flag', 'dividend_announced_flag', 'Dividend Reliability', 'calc_dividend_timing',
     'Dividend Announced Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_dividend_consistency', 'dividend_consistency', 'Dividend Reliability', 'calc_dividend_timing',
     'Dividend Consistency',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_dividend_frequency_score', 'dividend_frequency_score', 'Dividend Reliability', 'calc_dividend_timing',
     'Dividend Frequency Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_dividend_yield_vs_5y_avg', 'dividend_yield_vs_5y_avg', 'Dividend Reliability', 'calc_dividend_timing',
     'Dividend Yield Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ex_date_approaching_flag', 'ex_date_approaching_flag', 'Dividend Reliability', 'calc_dividend_timing',
     'Ex Date Approaching Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_recent_dividend_change', 'recent_dividend_change', 'Dividend Reliability', 'calc_dividend_timing',
     'Recent Dividend Change',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_div_yield_1fy_ind', 'div_yield_1fy_ind', 'Dividend Reliability', 'calc_dividend_yield_comprehensive',
     'Div Yield 1fy Ind',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_div_yield_5y_avg', 'div_yield_5y_avg', 'Dividend Reliability', 'calc_dividend_yield_comprehensive',
     'Div Yield 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_div_yield_growth_expected', 'div_yield_growth_expected', 'Dividend Reliability',
     'calc_dividend_yield_comprehensive', 'Div Yield Growth Expected',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_div_yield_ind', 'div_yield_ind', 'Dividend Reliability', 'calc_dividend_yield_comprehensive',
     'Div Yield Ind',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_div_yield_ltm', 'div_yield_ltm', 'Dividend Reliability', 'calc_dividend_yield_comprehensive',
     'Div Yield Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_div_yield_ntm', 'div_yield_ntm', 'Dividend Reliability', 'calc_dividend_yield_comprehensive',
     'Div Yield Ntm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_div_yield_vs_5y_avg', 'div_yield_vs_5y_avg', 'Dividend Reliability', 'calc_dividend_yield_comprehensive',
     'Div Yield Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_high_yield_flag', 'high_yield_flag', 'Dividend Reliability', 'calc_dividend_yield_comprehensive',
     'High Yield Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_sustainable_dividend_flag', 'sustainable_dividend_flag', 'Dividend Reliability',
     'calc_dividend_yield_comprehensive', 'Sustainable Dividend Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    -- EARNINGS QUALITY (calc_eps_comprehensive)
    ('feat_eps_adj_ltm', 'eps_adj_ltm', 'Earnings Quality', 'calc_eps_comprehensive', 'Eps Adj Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_basic_fq', 'eps_basic_fq', 'Earnings Quality', 'calc_eps_comprehensive', 'Eps Basic Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_basic_fy', 'eps_basic_fy', 'Earnings Quality', 'calc_eps_comprehensive', 'Eps Basic Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_basic_ltm', 'eps_basic_ltm', 'Earnings Quality', 'calc_eps_comprehensive', 'Eps Basic Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_norm_est_fy1e', 'eps_norm_est_fy1e', 'Earnings Quality', 'calc_eps_comprehensive', 'Eps Norm Est Fy1e',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_positive_years', 'eps_positive_years', 'Earnings Quality', 'calc_eps_comprehensive',
     'Eps Positive Years',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_core_earnings_stability', 'core_earnings_stability', 'Earnings Quality', 'calc_eps_continuing_features',
     'Core Earnings Stability',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_discontinued_ops_impact', 'discontinued_ops_impact', 'Earnings Quality', 'calc_eps_continuing_features',
     'Discontinued Ops Impact',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_1fqfq', 'eps_cont_1fqfq', 'Earnings Quality', 'calc_eps_continuing_features', 'Eps Cont 1fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_1fy', 'eps_cont_1fy', 'Earnings Quality', 'calc_eps_continuing_features', 'Eps Cont 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_2fqfq', 'eps_cont_2fqfq', 'Earnings Quality', 'calc_eps_continuing_features', 'Eps Cont 2fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_2fy', 'eps_cont_2fy', 'Earnings Quality', 'calc_eps_continuing_features', 'Eps Cont 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_3fqfq', 'eps_cont_3fqfq', 'Earnings Quality', 'calc_eps_continuing_features', 'Eps Cont 3fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_3fy', 'eps_cont_3fy', 'Earnings Quality', 'calc_eps_continuing_features', 'Eps Cont 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_4fqfq', 'eps_cont_4fqfq', 'Earnings Quality', 'calc_eps_continuing_features', 'Eps Cont 4fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_4fy', 'eps_cont_4fy', 'Earnings Quality', 'calc_eps_continuing_features', 'Eps Cont 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_cagr_3y', 'eps_cont_cagr_3y', 'Earnings Quality', 'calc_eps_continuing_features',
     'Eps Cont Cagr 3y',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_fq', 'eps_cont_fq', 'Earnings Quality', 'calc_eps_continuing_features', 'Eps Cont Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_fy', 'eps_cont_fy', 'Earnings Quality', 'calc_eps_continuing_features', 'Eps Cont Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_ltm', 'eps_cont_ltm', 'Earnings Quality', 'calc_eps_continuing_features', 'Eps Cont Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_positive_streak', 'eps_cont_positive_streak', 'Earnings Quality', 'calc_eps_continuing_features',
     'Eps Cont Positive Streak',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_qoq_growth', 'eps_cont_qoq_growth', 'Earnings Quality', 'calc_eps_continuing_features',
     'Eps Cont Qoq Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_trajectory_score', 'eps_cont_trajectory_score', 'Earnings Quality', 'calc_eps_continuing_features',
     'Eps Cont Trajectory Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_vs_total_eps', 'eps_cont_vs_total_eps', 'Earnings Quality', 'calc_eps_continuing_features',
     'Eps Cont Vs Total Eps',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_cont_yoy_growth', 'eps_cont_yoy_growth', 'Earnings Quality', 'calc_eps_continuing_features',
     'Eps Cont Yoy Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    -- Cash flow (calc_enhanced_cashflow_features)
    ('feat_acquisition_pause_flag', 'acquisition_pause_flag', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Acquisition Pause Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_acquisition_to_fcf', 'acquisition_to_fcf', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Acquisition To Fcf',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_acquisitions_ltm_total', 'acquisitions_ltm_total', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Acquisitions Ltm Total',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_acquisitions_vs_5y_avg', 'acquisitions_vs_5y_avg', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Acquisitions Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_acquisitions_yoy_growth', 'acquisitions_yoy_growth', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Acquisitions Yoy Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_capex_3y_trend', 'capex_3y_trend', 'Cash flow', 'calc_enhanced_cashflow_features', 'Capex 3y Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_capex_acceleration', 'capex_acceleration', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Capex Acceleration',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_capex_cut_flag', 'capex_cut_flag', 'Cash flow', 'calc_enhanced_cashflow_features', 'Capex Cut Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_capex_qoq_growth', 'capex_qoq_growth', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Capex Qoq Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_capex_volatility', 'capex_volatility', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Capex Volatility',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_capex_vs_5y_avg', 'capex_vs_5y_avg', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Capex Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_capex_yoy_growth', 'capex_yoy_growth', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Capex Yoy Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cash_flow_quality_score', 'cash_flow_quality_score', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Cash Flow Quality Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cff_share_of_cf', 'cff_share_of_cf', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Cff Share Of Cf',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cfi_share_of_cf', 'cfi_share_of_cf', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Cfi Share Of Cf',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_cfo_share_of_cf', 'cfo_share_of_cf', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Cfo Share Of Cf',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_4q_improvement', 'fcf_4q_improvement', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Fcf 4q Improvement',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_always_positive', 'fcf_always_positive', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Fcf Always Positive',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_positive_years', 'fcf_positive_years', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Fcf Positive Years',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_investment_efficiency', 'investment_efficiency', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Investment Efficiency',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ma_intensity_score', 'ma_intensity_score', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Ma Intensity Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_organic_vs_inorganic', 'organic_vs_inorganic', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Organic Vs Inorganic',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_overinvestment_flag', 'overinvestment_flag', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Overinvestment Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_self_funding_flag', 'self_funding_flag', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Self Funding Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_serial_acquirer_flag', 'serial_acquirer_flag', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Serial Acquirer Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_sustainable_ma_flag', 'sustainable_ma_flag', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Sustainable Ma Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_total_investment_to_cfo', 'total_investment_to_cfo', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Total Investment To Cfo',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_underinvestment_flag', 'underinvestment_flag', 'Cash flow', 'calc_enhanced_cashflow_features',
     'Underinvestment Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    -- EMPLOYMENT DYNAMICS (calc_employment_dynamics)
    ('feat_fte_acceleration', 'fte_acceleration', 'Employment Dynamics', 'calc_employment_dynamics', 'Fte Acceleration',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fte_growth_2y_pct', 'fte_growth_2y_pct', 'Employment Dynamics', 'calc_employment_dynamics',
     'Fte Growth 2y Pct',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_headcount_vs_revenue', 'headcount_vs_revenue', 'Employment Dynamics', 'calc_employment_dynamics',
     'Headcount Vs Revenue',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_hiring_intensity', 'hiring_intensity', 'Employment Dynamics', 'calc_employment_dynamics', 'Hiring Intensity',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_layoff_risk_flag', 'layoff_risk_flag', 'Employment Dynamics', 'calc_employment_dynamics', 'Layoff Risk Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_productivity_trend', 'productivity_trend', 'Employment Dynamics', 'calc_employment_dynamics',
     'Productivity Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rapid_hiring_flag', 'rapid_hiring_flag', 'Employment Dynamics', 'calc_employment_dynamics',
     'Rapid Hiring Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_sustainable_growth_flag', 'sustainable_growth_flag', 'Employment Dynamics', 'calc_employment_dynamics',
     'Sustainable Growth Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_workforce_efficiency_gain', 'workforce_efficiency_gain', 'Employment Dynamics', 'calc_employment_dynamics',
     'Workforce Efficiency Gain',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_workforce_volatility', 'workforce_volatility', 'Employment Dynamics', 'calc_employment_dynamics',
     'Workforce Volatility',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- PROFITABILITY (calc_ebit_ebitda_comprehensive)
    ('feat_ebit_1fqfq', 'ebit_1fqfq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit 1fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_1fy', 'ebit_1fy', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_2fqfq', 'ebit_2fqfq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit 2fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_2fy', 'ebit_2fy', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_3fqfq', 'ebit_3fqfq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit 3fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_3fy', 'ebit_3fy', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_4fqfq', 'ebit_4fqfq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit 4fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_4fy', 'ebit_4fy', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_5yavgfq', 'ebit_5yavgfq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit 5yavgfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_5yavgltm', 'ebit_5yavgltm', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit 5yavgltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adj_fq', 'ebit_adj_fq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit Adj Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adj_fy', 'ebit_adj_fy', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit Adj Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adj_ltm', 'ebit_adj_ltm', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit Adj Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_cagr_3y', 'ebit_cagr_3y', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit Cagr 3y',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_fq', 'ebit_fq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_fy', 'ebit_fy', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_growth_yoy', 'ebit_growth_yoy', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit Growth Yoy',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_ltm', 'ebit_ltm', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_margin_ltm', 'ebit_margin_ltm', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit Margin Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_positive_years', 'ebit_positive_years', 'Profitability', 'calc_ebit_ebitda_comprehensive',
     'Ebit Positive Years',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_qoq_growth', 'ebit_qoq_growth', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit Qoq Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_vs_5y_avg', 'ebit_vs_5y_avg', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebit Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_1fqfq', 'ebitda_1fqfq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda 1fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_1fy', 'ebitda_1fy', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_2fqfq', 'ebitda_2fqfq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda 2fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_2fy', 'ebitda_2fy', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_3fqfq', 'ebitda_3fqfq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda 3fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_3fy', 'ebitda_3fy', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_4fqfq', 'ebitda_4fqfq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda 4fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_4fy', 'ebitda_4fy', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_5yavgfq', 'ebitda_5yavgfq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda 5yavgfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_5yavgltm', 'ebitda_5yavgltm', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda 5yavgltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adj_fq', 'ebitda_adj_fq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda Adj Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adj_fy', 'ebitda_adj_fy', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda Adj Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adj_ltm', 'ebitda_adj_ltm', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda Adj Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_cagr_3y', 'ebitda_cagr_3y', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda Cagr 3y',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_fq', 'ebitda_fq', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_fy', 'ebitda_fy', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_ltm', 'ebitda_ltm', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_margin_ltm', 'ebitda_margin_ltm', 'Profitability', 'calc_ebit_ebitda_comprehensive',
     'Ebitda Margin Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_positive_years', 'ebitda_positive_years', 'Profitability', 'calc_ebit_ebitda_comprehensive',
     'Ebitda Positive Years',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_qoq_growth', 'ebitda_qoq_growth', 'Profitability', 'calc_ebit_ebitda_comprehensive',
     'Ebitda Qoq Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_vs_5y_avg', 'ebitda_vs_5y_avg', 'Profitability', 'calc_ebit_ebitda_comprehensive', 'Ebitda Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- EPS TRAJECTORY (calc_eps_trajectory_features)
    ('feat_composite_eps_trajectory_score', 'composite_eps_trajectory_score', 'EPS Trajectory',
     'calc_eps_trajectory_features', 'Composite Eps Trajectory Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_growth_accel', 'eps_growth_accel', 'EPS Trajectory', 'calc_eps_trajectory_features', 'Eps Growth Accel',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_stability', 'eps_stability', 'EPS Trajectory', 'calc_eps_trajectory_features', 'Eps Stability',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_vs_5y_avg', 'eps_vs_5y_avg', 'EPS Trajectory', 'calc_eps_trajectory_features', 'Eps Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- VALUATION TIMESERIES (calc_extended_valuation_timeseries)
    ('feat_ev_ebitda_qoq_trend', 'ev_ebitda_qoq_trend', 'Valuation Timeseries', 'calc_extended_valuation_timeseries',
     'Ev Ebitda Qoq Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ev_sales_qoq_2q', 'ev_sales_qoq_2q', 'Valuation Timeseries', 'calc_extended_valuation_timeseries',
     'Ev Sales Qoq 2q',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ev_sales_qoq_3q', 'ev_sales_qoq_3q', 'Valuation Timeseries', 'calc_extended_valuation_timeseries',
     'Ev Sales Qoq 3q',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ev_sales_qoq_4q', 'ev_sales_qoq_4q', 'Valuation Timeseries', 'calc_extended_valuation_timeseries',
     'Ev Sales Qoq 4q',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_p_e_percentile_proxy', 'p_e_percentile_proxy', 'Valuation Timeseries', 'calc_extended_valuation_timeseries',
     'P E Percentile Proxy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_p_e_vs_5y_avg', 'p_e_vs_5y_avg', 'Valuation Timeseries', 'calc_extended_valuation_timeseries',
     'P E Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_valuation_compression', 'valuation_compression', 'Valuation Timeseries',
     'calc_extended_valuation_timeseries', 'Valuation Compression',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_valuation_mean_reversion', 'valuation_mean_reversion', 'Valuation Timeseries',
     'calc_extended_valuation_timeseries', 'Valuation Mean Reversion',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- TEMPORAL PATTERNS (calc_fiscal_calendar_features)
    ('feat_days_since_last_report', 'days_since_last_report', 'Temporal Patterns', 'calc_fiscal_calendar_features',
     'Days Since Last Report',
     NULL, NULL, 'difference', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_days_to_fy_end', 'days_to_fy_end', 'Temporal Patterns', 'calc_fiscal_calendar_features', 'Days To Fy End',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_earnings_season_flag', 'earnings_season_flag', 'Temporal Patterns', 'calc_fiscal_calendar_features',
     'Earnings Season Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_fiscal_quarter_progress', 'fiscal_quarter_progress', 'Temporal Patterns', 'calc_fiscal_calendar_features',
     'Fiscal Quarter Progress',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_is_fy_end_month', 'is_fy_end_month', 'Temporal Patterns', 'calc_fiscal_calendar_features', 'Is Fy End Month',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_is_quarter_end_month', 'is_quarter_end_month', 'Temporal Patterns', 'calc_fiscal_calendar_features',
     'Is Quarter End Month',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_post_earnings_window', 'post_earnings_window', 'Temporal Patterns', 'calc_fiscal_calendar_features',
     'Post Earnings Window',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pre_earnings_window', 'pre_earnings_window', 'Temporal Patterns', 'calc_fiscal_calendar_features',
     'Pre Earnings Window',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_reporting_freshness_score', 'reporting_freshness_score', 'Temporal Patterns',
     'calc_fiscal_calendar_features', 'Reporting Freshness Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    -- FINANCIAL DISTRESS (calc_financial_distress_features)
    ('feat_combined_distress_score', 'combined_distress_score', 'Financial Distress',
     'calc_financial_distress_features', 'Combined Distress Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_retained_earnings_growth', 'retained_earnings_growth', 'Financial Distress',
     'calc_financial_distress_features', 'Retained Earnings Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_deteriorating_flag', 'wc_deteriorating_flag', 'Financial Distress', 'calc_financial_distress_features',
     'Wc Deteriorating Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    -- CASH FLOW (calc_fcf_growth_estimates)
    ('feat_fcf_est_always_positive_fwd', 'fcf_est_always_positive_fwd', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Fcf Est Always Positive Fwd',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_cagr_5y_fwd', 'fcf_est_cagr_5y_fwd', 'Cash Flow', 'calc_fcf_growth_estimates', 'Fcf Est Cagr 5y Fwd',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_capex_implied_ratio', 'fcf_est_capex_implied_ratio', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Fcf Est Capex Implied Ratio',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_fy4', 'fcf_est_fy4', 'Cash Flow', 'calc_fcf_growth_estimates', 'Fcf Est Fy4',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_fy5', 'fcf_est_fy5', 'Cash Flow', 'calc_fcf_growth_estimates', 'Fcf Est Fy5',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_growth_deceleration', 'fcf_est_growth_deceleration', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Fcf Est Growth Deceleration',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_growth_fy3_vs_fy2', 'fcf_est_growth_fy3_vs_fy2', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Fcf Est Growth Fy3 Vs Fy2',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_growth_fy4_vs_fy3', 'fcf_est_growth_fy4_vs_fy3', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Fcf Est Growth Fy4 Vs Fy3',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_growth_fy5_vs_fy4', 'fcf_est_growth_fy5_vs_fy4', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Fcf Est Growth Fy5 Vs Fy4',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fcf_est_trajectory_score_fwd', 'fcf_est_trajectory_score_fwd', 'Cash Flow', 'calc_fcf_growth_estimates',
     'Fcf Est Trajectory Score Fwd',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- GAAP VS ADJUSTED (calc_gaap_adjusted_analytics)
    ('feat_earnings_quality_score', 'earnings_quality_score', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Earnings Quality Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_earnings_quality_warning', 'earnings_quality_warning', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Earnings Quality Warning',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adjustment_pct_1fqfq', 'ebit_adjustment_pct_1fqfq', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebit Adjustment Pct 1fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adjustment_pct_1fy', 'ebit_adjustment_pct_1fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebit Adjustment Pct 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adjustment_pct_2fqfq', 'ebit_adjustment_pct_2fqfq', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebit Adjustment Pct 2fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adjustment_pct_2fy', 'ebit_adjustment_pct_2fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebit Adjustment Pct 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adjustment_pct_3fqfq', 'ebit_adjustment_pct_3fqfq', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebit Adjustment Pct 3fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adjustment_pct_3fy', 'ebit_adjustment_pct_3fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebit Adjustment Pct 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adjustment_pct_4fqfq', 'ebit_adjustment_pct_4fqfq', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebit Adjustment Pct 4fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adjustment_pct_4fy', 'ebit_adjustment_pct_4fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebit Adjustment Pct 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adjustment_pct_fq', 'ebit_adjustment_pct_fq', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebit Adjustment Pct Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adjustment_pct_fy', 'ebit_adjustment_pct_fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebit Adjustment Pct Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_adjustment_pct_ltm', 'ebit_adjustment_pct_ltm', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebit Adjustment Pct Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adjustment_pct_1fqfq', 'ebitda_adjustment_pct_1fqfq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Ebitda Adjustment Pct 1fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adjustment_pct_1fy', 'ebitda_adjustment_pct_1fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebitda Adjustment Pct 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adjustment_pct_2fqfq', 'ebitda_adjustment_pct_2fqfq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Ebitda Adjustment Pct 2fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adjustment_pct_2fy', 'ebitda_adjustment_pct_2fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebitda Adjustment Pct 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adjustment_pct_3fqfq', 'ebitda_adjustment_pct_3fqfq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Ebitda Adjustment Pct 3fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adjustment_pct_3fy', 'ebitda_adjustment_pct_3fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebitda Adjustment Pct 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adjustment_pct_4fqfq', 'ebitda_adjustment_pct_4fqfq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Ebitda Adjustment Pct 4fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adjustment_pct_4fy', 'ebitda_adjustment_pct_4fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebitda Adjustment Pct 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adjustment_pct_fq', 'ebitda_adjustment_pct_fq', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebitda Adjustment Pct Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adjustment_pct_fy', 'ebitda_adjustment_pct_fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebitda Adjustment Pct Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_adjustment_pct_ltm', 'ebitda_adjustment_pct_ltm', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Ebitda Adjustment Pct Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_adjustment_pct', 'eps_adjustment_pct', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Eps Adjustment Pct',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_adjustment_spread_1fqfq', 'eps_adjustment_spread_1fqfq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Eps Adjustment Spread 1fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_adjustment_spread_1fy', 'eps_adjustment_spread_1fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Eps Adjustment Spread 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_adjustment_spread_2fqfq', 'eps_adjustment_spread_2fqfq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Eps Adjustment Spread 2fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_adjustment_spread_2fy', 'eps_adjustment_spread_2fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Eps Adjustment Spread 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_adjustment_spread_3fqfq', 'eps_adjustment_spread_3fqfq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Eps Adjustment Spread 3fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_adjustment_spread_3fy', 'eps_adjustment_spread_3fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Eps Adjustment Spread 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_adjustment_spread_4fqfq', 'eps_adjustment_spread_4fqfq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Eps Adjustment Spread 4fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_adjustment_spread_4fy', 'eps_adjustment_spread_4fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Eps Adjustment Spread 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_adjustment_spread_fq', 'eps_adjustment_spread_fq', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Eps Adjustment Spread Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_adjustment_spread_fy', 'eps_adjustment_spread_fy', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Eps Adjustment Spread Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_adjustment_spread_ltm', 'eps_adjustment_spread_ltm', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Eps Adjustment Spread Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_forward_eps_gaap_adj_spread', 'forward_eps_gaap_adj_spread', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Forward Eps Gaap Adj Spread',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adjustment_pct', 'net_income_adjustment_pct', 'GAAP vs Adjusted', 'calc_gaap_adjusted_analytics',
     'Net Income Adjustment Pct',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adjustment_ratio_1fqfq', 'net_income_adjustment_ratio_1fqfq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Net Income Adjustment Ratio 1fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adjustment_ratio_1fy', 'net_income_adjustment_ratio_1fy', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Net Income Adjustment Ratio 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adjustment_ratio_2fqfq', 'net_income_adjustment_ratio_2fqfq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Net Income Adjustment Ratio 2fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adjustment_ratio_2fy', 'net_income_adjustment_ratio_2fy', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Net Income Adjustment Ratio 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adjustment_ratio_3fqfq', 'net_income_adjustment_ratio_3fqfq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Net Income Adjustment Ratio 3fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adjustment_ratio_3fy', 'net_income_adjustment_ratio_3fy', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Net Income Adjustment Ratio 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adjustment_ratio_4fqfq', 'net_income_adjustment_ratio_4fqfq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Net Income Adjustment Ratio 4fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adjustment_ratio_4fy', 'net_income_adjustment_ratio_4fy', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Net Income Adjustment Ratio 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adjustment_ratio_5yavgfq', 'net_income_adjustment_ratio_5yavgfq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Net Income Adjustment Ratio 5yavgfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adjustment_ratio_fq', 'net_income_adjustment_ratio_fq', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Net Income Adjustment Ratio Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adjustment_ratio_fy', 'net_income_adjustment_ratio_fy', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Net Income Adjustment Ratio Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adjustment_ratio_ltm', 'net_income_adjustment_ratio_ltm', 'GAAP vs Adjusted',
     'calc_gaap_adjusted_analytics', 'Net Income Adjustment Ratio Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- GROWTH METRICS (calc_growth_features)
    ('feat_growth_ebitda_growth_yoy', 'growth_ebitda_growth_yoy', 'Growth Metrics', 'calc_growth_features',
     'Growth Ebitda Growth Yoy',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    -- PROFITABILITY (calc_gross_profit_temporal)
    ('feat_gp_1fqfq', 'gp_1fqfq', 'Profitability', 'calc_gross_profit_temporal', 'Gp 1fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_1fy', 'gp_1fy', 'Profitability', 'calc_gross_profit_temporal', 'Gp 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_2fqfq', 'gp_2fqfq', 'Profitability', 'calc_gross_profit_temporal', 'Gp 2fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_2fy', 'gp_2fy', 'Profitability', 'calc_gross_profit_temporal', 'Gp 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_3fqfq', 'gp_3fqfq', 'Profitability', 'calc_gross_profit_temporal', 'Gp 3fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_3fy', 'gp_3fy', 'Profitability', 'calc_gross_profit_temporal', 'Gp 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_4fqfq', 'gp_4fqfq', 'Profitability', 'calc_gross_profit_temporal', 'Gp 4fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_4fy', 'gp_4fy', 'Profitability', 'calc_gross_profit_temporal', 'Gp 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_fq', 'gp_fq', 'Profitability', 'calc_gross_profit_temporal', 'Gp Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_fy', 'gp_fy', 'Profitability', 'calc_gross_profit_temporal', 'Gp Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_ltm', 'gp_ltm', 'Profitability', 'calc_gross_profit_temporal', 'Gp Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_margin_expansion', 'gp_margin_expansion', 'Profitability', 'calc_gross_profit_temporal',
     'Gp Margin Expansion',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_margin_fq', 'gp_margin_fq', 'Profitability', 'calc_gross_profit_temporal', 'Gp Margin Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_margin_trend', 'gp_margin_trend', 'Profitability', 'calc_gross_profit_temporal', 'Gp Margin Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_positive_quarters', 'gp_positive_quarters', 'Profitability', 'calc_gross_profit_temporal',
     'Gp Positive Quarters',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_qoq_growth', 'gp_qoq_growth', 'Profitability', 'calc_gross_profit_temporal', 'Gp Qoq Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gp_yoy_growth', 'gp_yoy_growth', 'Profitability', 'calc_gross_profit_temporal', 'Gp Yoy Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    -- EARNINGS QUALITY (calc_gaap_revision_features)
    ('feat_gaap_positive_revision_flag', 'gaap_positive_revision_flag', 'Earnings Quality',
     'calc_gaap_revision_features', 'Gaap Positive Revision Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_gaap_revision_1m', 'gaap_revision_1m', 'Earnings Quality', 'calc_gaap_revision_features', 'Gaap Revision 1m',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gaap_revision_1y', 'gaap_revision_1y', 'Earnings Quality', 'calc_gaap_revision_features', 'Gaap Revision 1y',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gaap_revision_3m', 'gaap_revision_3m', 'Earnings Quality', 'calc_gaap_revision_features', 'Gaap Revision 3m',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gaap_revision_6m', 'gaap_revision_6m', 'Earnings Quality', 'calc_gaap_revision_features', 'Gaap Revision 6m',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gaap_revision_acceleration', 'gaap_revision_acceleration', 'Earnings Quality', 'calc_gaap_revision_features',
     'Gaap Revision Acceleration',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gaap_revision_momentum', 'gaap_revision_momentum', 'Earnings Quality', 'calc_gaap_revision_features',
     'Gaap Revision Momentum',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_gaap_vs_norm_revision_spread', 'gaap_vs_norm_revision_spread', 'Earnings Quality',
     'calc_gaap_revision_features', 'Gaap Vs Norm Revision Spread',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revision_quality_divergence', 'revision_quality_divergence', 'Earnings Quality',
     'calc_gaap_revision_features', 'Revision Quality Divergence',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    -- ACCOUNTING QUALITY (calc_goodwill_temporal_features)
    ('feat_goodwill_1fq', 'goodwill_1fq', 'Accounting Quality', 'calc_goodwill_temporal_features', 'Goodwill 1fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_1fy', 'goodwill_1fy', 'Accounting Quality', 'calc_goodwill_temporal_features', 'Goodwill 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_2fq', 'goodwill_2fq', 'Accounting Quality', 'calc_goodwill_temporal_features', 'Goodwill 2fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_2fy', 'goodwill_2fy', 'Accounting Quality', 'calc_goodwill_temporal_features', 'Goodwill 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_3fq', 'goodwill_3fq', 'Accounting Quality', 'calc_goodwill_temporal_features', 'Goodwill 3fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_3fy', 'goodwill_3fy', 'Accounting Quality', 'calc_goodwill_temporal_features', 'Goodwill 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_3y_growth', 'goodwill_3y_growth', 'Accounting Quality', 'calc_goodwill_temporal_features',
     'Goodwill 3y Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_4fq', 'goodwill_4fq', 'Accounting Quality', 'calc_goodwill_temporal_features', 'Goodwill 4fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_4fy', 'goodwill_4fy', 'Accounting Quality', 'calc_goodwill_temporal_features', 'Goodwill 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_accumulation_rate', 'goodwill_accumulation_rate', 'Accounting Quality',
     'calc_goodwill_temporal_features', 'Goodwill Accumulation Rate',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_concentration', 'goodwill_concentration', 'Accounting Quality', 'calc_goodwill_temporal_features',
     'Goodwill Concentration',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_fq', 'goodwill_fq', 'Accounting Quality', 'calc_goodwill_temporal_features', 'Goodwill Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_fy', 'goodwill_fy', 'Accounting Quality', 'calc_goodwill_temporal_features', 'Goodwill Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_ltm', 'goodwill_ltm', 'Accounting Quality', 'calc_goodwill_temporal_features', 'Goodwill Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_qoq_change', 'goodwill_qoq_change', 'Accounting Quality', 'calc_goodwill_temporal_features',
     'Goodwill Qoq Change',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_to_assets_trend', 'goodwill_to_assets_trend', 'Accounting Quality',
     'calc_goodwill_temporal_features', 'Goodwill To Assets Trend',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_vs_5y_avg', 'goodwill_vs_5y_avg', 'Accounting Quality', 'calc_goodwill_temporal_features',
     'Goodwill Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_yoy_change', 'goodwill_yoy_change', 'Accounting Quality', 'calc_goodwill_temporal_features',
     'Goodwill Yoy Change',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_impairment_risk_score', 'impairment_risk_score', 'Accounting Quality', 'calc_goodwill_temporal_features',
     'Impairment Risk Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_recent_acquisition_flag', 'recent_acquisition_flag', 'Accounting Quality', 'calc_goodwill_temporal_features',
     'Recent Acquisition Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    -- IDENTIFIER (direct)
    ('feat_description', 'description', 'Identifier', NULL, 'Description',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_dividend_record_announce_date', 'dividend_record_announce_date', 'Identifier', NULL,
     'Dividend Record Announce Date',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_dividend_record_ex_date', 'dividend_record_ex_date', 'Identifier', NULL, 'Dividend Record Ex Date',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_dividend_record_frequency', 'dividend_record_frequency', 'Identifier', NULL, 'Dividend Record Frequency',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_dividend_record_payable_date', 'dividend_record_payable_date', 'Identifier', NULL,
     'Dividend Record Payable Date',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_dividend_record_record_date', 'dividend_record_record_date', 'Identifier', NULL,
     'Dividend Record Record Date',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_earnings_report_frequency', 'earnings_report_frequency', 'Identifier', NULL, 'Earnings Report Frequency',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fy_end', 'fy_end', 'Identifier', NULL, 'Fy End',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_fy_end_date', 'fy_end_date', 'Identifier', NULL, 'Fy End Date',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_income_statement_report_date', 'income_statement_report_date', 'Identifier', NULL,
     'Income Statement Report Date',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_last_updated', 'last_updated', 'Identifier', NULL, 'Last Updated',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_next_earnings', 'next_earnings', 'Identifier', NULL, 'Next Earnings',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_next_earnings_report', 'next_earnings_report', 'Identifier', NULL, 'Next Earnings Report',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_next_earnings_status', 'next_earnings_status', 'Identifier', NULL, 'Next Earnings Status',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_next_earnings_when', 'next_earnings_when', 'Identifier', NULL, 'Next Earnings When',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_next_fiscal_quarter', 'next_fiscal_quarter', 'Identifier', NULL, 'Next Fiscal Quarter',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_next_fy_end_date', 'next_fy_end_date', 'Identifier', NULL, 'Next Fy End Date',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_next_income_statement_report_date', 'next_income_statement_report_date', 'Identifier', NULL,
     'Next Income Statement Report Date',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_reference_date', 'reference_date', 'Identifier', NULL, 'Reference Date',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_reporting_interval', 'reporting_interval', 'Identifier', NULL, 'Reporting Interval',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_unit', 'unit', 'Identifier', NULL, 'Unit',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- INTEREST INCOME (calc_interest_income_features)
    ('feat_interest_coverage_ratio', 'interest_coverage_ratio', 'Interest Income', 'calc_interest_income_features',
     'Interest Coverage Ratio',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_interest_expense_ltm', 'interest_expense_ltm', 'Interest Income', 'calc_interest_income_features',
     'Interest Expense Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_interest_expense_to_revenue', 'interest_expense_to_revenue', 'Interest Income',
     'calc_interest_income_features', 'Interest Expense To Revenue',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_interest_income_ltm', 'interest_income_ltm', 'Interest Income', 'calc_interest_income_features',
     'Interest Income Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_interest_income_to_revenue', 'interest_income_to_revenue', 'Interest Income',
     'calc_interest_income_features', 'Interest Income To Revenue',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_interest_income', 'net_interest_income', 'Interest Income', 'calc_interest_income_features',
     'Net Interest Income',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_interest_margin_proxy', 'net_interest_margin_proxy', 'Interest Income', 'calc_interest_income_features',
     'Net Interest Margin Proxy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- BALANCE SHEET (calc_inventory_temporal_features)
    ('feat_inventory_1fq', 'inventory_1fq', 'Balance Sheet', 'calc_inventory_temporal_features', 'Inventory 1fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_1fy', 'inventory_1fy', 'Balance Sheet', 'calc_inventory_temporal_features', 'Inventory 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_2fq', 'inventory_2fq', 'Balance Sheet', 'calc_inventory_temporal_features', 'Inventory 2fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_2fy', 'inventory_2fy', 'Balance Sheet', 'calc_inventory_temporal_features', 'Inventory 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_3fq', 'inventory_3fq', 'Balance Sheet', 'calc_inventory_temporal_features', 'Inventory 3fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_3fy', 'inventory_3fy', 'Balance Sheet', 'calc_inventory_temporal_features', 'Inventory 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_4fq', 'inventory_4fq', 'Balance Sheet', 'calc_inventory_temporal_features', 'Inventory 4fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_4fy', 'inventory_4fy', 'Balance Sheet', 'calc_inventory_temporal_features', 'Inventory 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_4q_trend', 'inventory_4q_trend', 'Balance Sheet', 'calc_inventory_temporal_features',
     'Inventory 4q Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_buildup_flag', 'inventory_buildup_flag', 'Balance Sheet', 'calc_inventory_temporal_features',
     'Inventory Buildup Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_inventory_days', 'inventory_days', 'Balance Sheet', 'calc_inventory_temporal_features', 'Inventory Days',
     NULL, NULL, 'difference', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_inventory_fq', 'inventory_fq', 'Balance Sheet', 'calc_inventory_temporal_features', 'Inventory Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_fy', 'inventory_fy', 'Balance Sheet', 'calc_inventory_temporal_features', 'Inventory Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_ltm', 'inventory_ltm', 'Balance Sheet', 'calc_inventory_temporal_features', 'Inventory Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_qoq_change', 'inventory_qoq_change', 'Balance Sheet', 'calc_inventory_temporal_features',
     'Inventory Qoq Change',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_reduction_flag', 'inventory_reduction_flag', 'Balance Sheet', 'calc_inventory_temporal_features',
     'Inventory Reduction Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_inventory_to_assets', 'inventory_to_assets', 'Balance Sheet', 'calc_inventory_temporal_features',
     'Inventory To Assets',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_to_revenue', 'inventory_to_revenue', 'Balance Sheet', 'calc_inventory_temporal_features',
     'Inventory To Revenue',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_turnover_itf', 'inventory_turnover_itf', 'Balance Sheet', 'calc_inventory_temporal_features',
     'Inventory Turnover Itf',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_volatility', 'inventory_volatility', 'Balance Sheet', 'calc_inventory_temporal_features',
     'Inventory Volatility',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_vs_5y_avg_itf', 'inventory_vs_5y_avg_itf', 'Balance Sheet', 'calc_inventory_temporal_features',
     'Inventory Vs 5y Avg Itf',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_inventory_yoy_change', 'inventory_yoy_change', 'Balance Sheet', 'calc_inventory_temporal_features',
     'Inventory Yoy Change',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    -- PROFITABILITY (calc_margin_trends)
    ('feat_margin_stability_score', 'margin_stability_score', 'Profitability', 'calc_margin_trends',
     'Margin Stability Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    -- EARNINGS QUALITY (calc_net_income_comprehensive)
    ('feat_earnings_quality_composite', 'earnings_quality_composite', 'Earnings Quality',
     'calc_net_income_comprehensive', 'Earnings Quality Composite',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_adj_ltm', 'net_income_adj_ltm', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Adj Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_growth_yoy', 'net_income_growth_yoy', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Growth Yoy',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_is_1fqfq', 'net_income_is_1fqfq', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Is 1fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_is_1fy', 'net_income_is_1fy', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Is 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_is_2fqfq', 'net_income_is_2fqfq', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Is 2fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_is_2fy', 'net_income_is_2fy', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Is 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_is_3fqfq', 'net_income_is_3fqfq', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Is 3fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_is_3fy', 'net_income_is_3fy', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Is 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_is_4fqfq', 'net_income_is_4fqfq', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Is 4fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_is_4fy', 'net_income_is_4fy', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Is 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_is_5yavgfq', 'net_income_is_5yavgfq', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Is 5yavgfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_is_5yavgltm', 'net_income_is_5yavgltm', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Is 5yavgltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_is_fq', 'net_income_is_fq', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Is Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_is_fy', 'net_income_is_fy', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Is Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_is_ltm', 'net_income_is_ltm', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Is Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_margin_ltm', 'net_income_margin_ltm', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Margin Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_positive_years', 'net_income_positive_years', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Positive Years',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_qoq_growth', 'net_income_qoq_growth', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Qoq Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_vs_5y_avg', 'net_income_vs_5y_avg', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_net_income_yoy_quarterly', 'net_income_yoy_quarterly', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Net Income Yoy Quarterly',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ni_adjustment_ratio', 'ni_adjustment_ratio', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Ni Adjustment Ratio',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_normalized_ni_5yavgfq', 'normalized_ni_5yavgfq', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Normalized Ni 5yavgfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_normalized_ni_5yavgltm', 'normalized_ni_5yavgltm', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Normalized Ni 5yavgltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_normalized_ni_ltm', 'normalized_ni_ltm', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Normalized Ni Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_normalized_ni_vs_5y_avg', 'normalized_ni_vs_5y_avg', 'Earnings Quality', 'calc_net_income_comprehensive',
     'Normalized Ni Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- PRICE TARGET DYNAMICS (calc_price_target_dynamics)
    ('feat_analyst_coverage_trend', 'analyst_coverage_trend', 'Price Target Dynamics', 'calc_price_target_dynamics',
     'Analyst Coverage Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_acceleration_long', 'pt_acceleration_long', 'Price Target Dynamics', 'calc_price_target_dynamics',
     'Pt Acceleration Long',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_acceleration_short', 'pt_acceleration_short', 'Price Target Dynamics', 'calc_price_target_dynamics',
     'Pt Acceleration Short',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_consensus_convergence', 'pt_consensus_convergence', 'Price Target Dynamics', 'calc_price_target_dynamics',
     'Pt Consensus Convergence',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_median_momentum_1m', 'pt_median_momentum_1m', 'Price Target Dynamics', 'calc_price_target_dynamics',
     'Pt Median Momentum 1m',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_median_momentum_3m', 'pt_median_momentum_3m', 'Price Target Dynamics', 'calc_price_target_dynamics',
     'Pt Median Momentum 3m',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_pt_vs_price_momentum', 'pt_vs_price_momentum', 'Price Target Dynamics', 'calc_price_target_dynamics',
     'Pt Vs Price Momentum',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    -- ACCOUNTING QUALITY (calc_quality_features_comprehensive)
    ('feat_asset_writedown_frequency', 'asset_writedown_frequency', 'Accounting Quality',
     'calc_quality_features_comprehensive', 'Asset Writedown Frequency',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_asset_writedown_ltm', 'asset_writedown_ltm', 'Accounting Quality', 'calc_quality_features_comprehensive',
     'Asset Writedown Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_exceptional_items_total_ltm', 'exceptional_items_total_ltm', 'Accounting Quality',
     'calc_quality_features_comprehensive', 'Exceptional Items Total Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_impairment_frequency', 'goodwill_impairment_frequency', 'Accounting Quality',
     'calc_quality_features_comprehensive', 'Goodwill Impairment Frequency',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_goodwill_impairment_ltm', 'goodwill_impairment_ltm', 'Accounting Quality',
     'calc_quality_features_comprehensive', 'Goodwill Impairment Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_has_goodwill_impairment_ltm', 'has_goodwill_impairment_ltm', 'Accounting Quality',
     'calc_quality_features_comprehensive', 'Has Goodwill Impairment Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_quality_issues_count_5y', 'quality_issues_count_5y', 'Accounting Quality',
     'calc_quality_features_comprehensive', 'Quality Issues Count 5y',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_restructuring_frequency', 'restructuring_frequency', 'Accounting Quality',
     'calc_quality_features_comprehensive', 'Restructuring Frequency',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_restructuring_ltm', 'restructuring_ltm', 'Accounting Quality', 'calc_quality_features_comprehensive',
     'Restructuring Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- REVENUE FORECASTING (calc_revenue_estimate_consensus)
    ('feat_revenue_avg_med_diff_pct', 'revenue_avg_med_diff_pct', 'Revenue Forecasting',
     'calc_revenue_estimate_consensus', 'Revenue Avg Med Diff Pct',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_consensus_strength', 'revenue_consensus_strength', 'Revenue Forecasting',
     'calc_revenue_estimate_consensus', 'Revenue Consensus Strength',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_est_avg_fy1e', 'revenue_est_avg_fy1e', 'Revenue Forecasting', 'calc_revenue_estimate_consensus',
     'Revenue Est Avg Fy1e',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_est_avg_ntm', 'revenue_est_avg_ntm', 'Revenue Forecasting', 'calc_revenue_estimate_consensus',
     'Revenue Est Avg Ntm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_est_med_fy1e', 'revenue_est_med_fy1e', 'Revenue Forecasting', 'calc_revenue_estimate_consensus',
     'Revenue Est Med Fy1e',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_est_med_ntm', 'revenue_est_med_ntm', 'Revenue Forecasting', 'calc_revenue_estimate_consensus',
     'Revenue Est Med Ntm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_revision_trend', 'revenue_revision_trend', 'Revenue Forecasting', 'calc_revenue_estimate_consensus',
     'Revenue Revision Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_vs_current', 'revenue_vs_current', 'Revenue Forecasting', 'calc_revenue_estimate_consensus',
     'Revenue Vs Current',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_consensus_revenue_growth', 'consensus_revenue_growth', 'Revenue Forecasting',
     'calc_revenue_forecast_features', 'Consensus Revenue Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebit_estimate_spread', 'ebit_estimate_spread', 'Revenue Forecasting', 'calc_revenue_forecast_features',
     'Ebit Estimate Spread',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_ebitda_est_vs_actual', 'ebitda_est_vs_actual', 'Revenue Forecasting', 'calc_revenue_forecast_features',
     'Ebitda Est Vs Actual',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_estimate_confidence_score', 'estimate_confidence_score', 'Revenue Forecasting',
     'calc_revenue_forecast_features', 'Estimate Confidence Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_forward_ebitda_margin', 'forward_ebitda_margin', 'Revenue Forecasting', 'calc_revenue_forecast_features',
     'Forward Ebitda Margin',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_forward_revenue_multiple', 'forward_revenue_multiple', 'Revenue Forecasting',
     'calc_revenue_forecast_features', 'Forward Revenue Multiple',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_acceleration', 'revenue_acceleration', 'Revenue Forecasting', 'calc_revenue_forecast_features',
     'Revenue Acceleration',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_beat_potential', 'revenue_beat_potential', 'Revenue Forecasting', 'calc_revenue_forecast_features',
     'Revenue Beat Potential',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_est_revision_trend', 'revenue_est_revision_trend', 'Revenue Forecasting',
     'calc_revenue_forecast_features', 'Revenue Est Revision Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_est_spread', 'revenue_est_spread', 'Revenue Forecasting', 'calc_revenue_forecast_features',
     'Revenue Est Spread',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_estimate_count', 'revenue_estimate_count', 'Revenue Forecasting', 'calc_revenue_forecast_features',
     'Revenue Estimate Count',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_guidance_gap', 'revenue_guidance_gap', 'Revenue Forecasting', 'calc_revenue_forecast_features',
     'Revenue Guidance Gap',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_1fqfq', 'revenue_1fqfq', 'Revenue Forecasting', 'calc_revenue_quarterly_features', 'Revenue 1fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_1fy', 'revenue_1fy', 'Revenue Forecasting', 'calc_revenue_quarterly_features', 'Revenue 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_2fqfq', 'revenue_2fqfq', 'Revenue Forecasting', 'calc_revenue_quarterly_features', 'Revenue 2fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_2fy', 'revenue_2fy', 'Revenue Forecasting', 'calc_revenue_quarterly_features', 'Revenue 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_2y_growth', 'revenue_2y_growth', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue 2y Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_3fqfq', 'revenue_3fqfq', 'Revenue Forecasting', 'calc_revenue_quarterly_features', 'Revenue 3fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_3fy', 'revenue_3fy', 'Revenue Forecasting', 'calc_revenue_quarterly_features', 'Revenue 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_3y_growth', 'revenue_3y_growth', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue 3y Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_4fqfq', 'revenue_4fqfq', 'Revenue Forecasting', 'calc_revenue_quarterly_features', 'Revenue 4fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_4fy', 'revenue_4fy', 'Revenue Forecasting', 'calc_revenue_quarterly_features', 'Revenue 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_4q_avg', 'revenue_4q_avg', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue 4q Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_4q_trend', 'revenue_4q_trend', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue 4q Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_4y_growth', 'revenue_4y_growth', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue 4y Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_5y_avg', 'revenue_5y_avg', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_accelerating_flag', 'revenue_accelerating_flag', 'Revenue Forecasting',
     'calc_revenue_quarterly_features', 'Revenue Accelerating Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_revenue_cagr_3y', 'revenue_cagr_3y', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue Cagr 3y',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_cagr_4y', 'revenue_cagr_4y', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue Cagr 4y',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_fq', 'revenue_fq', 'Revenue Forecasting', 'calc_revenue_quarterly_features', 'Revenue Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_fq_vs_4q_avg', 'revenue_fq_vs_4q_avg', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue Fq Vs 4q Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_fy', 'revenue_fy', 'Revenue Forecasting', 'calc_revenue_quarterly_features', 'Revenue Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_growth_flag', 'revenue_growth_flag', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue Growth Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_revenue_ltm', 'revenue_ltm', 'Revenue Forecasting', 'calc_revenue_quarterly_features', 'Revenue Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_positive_qoq_streak', 'revenue_positive_qoq_streak', 'Revenue Forecasting',
     'calc_revenue_quarterly_features', 'Revenue Positive Qoq Streak',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_qoq_2q', 'revenue_qoq_2q', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue Qoq 2q',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_qoq_3q', 'revenue_qoq_3q', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue Qoq 3q',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_qoq_4q', 'revenue_qoq_4q', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue Qoq 4q',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_qoq_growth', 'revenue_qoq_growth', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue Qoq Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_stability_score', 'revenue_stability_score', 'Revenue Forecasting',
     'calc_revenue_quarterly_features', 'Revenue Stability Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_yoy_quarterly', 'revenue_yoy_quarterly', 'Revenue Forecasting', 'calc_revenue_quarterly_features',
     'Revenue Yoy Quarterly',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- EFFICIENCY RATIOS (calc_rnd_temporal_features)
    ('feat_high_rnd_intensity_flag', 'high_rnd_intensity_flag', 'Efficiency Ratios', 'calc_rnd_temporal_features',
     'High Rnd Intensity Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_rnd_1fqfq', 'rnd_1fqfq', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd 1fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_1fy', 'rnd_1fy', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_2fqfq', 'rnd_2fqfq', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd 2fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_2fy', 'rnd_2fy', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_3fqfq', 'rnd_3fqfq', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd 3fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_3fy', 'rnd_3fy', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_4fqfq', 'rnd_4fqfq', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd 4fqfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_4fy', 'rnd_4fy', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_cagr_3y', 'rnd_cagr_3y', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd Cagr 3y',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_cut_flag', 'rnd_cut_flag', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd Cut Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_rnd_fq', 'rnd_fq', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_fy', 'rnd_fy', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_increasing_flag', 'rnd_increasing_flag', 'Efficiency Ratios', 'calc_rnd_temporal_features',
     'Rnd Increasing Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_rnd_intensity_fy', 'rnd_intensity_fy', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd Intensity Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_intensity_ltm', 'rnd_intensity_ltm', 'Efficiency Ratios', 'calc_rnd_temporal_features',
     'Rnd Intensity Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_intensity_trend', 'rnd_intensity_trend', 'Efficiency Ratios', 'calc_rnd_temporal_features',
     'Rnd Intensity Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_ltm', 'rnd_ltm', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_per_employee', 'rnd_per_employee', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd Per Employee',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_qoq_growth', 'rnd_qoq_growth', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd Qoq Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_roi_proxy', 'rnd_roi_proxy', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd Roi Proxy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_to_gross_profit', 'rnd_to_gross_profit', 'Efficiency Ratios', 'calc_rnd_temporal_features',
     'Rnd To Gross Profit',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rnd_yoy_growth', 'rnd_yoy_growth', 'Efficiency Ratios', 'calc_rnd_temporal_features', 'Rnd Yoy Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    -- ANALYST SENTIMENT (calc_sentiment_features)
    ('feat_analyst_conviction', 'analyst_conviction', 'Analyst Sentiment', 'calc_sentiment_features',
     'Analyst Conviction',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- BALANCE SHEET (calc_total_assets_temporal)
    ('feat_asset_base_stable', 'asset_base_stable', 'Balance Sheet', 'calc_total_assets_temporal', 'Asset Base Stable',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_asset_growth_accel', 'asset_growth_accel', 'Balance Sheet', 'calc_total_assets_temporal',
     'Asset Growth Accel',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_1fq', 'assets_1fq', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets 1fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_1fy', 'assets_1fy', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_2fq', 'assets_2fq', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets 2fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_2fy', 'assets_2fy', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_3fq', 'assets_3fq', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets 3fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_3fy', 'assets_3fy', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_3y_cagr', 'assets_3y_cagr', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets 3y Cagr',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_4fq', 'assets_4fq', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets 4fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_4fy', 'assets_4fy', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_fq', 'assets_fq', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_fy', 'assets_fy', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_ltm', 'assets_ltm', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_qoq_growth', 'assets_qoq_growth', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets Qoq Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_assets_yoy_growth', 'assets_yoy_growth', 'Balance Sheet', 'calc_total_assets_temporal', 'Assets Yoy Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    -- VALUATION RATIOS (calc_tangible_book_features)
    ('feat_tangible_asset_quality', 'tangible_asset_quality', 'Valuation Ratios', 'calc_tangible_book_features',
     'Tangible Asset Quality',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_tangible_book_value_fy', 'tangible_book_value_fy', 'Valuation Ratios', 'calc_tangible_book_features',
     'Tangible Book Value Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_tangible_book_value_ltm', 'tangible_book_value_ltm', 'Valuation Ratios', 'calc_tangible_book_features',
     'Tangible Book Value Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_tbv_vs_calculated', 'tbv_vs_calculated', 'Valuation Ratios', 'calc_tangible_book_features',
     'Tbv Vs Calculated',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_tbv_yoy_growth', 'tbv_yoy_growth', 'Valuation Ratios', 'calc_tangible_book_features', 'Tbv Yoy Growth',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    -- LEVERAGE & LIQUIDITY (calc_total_debt_temporal)
    ('feat_debt_1fq', 'debt_1fq', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt 1fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_1fy', 'debt_1fy', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_2fq', 'debt_2fq', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt 2fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_2fy', 'debt_2fy', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_3fq', 'debt_3fq', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt 3fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_3fy', 'debt_3fy', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_3y_cagr', 'debt_3y_cagr', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt 3y Cagr',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_4fq', 'debt_4fq', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt 4fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_4fy', 'debt_4fy', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_4q_trend', 'debt_4q_trend', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt 4q Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_deleveraging', 'debt_deleveraging', 'Leverage & Liquidity', 'calc_total_debt_temporal',
     'Debt Deleveraging',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_fq', 'debt_fq', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_fy', 'debt_fy', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_ltm', 'debt_ltm', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_qoq_change', 'debt_qoq_change', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt Qoq Change',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_to_equity_trend', 'debt_to_equity_trend', 'Leverage & Liquidity', 'calc_total_debt_temporal',
     'Debt To Equity Trend',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_debt_yoy_change', 'debt_yoy_change', 'Leverage & Liquidity', 'calc_total_debt_temporal', 'Debt Yoy Change',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    -- GROWTH METRICS (calc_total_revenues_temporal)
    ('feat_revenue_5yavgfq', 'revenue_5yavgfq', 'Growth Metrics', 'calc_total_revenues_temporal', 'Revenue 5yavgfq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_5yavgltm', 'revenue_5yavgltm', 'Growth Metrics', 'calc_total_revenues_temporal', 'Revenue 5yavgltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_fq_vs_avg', 'revenue_fq_vs_avg', 'Growth Metrics', 'calc_total_revenues_temporal',
     'Revenue Fq Vs Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_momentum', 'revenue_momentum', 'Growth Metrics', 'calc_total_revenues_temporal', 'Revenue Momentum',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_vs_5y_avg_fq', 'revenue_vs_5y_avg_fq', 'Growth Metrics', 'calc_total_revenues_temporal',
     'Revenue Vs 5y Avg Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_revenue_vs_5y_avg_ltm', 'revenue_vs_5y_avg_ltm', 'Growth Metrics', 'calc_total_revenues_temporal',
     'Revenue Vs 5y Avg Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    -- EARNINGS QUALITY (calc_unusual_items_features)
    ('feat_earnings_quality_impact', 'earnings_quality_impact', 'Earnings Quality', 'calc_unusual_items_features',
     'Earnings Quality Impact',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_has_unusual_items_flag', 'has_unusual_items_flag', 'Earnings Quality', 'calc_unusual_items_features',
     'Has Unusual Items Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_impairment_goodwill_ltm', 'impairment_goodwill_ltm', 'Earnings Quality', 'calc_unusual_items_features',
     'Impairment Goodwill Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_other_unusual_items_ltm', 'other_unusual_items_ltm', 'Earnings Quality', 'calc_unusual_items_features',
     'Other Unusual Items Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_restructuring_charges_ltm', 'restructuring_charges_ltm', 'Earnings Quality', 'calc_unusual_items_features',
     'Restructuring Charges Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_total_unusual_items', 'total_unusual_items', 'Earnings Quality', 'calc_unusual_items_features',
     'Total Unusual Items',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_unusual_asset_writedown_ltm', 'unusual_asset_writedown_ltm', 'Earnings Quality',
     'calc_unusual_items_features', 'Unusual Asset Writedown Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_unusual_items_to_ebitda', 'unusual_items_to_ebitda', 'Earnings Quality', 'calc_unusual_items_features',
     'Unusual Items To Ebitda',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_unusual_items_to_revenue', 'unusual_items_to_revenue', 'Earnings Quality', 'calc_unusual_items_features',
     'Unusual Items To Revenue',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    -- LEVERAGE & LIQUIDITY (calc_working_capital_deep_features)
    ('feat_days_working_capital', 'days_working_capital', 'Leverage & Liquidity', 'calc_working_capital_deep_features',
     'Days Working Capital',
     NULL, NULL, 'difference', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_negative_wc_flag', 'negative_wc_flag', 'Leverage & Liquidity', 'calc_working_capital_deep_features',
     'Negative Wc Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_wc_efficiency_score', 'wc_efficiency_score', 'Leverage & Liquidity', 'calc_working_capital_deep_features',
     'Wc Efficiency Score',
     NULL, NULL, 'score', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_to_assets', 'wc_to_assets', 'Leverage & Liquidity', 'calc_working_capital_deep_features', 'Wc To Assets',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_to_revenue', 'wc_to_revenue', 'Leverage & Liquidity', 'calc_working_capital_deep_features',
     'Wc To Revenue',
     NULL, NULL, 'ratio', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_1fq', 'wc_1fq', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc 1fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_1fy', 'wc_1fy', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc 1fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_2fq', 'wc_2fq', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc 2fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_2fy', 'wc_2fy', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc 2fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_3fq', 'wc_3fq', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc 3fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_3fy', 'wc_3fy', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc 3fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_4fq', 'wc_4fq', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc 4fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_4fy', 'wc_4fy', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc 4fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_4q_trend', 'wc_4q_trend', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc 4q Trend',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_5yavgfy', 'wc_5yavgfy', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc 5yavgfy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_fq', 'wc_fq', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc Fq',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_fy', 'wc_fy', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc Fy',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_improving_flag', 'wc_improving_flag', 'Leverage & Liquidity', 'calc_working_capital_temporal',
     'Wc Improving Flag',
     NULL, NULL, 'flag', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_wc_ltm', 'wc_ltm', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc Ltm',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_positive_quarters', 'wc_positive_quarters', 'Leverage & Liquidity', 'calc_working_capital_temporal',
     'Wc Positive Quarters',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_qoq_change', 'wc_qoq_change', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc Qoq Change',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_volatility', 'wc_volatility', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc Volatility',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_vs_5y_avg', 'wc_vs_5y_avg', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc Vs 5y Avg',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_wc_yoy_change', 'wc_yoy_change', 'Leverage & Liquidity', 'calc_working_capital_temporal', 'Wc Yoy Change',
     NULL, NULL, 'growth', 'NUMERIC', CURRENT_TIMESTAMP),

    -- DIRECT EQUITIES COLUMNS (Market Data & Analyst Estimates)
    ('feat_analyst_rating', 'analyst_rating', 'Market Data', NULL, 'Analyst rating consensus',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_beta_short_term_shift', 'beta_short_term_shift', 'Market Data', NULL, 'Beta short-term shift (1Y - 2Y)',
     NULL, NULL, 'difference', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_current_fiscal_quarter', 'current_fiscal_quarter', 'Market Data', NULL, 'Current fiscal quarter',
     NULL, NULL, 'direct', 'TEXT', CURRENT_TIMESTAMP),
    ('feat_dividend_per_share_ltm', 'dividend_per_share_ltm', 'Market Data', NULL, 'Dividend per share (LTM)',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_dividend_record_amount', 'dividend_record_amount', 'Market Data', NULL, 'Dividend record amount',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_dividend_record_currency', 'dividend_record_currency', 'Market Data', NULL, 'Dividend record currency',
     NULL, NULL, 'direct', 'TEXT', CURRENT_TIMESTAMP),
    ('feat_eps_gaap_est_avg_fy1e', 'eps_gaap_est_avg_fy1e', 'Market Data', NULL, 'EPS GAAP estimated average (FY1E)',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_gaap_est_avg_ntm', 'eps_gaap_est_avg_ntm', 'Market Data', NULL, 'EPS GAAP estimated average (NTM)',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_norm_est_avg_fy1e', 'eps_norm_est_avg_fy1e', 'Market Data', NULL,
     'EPS normalized estimated average (FY1E)',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_norm_est_avg_ntm', 'eps_norm_est_avg_ntm', 'Market Data', NULL, 'EPS normalized estimated average (NTM)',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_eps_norm_est_num_fy1e', 'eps_norm_est_num_fy1e', 'Market Data', NULL, 'EPS normalized estimate count (FY1E)',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_num_buys_ratings', 'num_buys_ratings', 'Market Data', NULL, 'Number of buy ratings',
     NULL, NULL, 'direct', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_num_hold_ratings', 'num_hold_ratings', 'Market Data', NULL, 'Number of hold ratings',
     NULL, NULL, 'direct', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_num_sell_ratings', 'num_sell_ratings', 'Market Data', NULL, 'Number of sell ratings',
     NULL, NULL, 'direct', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_num_strong_buys_ratings', 'num_strong_buys_ratings', 'Market Data', NULL, 'Number of strong buy ratings',
     NULL, NULL, 'direct', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_num_strong_sell_ratings', 'num_strong_sell_ratings', 'Market Data', NULL, 'Number of strong sell ratings',
     NULL, NULL, 'direct', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_one_day_pct', 'one_day_pct', 'Market Data', NULL, 'One-day price change percentage',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_price_target', 'price_target', 'Market Data', NULL, 'Consensus price target',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_price_target_count', 'price_target_count', 'Market Data', NULL, 'Number of analyst price targets',
     NULL, NULL, 'direct', 'INTEGER', CURRENT_TIMESTAMP),
    ('feat_price_target_high', 'price_target_high', 'Market Data', NULL, 'Highest analyst price target',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_price_target_low', 'price_target_low', 'Market Data', NULL, 'Lowest analyst price target',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_price_target_median', 'price_target_median', 'Market Data', NULL, 'Median analyst price target',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_rel_volume', 'rel_volume', 'Market Data', NULL, 'Relative trading volume',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_tot_return_pct_cagr_10y', 'tot_return_pct_cagr_10y', 'Market Data', NULL, 'Total return CAGR (10Y)',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_tot_return_pct_cagr_3y', 'tot_return_pct_cagr_3y', 'Market Data', NULL, 'Total return CAGR (3Y)',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_total_return_10y', 'total_return_10y', 'Market Data', NULL, 'Total return (10Y)',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_total_return_5y', 'total_return_5y', 'Market Data', NULL, 'Total return (5Y)',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_total_return_ytd', 'total_return_ytd', 'Market Data', NULL, 'Total return (YTD)',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP),
    ('feat_total_revenues_cagr_5y_fy', 'total_revenues_cagr_5y_fy', 'Market Data', NULL, 'Total revenues CAGR (5Y FY)',
     NULL, NULL, 'direct', 'NUMERIC', CURRENT_TIMESTAMP)


ON CONFLICT (feature_key) DO UPDATE SET feature_alias      = EXCLUDED.feature_alias,
                                        category           = EXCLUDED.category,
                                        source_function    = EXCLUDED.source_function,
                                        description        = EXCLUDED.description,
                                        source_columns     = EXCLUDED.source_columns,
                                        primary_source_col = EXCLUDED.primary_source_col,
                                        calculation_type   = EXCLUDED.calculation_type,
                                        data_type          = EXCLUDED.data_type;

COMMIT;

-- Refresh table statistics for optimal query planning
ANALYZE feature_registry_metadata;
ANALYZE calculated_features_registry;


-- Function to get feature count summary by category
CREATE OR REPLACE FUNCTION get_feature_registry_summary()
    RETURNS TABLE
            (
                category       TEXT,
                function_count INTEGER,
                total_features INTEGER
            )
AS
$$
SELECT category,
       COUNT(*)::INTEGER                        AS function_count,
       SUM(COALESCE(feature_count, 0))::INTEGER AS total_features
FROM feature_registry_metadata
GROUP BY category
ORDER BY total_features DESC;
$$ LANGUAGE SQL STABLE;









