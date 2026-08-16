-- =============================================================================
-- SQL Feature Catalogue for PML Finance Project
-- PyMC Feature Engineering - PostgreSQL Implementation
-- =============================================================================
-- *** SINGLE SOURCE OF TRUTH ***
--
-- This file is authoritative for, in order:
--   1. the pml.* helper FUNCTIONS (arithmetic, transforms, domain, fiscal),
--   2. all seven pml.mv_pymc_* MATERIALIZED VIEWS (the per-model feature
--      matrices — this is where every feat_* / observed_* / n_* column is
--      actually defined),
--   3. the catalogue VIEWS (vw_pymc_feature_catalogue / _aliases / _coverage),
--   4. the coverage regression check + pml.refresh_pymc_materialized_views().
--
-- The per-object files under sql_scripts/pml/ are pg_dump-style EXTRACTS, not
-- sources: every mv_pymc_*.sql, every vw_pymc_*.sql and all 37 function files
-- there carry a `-- missing source code` body and cannot recreate anything.
-- They are authoritative only for the base TABLES (pml_df.sql, staging.sql,
-- pml_df_metadata.sql, pml_df_feature_alias.sql) and the five vw_pml_df_*
-- views. Edit the definitions HERE.
--
-- Companion SSOT files:
--   * pml_df_metadata.sql          -- metadata/alias table DDL + vocabularies
--   * pml_df_metadata_populate.sql -- pymc_role / model_targets assignment
--
-- NOTE (rebuild order): recreating pml.pml_df cascade-drops every
-- mv_pymc_* that depends on it, so a pml_df rebuild must be followed by
-- re-running this file to recreate the MVs and catalogue views.
-- =============================================================================
-- OPTIMIZATIONS APPLIED:
-- 1. STABLE modifier on all functions (enables query optimizer caching)
-- 2. Optional isin parameter for filtered access (uses pml_df_isin_pk)
-- 3. Materialized views for pymc (uses pml_df_materialized_view)
-- 4. PARALLEL SAFE where applicable
-- 5. Helper functions for common calculations (DRY principle)
--
-- Volatility caveat: most helpers are IMMUTABLE PARALLEL SAFE with paired
-- NUMERIC + DOUBLE PRECISION overloads, but NOT all --
-- pml.calc_piotroski_f_score is STABLE and single-overload (it reads pml_df),
-- as are the country_name / currency_name / exchange_name lookups.
-- =============================================================================

-- =============================================================================
-- HELPER FUNCTIONS: Extracted Common Calculations
-- =============================================================================

-- Safe division helper (avoids division by zero)
CREATE OR REPLACE FUNCTION safe_divide(
	numerator   NUMERIC,
	denominator NUMERIC
) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT numerator / NULLIF(denominator, 0) AS result;
$$;

-- Percentage change helper
CREATE OR REPLACE FUNCTION pml.pct_change(current_val NUMERIC, previous_val NUMERIC) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) * 100 AS result;
$$
	LANGUAGE sql;

-- Momentum/change ratio helper (without percentage multiplier)
CREATE OR REPLACE FUNCTION pml.calc_change_ratio(current_val NUMERIC, previous_val NUMERIC) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) AS result;
$$
	LANGUAGE sql;

-- Score clamping helper (constrains value between 0 and 100)
CREATE OR REPLACE FUNCTION pml.clamp_score(val NUMERIC, min_val NUMERIC DEFAULT 0, max_val NUMERIC DEFAULT 100) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE AS
$$
SELECT GREATEST(min_val, LEAST(max_val, val)) AS result;
$$
	LANGUAGE sql;

-- EMA crossover signal helper
CREATE OR REPLACE FUNCTION pml.ema_crossover_signal(fast_ema NUMERIC, slow_ema NUMERIC) RETURNS INTEGER
	IMMUTABLE PARALLEL SAFE AS
$$
SELECT CASE WHEN fast_ema > slow_ema THEN 1 WHEN fast_ema < slow_ema THEN -1 ELSE 0 END AS result;
$$
	LANGUAGE sql;

-- =============================================================================
-- DOUBLE PRECISION OVERLOADS
-- =============================================================================
-- pml.pml_df columns are stored as DOUBLE PRECISION. PostgreSQL does not
-- implicitly cast DOUBLE PRECISION -> NUMERIC, so we expose overloads that
-- accept DOUBLE PRECISION directly. This eliminates "function does not exist"
-- errors (42883) when the materialized views below pass raw column values.
-- =============================================================================

CREATE OR REPLACE FUNCTION pml.safe_divide(numerator   DOUBLE PRECISION,
                                           denominator DOUBLE PRECISION) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT numerator / NULLIF(denominator, 0);
$$;

CREATE OR REPLACE FUNCTION pml.pct_change(current_val  DOUBLE PRECISION,
                                          previous_val DOUBLE PRECISION) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) * 100;
$$;

CREATE OR REPLACE FUNCTION pml.calc_change_ratio(current_val  DOUBLE PRECISION,
                                                 previous_val DOUBLE PRECISION) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0);
$$;

CREATE OR REPLACE FUNCTION pml.clamp_score(val     DOUBLE PRECISION,
                                           min_val DOUBLE PRECISION DEFAULT 0,
                                           max_val DOUBLE PRECISION DEFAULT 100) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT GREATEST(min_val, LEAST(max_val, val));
$$;

CREATE OR REPLACE FUNCTION pml.ema_crossover_signal(fast_ema DOUBLE PRECISION,
                                                    slow_ema DOUBLE PRECISION) RETURNS INTEGER
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN fast_ema > slow_ema THEN 1 WHEN fast_ema < slow_ema THEN -1 ELSE 0 END;
$$;

-- =============================================================================
-- PYMC FEATURE ENGINEERING FUNCTIONS
-- =============================================================================
-- These functions translate raw columns from pml.pml_df into PyMC-ready
-- numeric features keyed by `pymc_role` and `model_targets` defined in
-- pml.pml_df_metadata. Each function is STABLE + PARALLEL SAFE and accepts
-- an optional ISIN filter for fast per-stock invocation (uses pml_df PK).
-- =============================================================================

-- Logit transform with clipping (for Beta-Binomial / logit-Normal priors).
CREATE OR REPLACE FUNCTION pml.safe_logit(p NUMERIC, eps NUMERIC DEFAULT 1e-6) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT LN(GREATEST(eps, LEAST(1 - eps, p)) / (1 - GREATEST(eps, LEAST(1 - eps, p))));
$$;

-- Robust z-score within a hierarchical group (e.g. sector / industry).
-- Used as a standardised PyMC `mutable_predictor` input.
CREATE OR REPLACE FUNCTION pml.zscore(val NUMERIC, mu NUMERIC, sigma NUMERIC) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (val - mu) / NULLIF(sigma, 0);
$$;

-- Winsorise to [lo, hi] percentile bounds for tail-robust priors.
CREATE OR REPLACE FUNCTION pml.winsorise(val NUMERIC, lo NUMERIC, hi NUMERIC) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT GREATEST(lo, LEAST(hi, val));
$$;

-- Count of historical "beats" across an array of surprise % values
-- (positive surprise => beat). Drives EarningsBeatBayesian n_beats / n_total.
CREATE OR REPLACE FUNCTION pml.beat_counts(surprises NUMERIC[])
	RETURNS TABLE
	        (
		        n_total INT,
		        n_beats INT
	        )
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT SUM(CASE WHEN s IS NOT NULL THEN 1 ELSE 0 END)::INT           AS n_total,
       SUM(CASE WHEN s IS NOT NULL AND s > 0 THEN 1 ELSE 0 END)::INT AS n_beats
FROM UNNEST(surprises) AS s;
$$;

-- Drift (mean) of a price-target snapshot trail (current + 1w/1m/3m/6m/1y ago).
-- Feeds KalmanFilterPriceTarget momentum-informed priors.
CREATE OR REPLACE FUNCTION pml.target_drift(arr NUMERIC[]) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT AVG(pml.calc_change_ratio(arr[i], arr[i + 1]))
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1);
$$;

-- Coefficient of variation (sigma / |mu|) for analyst dispersion priors.
CREATE OR REPLACE FUNCTION pml.coef_var(mu NUMERIC, sigma NUMERIC) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT sigma / NULLIF(ABS(mu), 0);
$$;

-- FCF dividend-coverage ratio (>1 = safe). Feeds DividendSafetyBayesian.
CREATE OR REPLACE FUNCTION pml.fcf_dividend_coverage(fcf NUMERIC, dividends_paid NUMERIC) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT pml.safe_divide(fcf, ABS(dividends_paid));
$$;

-- Altman Z-score distress class (1=distress<1.81, 2=grey<2.99, 3=safe).
CREATE OR REPLACE FUNCTION pml.altman_zone(z NUMERIC) RETURNS INT
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN z IS NULL THEN NULL WHEN z < 1.81 THEN 1 WHEN z < 2.99 THEN 2 ELSE 3 END;
$$;

-- Accruals proxy (NI - CFO) / Assets-proxy; uses EV as denominator if assets
-- unavailable. Feeds AccountingAnomalyBayesian.
CREATE OR REPLACE FUNCTION pml.accruals_ratio(ni NUMERIC, cfo NUMERIC, scale NUMERIC) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT pml.safe_divide(ni - cfo, NULLIF(scale, 0));
$$;

-- Piotroski F-score (0-9): 9-signal fundamental-quality composite (positive
-- ROA / CFO, rising ROA, accruals quality CFO > NI, de-leveraging, rising
-- liquidity, no dilution, rising gross margin, rising asset turnover).
-- NULL-tolerant: a NULL comparison scores 0 for that signal (never NULL
-- overall). Called 4x per row by mv_pymc_kalman_pt (fy vs neg1fy .. neg3fy vs
-- neg4fy lag pairs) and by calc_piotroski_f_score (LTM screener variant).
CREATE OR REPLACE FUNCTION pml.piotroski_f_score(roa NUMERIC, roa_prev NUMERIC,
                                                 cfo NUMERIC, ni NUMERIC,
                                                 ltde NUMERIC, ltde_prev NUMERIC,
                                                 cr NUMERIC, cr_prev NUMERIC,
                                                 shrs NUMERIC, shrs_prev NUMERIC,
                                                 gpm NUMERIC, gpm_prev NUMERIC,
                                                 at NUMERIC, at_prev NUMERIC) RETURNS INTEGER
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (CASE WHEN roa > 0 THEN 1 ELSE 0 END +
        CASE WHEN cfo > 0 THEN 1 ELSE 0 END +
        CASE WHEN roa > roa_prev THEN 1 ELSE 0 END +
        CASE WHEN cfo > ni THEN 1 ELSE 0 END +
        CASE WHEN ltde < ltde_prev THEN 1 ELSE 0 END +
        CASE WHEN cr > cr_prev THEN 1 ELSE 0 END +
        CASE WHEN shrs <= shrs_prev THEN 1 ELSE 0 END +
        CASE WHEN gpm > gpm_prev THEN 1 ELSE 0 END +
        CASE WHEN at > at_prev THEN 1 ELSE 0 END)::INTEGER;
$$;

-- -----------------------------------------------------------------------------
-- DOUBLE PRECISION overloads of the pml.* PyMC helpers. pml.pml_df columns are
-- DOUBLE PRECISION, so these overloads let the materialized views below call
-- the helpers without explicit ::NUMERIC casts on every column.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION pml.safe_logit(p   DOUBLE PRECISION,
                                          eps DOUBLE PRECISION DEFAULT 1e-6) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT LN(GREATEST(eps, LEAST(1 - eps, p)) / (1 - GREATEST(eps, LEAST(1 - eps, p))));
$$;

CREATE OR REPLACE FUNCTION pml.zscore(val   DOUBLE PRECISION,
                                      mu    DOUBLE PRECISION,
                                      sigma DOUBLE PRECISION) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (val - mu) / NULLIF(sigma, 0);
$$;

CREATE OR REPLACE FUNCTION pml.winsorise(val DOUBLE PRECISION,
                                         lo  DOUBLE PRECISION,
                                         hi  DOUBLE PRECISION) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT GREATEST(lo, LEAST(hi, val));
$$;

CREATE OR REPLACE FUNCTION pml.beat_counts(surprises DOUBLE PRECISION[])
	RETURNS TABLE
	        (
		        n_total INT,
		        n_beats INT
	        )
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT SUM(CASE WHEN s IS NOT NULL THEN 1 ELSE 0 END)::INT           AS n_total,
       SUM(CASE WHEN s IS NOT NULL AND s > 0 THEN 1 ELSE 0 END)::INT AS n_beats
FROM UNNEST(surprises) AS s;
$$;

CREATE OR REPLACE FUNCTION pml.target_drift(arr DOUBLE PRECISION[]) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT AVG(pml.calc_change_ratio(arr[i], arr[i + 1]))
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1);
$$;

-- -----------------------------------------------------------------------------
-- target_drift COVERAGE + min-points GUARD.
--
-- ``target_drift`` averages ``calc_change_ratio(arr[i], arr[i+1])`` over
-- consecutive pairs, and ``calc_change_ratio`` is NULL whenever the predecessor
-- is NULL or 0 (NULLIF). A trail (e.g. ``price_target_*_ago``) where only the
-- current value is populated therefore yields a drift from a SINGLE noisy pair,
-- or NULL when the whole trail is empty. Downstream the fused Kalman panel z-scores
-- + 0-fills those NULLs, producing a near-constant response column whose rank-1
-- ICM loading is unidentified — the ridge that froze the sampler (max R-hat 4.45,
-- min ESS 4.3). These helpers expose the valid-pair COUNT and a min-points-guarded
-- drift so the MV and the model can gate on real data availability rather than the
-- post-fill zero spike.
-- -----------------------------------------------------------------------------

-- Count of VALID consecutive pairs (both endpoints non-null, predecessor <> 0).
CREATE OR REPLACE FUNCTION pml.target_drift_n(arr NUMERIC[]) RETURNS INT
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT COUNT(*)::INT
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1)
  AND arr[i] IS NOT NULL
  AND arr[i + 1] IS NOT NULL
  AND arr[i + 1] <> 0;
$$;

CREATE OR REPLACE FUNCTION pml.target_drift_n(arr DOUBLE PRECISION[]) RETURNS INT
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT COUNT(*)::INT
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1)
  AND arr[i] IS NOT NULL
  AND arr[i + 1] IS NOT NULL
  AND arr[i + 1] <> 0;
$$;

-- Min-points-guarded drift: NULL unless at least ``min_points`` valid pairs exist,
-- so a single noisy pair no longer masquerades as a populated drift signal.
CREATE OR REPLACE FUNCTION pml.target_drift(arr NUMERIC[], min_points INT) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN pml.target_drift_n(arr) >= min_points THEN pml.target_drift(arr) END;
$$;

CREATE OR REPLACE FUNCTION pml.target_drift(arr DOUBLE PRECISION[], min_points INT) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN pml.target_drift_n(arr) >= min_points THEN pml.target_drift(arr) END;
$$;

-- -----------------------------------------------------------------------------
-- SIGN-PRESERVING drift, for series that may be NEGATIVE or cross zero.
--
-- ``target_drift`` divides by the raw predecessor, which is correct only while the
-- series is strictly positive — true of every trail it was written for (prices,
-- analyst targets, coverage counts, realized vol). EPS, net income and FCF are NOT
-- strictly positive, and there the raw denominator INVERTS the signal: a loss
-- narrowing from -2.00 to -1.00 scores (-1 - -2) / -2 = -0.5, i.e. a clear
-- improvement recorded as negative drift. Winsorising caps the magnitude but keeps
-- the wrong sign, and a sign-flipped predictor is worse than an absent one.
--
-- ``signed_drift`` is identical except the denominator is ABS(prev), so the same
-- narrowing scores +0.5. The validity rule is unchanged (both endpoints non-null,
-- predecessor <> 0), so ``pml.target_drift_n`` is the counter for BOTH families —
-- there is deliberately no ``signed_drift_n``.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pml.signed_drift(arr NUMERIC[]) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT AVG((arr[i] - arr[i + 1]) / NULLIF(ABS(arr[i + 1]), 0))
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1);
$$;

CREATE OR REPLACE FUNCTION pml.signed_drift(arr DOUBLE PRECISION[]) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT AVG((arr[i] - arr[i + 1]) / NULLIF(ABS(arr[i + 1]), 0))
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1);
$$;

-- Min-points-guarded sign-preserving drift (mirrors pml.target_drift(arr, n)).
CREATE OR REPLACE FUNCTION pml.signed_drift(arr NUMERIC[], min_points INT) RETURNS NUMERIC
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN pml.target_drift_n(arr) >= min_points THEN pml.signed_drift(arr) END;
$$;

CREATE OR REPLACE FUNCTION pml.signed_drift(arr DOUBLE PRECISION[], min_points INT) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN pml.target_drift_n(arr) >= min_points THEN pml.signed_drift(arr) END;
$$;

CREATE OR REPLACE FUNCTION pml.coef_var(mu    DOUBLE PRECISION,
                                        sigma DOUBLE PRECISION) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT sigma / NULLIF(ABS(mu), 0);
$$;

CREATE OR REPLACE FUNCTION pml.fcf_dividend_coverage(fcf            DOUBLE PRECISION,
                                                     dividends_paid DOUBLE PRECISION) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT pml.safe_divide(fcf, ABS(dividends_paid));
$$;

CREATE OR REPLACE FUNCTION pml.altman_zone(z DOUBLE PRECISION) RETURNS INT
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN z IS NULL THEN NULL WHEN z < 1.81 THEN 1 WHEN z < 2.99 THEN 2 ELSE 3 END;
$$;

CREATE OR REPLACE FUNCTION pml.accruals_ratio(ni    DOUBLE PRECISION,
                                              cfo   DOUBLE PRECISION,
                                              scale DOUBLE PRECISION) RETURNS DOUBLE PRECISION
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT pml.safe_divide(ni - cfo, NULLIF(scale, 0));
$$;

CREATE OR REPLACE FUNCTION pml.piotroski_f_score(roa       DOUBLE PRECISION,
                                                 roa_prev  DOUBLE PRECISION,
                                                 cfo       DOUBLE PRECISION,
                                                 ni        DOUBLE PRECISION,
                                                 ltde      DOUBLE PRECISION,
                                                 ltde_prev DOUBLE PRECISION,
                                                 cr        DOUBLE PRECISION,
                                                 cr_prev   DOUBLE PRECISION,
                                                 shrs      DOUBLE PRECISION,
                                                 shrs_prev DOUBLE PRECISION,
                                                 gpm       DOUBLE PRECISION,
                                                 gpm_prev  DOUBLE PRECISION,
                                                 at        DOUBLE PRECISION,
                                                 at_prev   DOUBLE PRECISION) RETURNS INTEGER
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (CASE WHEN roa > 0 THEN 1 ELSE 0 END +
        CASE WHEN cfo > 0 THEN 1 ELSE 0 END +
        CASE WHEN roa > roa_prev THEN 1 ELSE 0 END +
        CASE WHEN cfo > ni THEN 1 ELSE 0 END +
        CASE WHEN ltde < ltde_prev THEN 1 ELSE 0 END +
        CASE WHEN cr > cr_prev THEN 1 ELSE 0 END +
        CASE WHEN shrs <= shrs_prev THEN 1 ELSE 0 END +
        CASE WHEN gpm > gpm_prev THEN 1 ELSE 0 END +
        CASE WHEN at > at_prev THEN 1 ELSE 0 END)::INTEGER;
$$;

-- piotroski_f_score LTM screener: thin set-returning wrapper over the scalar
-- pml.piotroski_f_score composite above (defined after both overloads so the
-- body validates on a fresh top-to-bottom run).
-- LTM-era signal wiring: ROA/CFO/leverage/liquidity/margin compare LTM vs
-- neg1fy, share count compares shrs_out vs shrs_out_neg1fy, and asset turnover
-- uses the FQ-vs-FY proxy (no LTM asset-turnover column exists).
CREATE OR REPLACE FUNCTION calc_piotroski_f_score(p_isin text DEFAULT NULL::text)
	RETURNS TABLE
	        (
		        isin              text,
		        piotroski_f_score integer
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT isin AS isin,
       pml.piotroski_f_score(return_on_assets_roa_pct_ltm, return_on_assets_roa_pct_neg1fy,
                             cfo_ltm, net_income_ltm,
                             long_term_debt_equity_ltm, long_term_debt_equity_neg1fy,
                             current_ratio_ltm, current_ratio_neg1fy,
                             shrs_out, shrs_out_neg1fy,
                             gross_profit_margin_pct_ltm, gross_profit_margin_pct_neg1fy,
                             asset_turnover_fq, asset_turnover_fy) AS piotroski_f_score
FROM pml.pml_df pd
WHERE p_isin IS NULL
   OR isin = p_isin;
$$;

ALTER FUNCTION calc_piotroski_f_score(text) OWNER TO postgres;

-- =============================================================================
-- PER-MODEL MATERIALIZED VIEWS (one per pymc model_target)
-- =============================================================================
-- Each MV pre-computes the PyMC-ready feature matrix for one model. Schema:
--   - isin / ticker / sector / industry  -> coords + indices
--   - observed_*                         -> pymc_role='observed'
--   - feat_*                             -> pymc_role='mutable_predictor'
--   - n_*                                -> pymc_role='constant_data'
-- Refresh via:  REFRESH MATERIALIZED VIEW CONCURRENTLY pml.mv_pymc_<model>;
-- =============================================================================

-- ---- 1. EarningsBeatBayesian -------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS pml.mv_pymc_earnings_beat AS
WITH beats AS (SELECT isin,
                      ticker,
                      trading_region,
                      region,
                      country,
                      trading_country,
                      exchange,
                      unit,
                      style_class,
                      size_class,
                      sector,
                      industry,
                      -- ---- EPS surprise trails (quarterly: neg0fq..neg4fq, annual: neg0fy..neg4fy) ----
                      ARRAY [eps_neg0fqsurprise_pct, eps_neg1fqsurprise_pct, eps_neg2fqsurprise_pct, eps_neg3fqsurprise_pct, eps_neg4fqsurprise_pct]                                           AS eps_surprises_q,
                      ARRAY [eps_neg0fysurprise_pct, eps_neg1fysurprise_pct, eps_neg2fysurprise_pct, eps_neg3fysurprise_pct, eps_neg4fysurprise_pct]                                           AS eps_surprises_y,
                      -- ---- EBIT surprise trails (quarterly: neg0fq..neg4fq, annual: neg0fy..neg5fy) ----
                      ARRAY [ebit_neg0fqsurprise_pct, ebit_neg1fqsurprise_pct, ebit_neg2fqsurprise_pct, ebit_neg3fqsurprise_pct, ebit_neg4fqsurprise_pct]                                      AS ebit_surprises_q,
                      ARRAY [ebit_neg0fysurprise_pct, ebit_neg1fysurprise_pct, ebit_neg2fysurprise_pct, ebit_neg3fysurprise_pct, ebit_neg4fysurprise_pct, ebit_neg5fysurprise_pct]             AS ebit_surprises_y,
                      -- ---- EBITDA surprise trails (quarterly: neg0fq..neg4fq, annual: neg0fy..neg5fy) ----
                      ARRAY [ebitda_neg0fqsurprise_pct, ebitda_neg1fqsurprise_pct, ebitda_neg2fqsurprise_pct, ebitda_neg3fqsurprise_pct, ebitda_neg4fqsurprise_pct]                            AS ebitda_surprises_q,
                      ARRAY [ebitda_neg0fysurprise_pct, ebitda_neg1fysurprise_pct, ebitda_neg2fysurprise_pct, ebitda_neg3fysurprise_pct, ebitda_neg4fysurprise_pct, ebitda_neg5fysurprise_pct] AS ebitda_surprises_y,
                      -- ---- Sales surprise trails (quarterly: neg0fq..neg4fq, annual: neg0fy..neg5fy) ----
                      ARRAY [sales_neg0fqsurprise_pct, sales_neg1fqsurprise_pct, sales_neg2fqsurprise_pct, sales_neg3fqsurprise_pct, sales_neg4fqsurprise_pct]                                 AS sales_surprises_q,
                      ARRAY [sales_neg0fysurprise_pct, sales_neg1fysurprise_pct, sales_neg2fysurprise_pct, sales_neg3fysurprise_pct, sales_neg4fysurprise_pct, sales_neg5fysurprise_pct]       AS sales_surprises_y,
                      eps_norm_est_avg_fy1e,
                      eps_norm_est_avg_fq1e,
                      eps_norm_est_num_fy1e,
                      eps_est_avg_rev_pct_fy1e_1w,
                      eps_est_avg_rev_pct_fy1e_1m,
                      eps_est_avg_rev_pct_fy1e_3m,
                      eps_est_avg_rev_pct_fy1e_6m,
                      eps_est_avg_rev_pct_fy1e_1y,
                      eps_gaap_est_avg_rev_pct_fy1e_1m,
                      eps_gaap_est_avg_rev_pct_fy1e_3m,
                      eps_gaap_est_avg_rev_pct_fy1e_6m,
                      -- ---- Most-recent single-period surprises (ready-to-use predictors) ----
                      eps_neg0fqsurprise_pct,
                      eps_neg0fysurprise_pct,
                      ebit_neg0fqsurprise_pct,
                      ebit_neg0fysurprise_pct,
                      ebitda_neg0fqsurprise_pct,
                      ebitda_neg0fysurprise_pct,
                      sales_neg0fqsurprise_pct,
                      sales_neg0fysurprise_pct,
                      days_to_earnings,
                      earnings_report_recency,
                      next_earnings_status,
                      -- ---- Market-cap / EV size & trend raw carriers ----
                      market_cap,
                      market_cap_neg1fy,
                      market_cap_3yavg,
                      enterprise_value,
                      enterprise_value_3yavg
               FROM pml.pml_df
              )
SELECT b.isin,
       b.ticker,
       b.trading_region,
       b.region,
       b.country,
       b.trading_country,
       b.exchange,
       b.unit,
       b.style_class,
       b.size_class,
       b.sector,
       b.industry,
       -- ---- EPS beat counts + logit beat rates (quarterly + annual) ----
       bc_q.n_total                                                                       AS n_total,
       bc_q.n_beats                                                                       AS n_beats,
       bc_y.n_total                                                                       AS n_total_annual,
       bc_y.n_beats                                                                       AS n_beats_annual,
       pml.safe_logit(pml.safe_divide(bc_q.n_beats::NUMERIC, bc_q.n_total))               AS feat_logit_beat_rate,
       pml.safe_logit(pml.safe_divide(bc_y.n_beats::NUMERIC, bc_y.n_total))               AS feat_logit_beat_rate_annual,
       -- ---- EBIT beat counts + logit beat rates (quarterly + annual) ----
       bc_ebit_q.n_total                                                                  AS n_ebit_total,
       bc_ebit_q.n_beats                                                                  AS n_ebit_beats,
       bc_ebit_y.n_total                                                                  AS n_ebit_total_annual,
       bc_ebit_y.n_beats                                                                  AS n_ebit_beats_annual,
       pml.safe_logit(pml.safe_divide(bc_ebit_q.n_beats::NUMERIC, bc_ebit_q.n_total))     AS feat_ebit_logit_beat_rate,
       pml.safe_logit(pml.safe_divide(bc_ebit_y.n_beats::NUMERIC, bc_ebit_y.n_total))     AS feat_ebit_logit_beat_rate_annual,
       -- ---- EBITDA beat counts + logit beat rates (quarterly + annual) ----
       bc_ebitda_q.n_total                                                                AS n_ebitda_total,
       bc_ebitda_q.n_beats                                                                AS n_ebitda_beats,
       bc_ebitda_y.n_total                                                                AS n_ebitda_total_annual,
       bc_ebitda_y.n_beats                                                                AS n_ebitda_beats_annual,
       pml.safe_logit(pml.safe_divide(bc_ebitda_q.n_beats::NUMERIC, bc_ebitda_q.n_total)) AS feat_ebitda_logit_beat_rate,
       pml.safe_logit(pml.safe_divide(bc_ebitda_y.n_beats::NUMERIC, bc_ebitda_y.n_total)) AS feat_ebitda_logit_beat_rate_annual,
       -- ---- Sales beat counts + logit beat rates (quarterly + annual) ----
       bc_sales_q.n_total                                                                 AS n_sales_total,
       bc_sales_q.n_beats                                                                 AS n_sales_beats,
       bc_sales_y.n_total                                                                 AS n_sales_total_annual,
       bc_sales_y.n_beats                                                                 AS n_sales_beats_annual,
       pml.safe_logit(pml.safe_divide(bc_sales_q.n_beats::NUMERIC, bc_sales_q.n_total))   AS feat_sales_logit_beat_rate,
       pml.safe_logit(pml.safe_divide(bc_sales_y.n_beats::NUMERIC, bc_sales_y.n_total))   AS feat_sales_logit_beat_rate_annual,
       b.eps_norm_est_avg_fy1e                                                            AS feat_eps_fy1e,
       b.eps_norm_est_avg_fq1e                                                            AS feat_eps_fq1e,
       b.eps_norm_est_num_fy1e                                                            AS n_eps_estimates,
       b.eps_est_avg_rev_pct_fy1e_1w                                                      AS feat_rev_1w,
       b.eps_est_avg_rev_pct_fy1e_1m                                                      AS feat_rev_1m,
       b.eps_est_avg_rev_pct_fy1e_3m                                                      AS feat_rev_3m,
       b.eps_est_avg_rev_pct_fy1e_6m                                                      AS feat_rev_6m,
       b.eps_est_avg_rev_pct_fy1e_1y                                                      AS feat_rev_1y,
       -- Acceleration of revisions (short vs long horizon) – picks up momentum shifts
       (b.eps_est_avg_rev_pct_fy1e_1m - b.eps_est_avg_rev_pct_fy1e_6m)                    AS feat_rev_accel_1m_6m,
       -- GAAP-vs-normalised revision divergence (quality-of-earnings signal)
       (b.eps_est_avg_rev_pct_fy1e_3m - b.eps_gaap_est_avg_rev_pct_fy1e_3m)               AS feat_rev_gaap_gap_3m,
       -- ---- Most-recent single-period surprises (per metric) ----
       b.eps_neg0fqsurprise_pct                                                           AS feat_last_q_surprise,
       b.eps_neg0fysurprise_pct                                                           AS feat_last_y_surprise,
       b.ebit_neg0fqsurprise_pct                                                          AS feat_ebit_last_q_surprise,
       b.ebit_neg0fysurprise_pct                                                          AS feat_ebit_last_y_surprise,
       b.ebitda_neg0fqsurprise_pct                                                        AS feat_ebitda_last_q_surprise,
       b.ebitda_neg0fysurprise_pct                                                        AS feat_ebitda_last_y_surprise,
       b.sales_neg0fqsurprise_pct                                                         AS feat_sales_last_q_surprise,
       b.sales_neg0fysurprise_pct                                                         AS feat_sales_last_y_surprise,
       b.days_to_earnings                                                                 AS feat_days_to_earnings,
       b.earnings_report_recency                                                          AS feat_report_recency,
       b.next_earnings_status                                                             AS feat_next_earnings_status,
       -- ---- Cross-cutting market-cap / EV size & trend feats ----
       pml.calc_change_ratio(b.market_cap, b.market_cap_neg1fy)                           AS feat_mcap_trend_1y,
       pml.safe_divide(b.market_cap, b.market_cap_3yavg)                                  AS feat_mcap_vs_3yavg,
       pml.safe_divide(b.enterprise_value, b.enterprise_value_3yavg)                      AS feat_ev_vs_3yavg
FROM beats                                                    b,
     LATERAL pml.beat_counts(b.eps_surprises_q::NUMERIC[])    bc_q,
     LATERAL pml.beat_counts(b.eps_surprises_y::NUMERIC[])    bc_y,
     LATERAL pml.beat_counts(b.ebit_surprises_q::NUMERIC[])   bc_ebit_q,
     LATERAL pml.beat_counts(b.ebit_surprises_y::NUMERIC[])   bc_ebit_y,
     LATERAL pml.beat_counts(b.ebitda_surprises_q::NUMERIC[]) bc_ebitda_q,
     LATERAL pml.beat_counts(b.ebitda_surprises_y::NUMERIC[]) bc_ebitda_y,
     LATERAL pml.beat_counts(b.sales_surprises_q::NUMERIC[])  bc_sales_q,
     LATERAL pml.beat_counts(b.sales_surprises_y::NUMERIC[])  bc_sales_y;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_pymc_earnings_beat_isin ON pml.mv_pymc_earnings_beat (isin);

-- ---- 2. PriceTargetAchievement -----------------------------------------------
-- Date columns (income_statement_report_date, next_earnings, fy_end_date,
-- next_income_statement_report_date, next_fy_end_date, expected_report_date)
-- are emitted alongside the cross-sectional features so the notebook can
-- assemble a (time, isin) panel for the MvGaussianRandomWalk time-series
-- formulation of PriceTargetAchievement.
CREATE MATERIALIZED VIEW IF NOT EXISTS pml.mv_pymc_price_target AS
SELECT isin,
       ticker,
       trading_region,
       country,
       trading_country,
       exchange,
       unit,
       style_class,
       size_class,
       sector,
       industry,
       -- ---- Fiscal-calendar anchors for the MvGRW time axis ----
       income_statement_report_date,
       next_earnings,
       -- Low-cardinality categorical fiscal-calendar coords (encode upstream in PyMC)
       next_earnings_when,
       next_earnings_status,
       fy_end_date,
       next_income_statement_report_date,
       next_fy_end_date,
       expected_report_date,
       -- Pre-computed numeric horizons (days). Safe to feed to pm.Data as
       -- the time-step deltas used by MvGaussianRandomWalk innovations.
       (next_earnings - CURRENT_DATE)::INT                                                                     AS days_to_next_earnings,
       (CURRENT_DATE - income_statement_report_date)::INT                                                      AS days_since_last_report,
       (next_fy_end_date - CURRENT_DATE)::INT                                                                  AS days_to_next_fy_end,
       (next_income_statement_report_date - CURRENT_DATE)::INT                                                 AS days_to_next_report,
       (expected_report_date - CURRENT_DATE)::INT                                                              AS days_to_expected_report,
       (fy_end_date - CURRENT_DATE)::INT                                                                       AS days_since_fy_end,
       target_pct_avg                                                                                          AS observed_target_pct,
       target_pct_med                                                                                          AS observed_target_pct_med,
       price_target,
       price_target_median,
       price_target_low,
       price_target_high,
       price_target_stddev,
       last_price,
       total_return_ytd,
       price_target_num                                                                                        AS n_analysts,
       num_strong_buys_ratings + num_buys_ratings - num_sell_ratings -
       num_strong_sell_ratings                                                                                 AS feat_net_buy_sentiment,
       -- Conviction = strong opinions / total opinions; complements net sentiment
       pml.safe_divide((num_strong_buys_ratings + num_strong_sell_ratings)::NUMERIC, NULLIF(
		       num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_sell_ratings +
		       num_strong_sell_ratings + num_no_opinion_ratings,
		       0)::NUMERIC)                                                                                    AS feat_conviction_ratio,
       num_hold_ratings                                                                                        AS feat_holds,
       num_strong_buys_ratings + num_buys_ratings                                                              AS feat_buys,
       num_strong_sell_ratings + num_sell_ratings                                                              AS feat_sells,
       num_no_opinion_ratings                                                                                  AS feat_no_opinion,
       -- ---- Normalised analyst-sentiment shares (coverage-invariant %s) ----
       -- All four normalise by the SAME total-opinions denominator (every
       -- num_*_ratings bucket, incl. no-opinion) so bullish + bearish + neutral
       -- sum to ~100 when no-opinion = 0. pml.safe_divide NULLIFs the
       -- denominator, so zero-coverage rows yield NULL rather than a divide-by-0.
       -- Bullish sentiment: strong-buy + buy share of all opinions (%)
       pml.safe_divide((num_strong_buys_ratings + num_buys_ratings)::NUMERIC,
                       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
                        num_sell_ratings +
                        num_strong_sell_ratings)::NUMERIC)                                                     AS feat_analyst_bullish_pct,
       -- Bearish sentiment: sell + strong-sell share of all opinions (%)
       pml.safe_divide((num_sell_ratings + num_strong_sell_ratings)::NUMERIC,
                       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
                        num_sell_ratings +
                        num_strong_sell_ratings)::NUMERIC)                                                     AS feat_analyst_bearish_pct,
       -- Neutral sentiment: hold share of all opinions (%)
       pml.safe_divide(num_hold_ratings::NUMERIC,
                       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
                        num_sell_ratings +
                        num_strong_sell_ratings)::NUMERIC)                                                     AS feat_analyst_neutral_pct,
       -- Conviction: absolute net directional-consensus magnitude (%)
       ABS(pml.safe_divide(
		       ((num_strong_buys_ratings + num_buys_ratings) - (num_sell_ratings + num_strong_sell_ratings))::NUMERIC,
		       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
		        num_sell_ratings +
		        num_strong_sell_ratings)::NUMERIC))                                                            AS feat_analyst_conviction,
       pml.calc_change_ratio(price_target::NUMERIC, last_price::NUMERIC)                                       AS feat_implied_upside,
       -- Asymmetry of the target distribution (skew of analyst optimism)
       pml.calc_change_ratio(target_pct_high::NUMERIC,
                             target_pct_low::NUMERIC)                                                          AS feat_target_range_width,
       pml.calc_change_ratio(price_target::NUMERIC,
                             price_target_3m_ago::NUMERIC)                                                     AS feat_pt_momentum_3m,
       pml.calc_change_ratio(price_target_num::NUMERIC,
                             price_target_num_3m_ago::NUMERIC)                                                 AS feat_coverage_change_3m,
       pml.coef_var(price_target::NUMERIC, price_target_stddev::NUMERIC)                                       AS feat_target_dispersion_cv,
       -- Position within 52-week range (0..1) – Bayesian prior on mean-reversion
       pml.safe_divide(last_price - w_52low_adj,
                       NULLIF(w_52high_adj - w_52low_adj, 0))                                                  AS feat_52w_range_position,
       p_e_ntm                                                                                                 AS feat_pe_ntm,
       ev_ebitda_ntm                                                                                           AS feat_ev_ebitda_ntm,
       volatility_3m                                                                                           AS feat_vol_3m,
       analyst_rating                                                                                          AS feat_analyst_rating,
       -- ---- Price-target achievement / accuracy (realised vs 1Y-ago targets) ----
       -- Capped achievement: 1.0 once price meets/exceeds the 1Y-ago target,
       -- else the fraction of that target the price has reached.
       CASE
	       WHEN price_target_1y_ago > 0 AND last_price >= price_target_1y_ago THEN 1.0
	       WHEN price_target_1y_ago > 0
		       THEN pml.safe_divide(last_price, price_target_1y_ago) END                                       AS feat_pt_achievement_1y,
       -- Absolute relative error of the 1Y-ago target vs realised price.
       pml.safe_divide(ABS(last_price - price_target_1y_ago),
                       ABS(price_target_1y_ago))                                                               AS feat_pt_accuracy_1y,
       -- Signed optimism bias: positive => 1Y-ago target overshot realised price.
       pml.safe_divide(price_target_1y_ago - last_price,
                       ABS(price_target_1y_ago))                                                               AS feat_pt_optimism_bias,
       -- Did the realised price land inside the 1Y-ago low/high band?
       CASE
	       WHEN last_price BETWEEN price_target_low_1y_ago AND price_target_high_1y_ago THEN 1.0
	       ELSE 0.0 END                                                                                        AS feat_pt_range_hit_rate,
       -- Mean-vs-median target spread (skew of the current target distribution).
       pml.safe_divide(price_target - price_target_median,
                       price_target_median)                                                                    AS feat_pt_median_vs_mean_spread,
       -- Change in normalised high-low range vs 1Y ago (>0 = analysts diverging,
       -- <0 = converging / firming conviction).
       pml.safe_divide(price_target_high - price_target_low, price_target_median) -
       pml.safe_divide(price_target_high_1y_ago - price_target_low_1y_ago,
                       price_target_median_1y_ago)                                                             AS feat_pt_high_low_convergence_1y,
       -- Current coverage vs its trailing 3m/6m/1y average (>1 = coverage rising).
       pml.safe_divide(price_target_num, (price_target_num_1y_ago + price_target_num_6m_ago + price_target_num_3m_ago) /
                                         3.0)                                                                  AS feat_analyst_count_stability,
       -- ---- Cross-cutting market-cap / EV size & trend feats ----
       pml.calc_change_ratio(market_cap, market_cap_neg1fy)                                                    AS feat_mcap_trend_1y,
       pml.safe_divide(market_cap, market_cap_3yavg)                                                           AS feat_mcap_vs_3yavg,
       pml.safe_divide(enterprise_value, enterprise_value_3yavg)                                               AS feat_ev_vs_3yavg
FROM pml.pml_df;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_pymc_price_target_isin ON pml.mv_pymc_price_target (isin);

-- ---- 3. KalmanFilterPriceTarget ----------------------------------------------
-- The single-security GaussianRandomWalk filter reconstructs its time axis from
-- the embedded *_ago cohort. The fiscal-calendar date anchors + day-count
-- horizons below mirror mv_pymc_price_target so the notebook can derive the
-- *real* (irregular) elapsed-time spacing that the marginalized GRW uses to
-- scale its process-variance covariance kernel (min(tau_s, tau_t)) — see
-- KalmanFilterModel._build_marginalized_likelihood / _resolve_time_deltas.
-- target_drift is now computed for every price_* / price_target_* trail (mean,
-- high, low, median, raw price, analyst-count and dispersion) so the per-ISIN
-- snapshot exposes the full drift/state-transition signal set.
--
-- FUSED MvGRW PANEL: the cross-sectional fused model
-- (build_fused_kalman_pt_model) consumes these same MV columns in fused
-- state-space roles — feat_avg_beta (the NULL-aware mean of beta_{1y,2y,5y}) as
-- the SYSTEMATIC RISK (CAPM beta) driver that conditions
-- risk_adj_return = expected_return * exp(-risk_penalty * z(feat_avg_beta)),
-- feat_pt_noise_sigma as the cv that widens
-- sigma_isin = sigma_base * (1 + cv) / sqrt(n_analysts), and n_analysts as the
-- precision count. feat_vol_drift (drift across the realized-vol term structure
-- 1m -> 1y, mirroring feat_pt_noise_drift) is the observation-noise-drift
-- widener; the absolute feat_vol_{1m,3m,6m,1y} levels are no longer emitted.
CREATE MATERIALIZED VIEW IF NOT EXISTS pml.mv_pymc_kalman_pt AS
SELECT isin,
       ticker,
       name,
       trading_region,
       region,
       country,
       country_name,
       trading_country,
       trading_country_name,
       exchange,
       exchange_name,
       unit,
       unit_name,
       style_class,
       size_class,
       sector,
       industry,
       last_updated,
       -- ---- Fiscal-calendar anchors (raw DATE coords for the GRW time axis) ----
       income_statement_report_date,
       next_earnings,
       -- Low-cardinality categorical fiscal-calendar coords (encode upstream in PyMC)
       next_earnings_when,
       next_earnings_status,
       fy_end_date,
       next_fiscal_quarter,
       next_income_statement_report_date,
       next_fy_end_date,
       expected_report_date,
       -- Pre-computed numeric horizons (days). Feed pm.Data as the time-step
       -- deltas the marginalized GaussianRandomWalk scales innovations by.
       --
       -- REPRODUCIBILITY: these seven days_* columns are computed against
       -- CURRENT_DATE, so this MV is NOT reproducible across refresh dates --
       -- refreshing on a different day silently shifts every horizon. That is
       -- acceptable for the live screen (the point is "days from today") but
       -- makes the MV unusable as-is for a point-in-time backtest, which would
       -- need an as-of date parameter instead. It is also part of why the
       -- days_* family is excluded from the drift design matrix in Python via
       -- KALMAN_TIME_COVARIATE_PREFIX ('days_') in KalmanFilterModel.py.
       (next_earnings - CURRENT_DATE)::INT                                                                                                                                                                     AS days_to_next_earnings,
       (CURRENT_DATE - income_statement_report_date)::INT                                                                                                                                                      AS days_since_last_report,
       (next_fy_end_date - CURRENT_DATE)::INT                                                                                                                                                                  AS days_to_next_fy_end,
       (next_fiscal_quarter - CURRENT_DATE)::INT                                                                                                                                                                  AS days_to_next_fiscal_quarter,
       (next_income_statement_report_date - CURRENT_DATE)::INT                                                                                                                                                 AS days_to_next_report,
       (expected_report_date - CURRENT_DATE)::INT                                                                                                                                                              AS days_to_expected_report,
       (fy_end_date - CURRENT_DATE)::INT                                                                                                                                                                       AS days_since_fy_end,
       market_cap,
       enterprise_value,
       -- Raw market-cap percentile ranks (0-100, 100 = largest), carried through
       -- for the §11-§13 candidate cohorts which filter on the raw scale
       -- (min_mcap_country_rank). The ratio-scale feat_mcap_*_r mirrors are below.
       market_cap_global_r,
       market_cap_global_sec_r,
       market_cap_region_r,
       market_cap_region_sec_r,
       market_cap_country_r,
       market_cap_country_sec_r,
       rel_volume                                                                                                                                                                                              AS feat_rel_volume,
       price_target                                                                                                                                                                                            AS observed_pt,
       last_price,
       price_target_median,
       price_target_high,
       price_target_low,
       price_target_num                                                                                                                                                                                        AS n_analysts,
       -- ---- Lagged analyst-target trail (kalman_pt observed state sequence) -----
       -- Full multi-horizon *_ago snapshots the Kalman panel observes as the latent
       -- price-target state sequence (level / low / high / median). pymc_role
       -- 'observed' for kalman_pt in pml.vw_pymc_feature_catalogue; emitted
       -- un-prefixed so feature_alias (== column_name) resolves against
       -- kalman_df.columns in the notebook present-check.
       price_target_1w_ago,
       price_target_mtd_ago,
       price_target_1m_ago,
       price_target_qtd_ago,
       price_target_3m_ago,
       price_target_6m_ago,
       price_target_ytd_ago,
       price_target_1y_ago,
       price_target_low_1w_ago,
       price_target_low_mtd_ago,
       price_target_low_1m_ago,
       price_target_low_qtd_ago,
       price_target_low_3m_ago,
       price_target_low_6m_ago,
       price_target_low_ytd_ago,
       price_target_low_1y_ago,
       price_target_high_1w_ago,
       price_target_high_mtd_ago,
       price_target_high_1m_ago,
       price_target_high_qtd_ago,
       price_target_high_3m_ago,
       price_target_high_6m_ago,
       price_target_high_ytd_ago,
       price_target_high_1y_ago,
       price_target_median_1w_ago,
       price_target_median_mtd_ago,
       price_target_median_1m_ago,
       price_target_median_qtd_ago,
       price_target_median_3m_ago,
       price_target_median_6m_ago,
       price_target_median_ytd_ago,
       price_target_median_1y_ago,
       -- ---- Lagged analyst-count trail (kalman_pt constant_data scale) ----------
       -- Coverage-count *_ago snapshots: fixed per-step analyst participation the
       -- panel conditions on (pymc_role 'constant_data'), incl. the 6m horizon
       -- (which also feeds feat_coverage_drift).
       price_target_num_1w_ago,
       price_target_num_mtd_ago,
       price_target_num_1m_ago,
       price_target_num_qtd_ago,
       price_target_num_3m_ago,
       price_target_num_6m_ago,
       price_target_num_ytd_ago,
       price_target_num_1y_ago,
       -- ---- Lagged spot-price trail (kalman_pt observed state sequence) ---------
       -- Raw historical price snapshots pooled with the analyst-target trail by
       -- KalmanFilterPriceTarget.build_price_target_history (the _AGO_HISTORY_RE
       -- unpivot already matches the bare price_* family), so the GRW filter's
       -- observed fair-value sequence gains the realised price path alongside the
       -- target path. pymc_role 'observed' for kalman_pt in
       -- pml.vw_pymc_feature_catalogue; emitted un-prefixed so feature_alias
       -- (== column_name) resolves against kalman_df.columns in the notebook
       -- present-check. Horizons mirror the feat_price_drift inputs plus the long
       -- 3y/5y anchors and the 1d / period-to-date (mtd/qtd/ytd) snapshots.
       price_1d_ago,
       price_5d_ago,
       price_1w_ago,
       price_mtd_ago,
       price_1m_ago,
       price_3m_ago,
       price_6m_ago,
       price_ytd_ago,
       price_1y_ago,
       price_3y_ago,
       price_5y_ago,
       price_qtd_ago,
       calc_change_ratio(price_target::numeric, last_price::numeric)                                                                                                                                           AS feat_implied_upside,
       -- ---- Analyst rating mix, conviction and 1y achievement / accuracy ---------
       -- Copied verbatim from mv_pymc_price_target so the kalman_pt cross-section
       -- carries the same analyst-sentiment predictors (all mutable_predictor).
       num_hold_ratings                                                                                                                                                                                        AS feat_holds,
       num_strong_buys_ratings + num_buys_ratings                                                                                                                                                              AS feat_buys,
       num_strong_sell_ratings + num_sell_ratings                                                                                                                                                              AS feat_sells,
       num_no_opinion_ratings                                                                                                                                                                                  AS feat_no_opinion,
       pml.safe_divide((num_strong_buys_ratings + num_buys_ratings)::numeric,
                       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
                        num_sell_ratings +
                        num_strong_sell_ratings)::numeric)                                                                                                                                                     AS feat_analyst_bullish_pct,
       pml.safe_divide((num_sell_ratings + num_strong_sell_ratings)::numeric,
                       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
                        num_sell_ratings +
                        num_strong_sell_ratings)::numeric)                                                                                                                                                     AS feat_analyst_bearish_pct,
       pml.safe_divide(num_hold_ratings::numeric,
                       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
                        num_sell_ratings +
                        num_strong_sell_ratings)::numeric)                                                                                                                                                     AS feat_analyst_neutral_pct,
       abs(pml.safe_divide(
		       (num_strong_buys_ratings + num_buys_ratings - (num_sell_ratings + num_strong_sell_ratings))::numeric,
		       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
		        num_sell_ratings +
		        num_strong_sell_ratings)::numeric))                                                                                                                                                            AS feat_analyst_conviction,
       analyst_rating                                                                                                                                                                                          AS feat_analyst_rating,
       CASE
	       WHEN price_target_1y_ago > 0::double precision AND last_price >= price_target_1y_ago
		       THEN 1.0::double precision
	       WHEN price_target_1y_ago > 0::double precision THEN pml.safe_divide(last_price, price_target_1y_ago)
	       ELSE NULL::double precision END                                                                                                                                                                     AS feat_pt_achievement_1y,
       pml.safe_divide(abs(last_price - price_target_1y_ago),
                       abs(price_target_1y_ago))                                                                                                                                                               AS feat_pt_accuracy_1y,
       CASE
	       WHEN last_price >= price_target_low_1y_ago AND last_price <= price_target_high_1y_ago THEN 1.0
	       ELSE 0.0 END                                                                                                                                                                                        AS feat_pt_range_hit_rate,
       -- ---- Per-step drift (mean log-uplift) across every price / target trail ----
       -- Each drift is min-points-guarded (>=2 valid consecutive pairs) so a single
       -- noisy pair can't masquerade as signal, and winsorised to [-1, 1] to bound the
       -- heavy ratio tails. A companion ``*_n`` valid-pair count is emitted so the
       -- fused Kalman panel (and its coverage guard) can gate on real data
       -- availability instead of the post-fill zero spike — see pml.target_drift_n /
       -- the KALMAN_RESPONSE_COVERAGE_MIN guard in prepare_kalman_panel_inputs.
       pml.winsorise(pml.target_drift(
		                     ARRAY [price_target::NUMERIC, price_target_1w_ago::NUMERIC, price_target_1m_ago::NUMERIC, price_target_3m_ago::NUMERIC, price_target_6m_ago::NUMERIC, price_target_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_pt_drift,
       pml.target_drift_n(ARRAY [price_target::NUMERIC, price_target_1w_ago::NUMERIC, price_target_1m_ago::NUMERIC, price_target_3m_ago::NUMERIC, price_target_6m_ago::NUMERIC, price_target_1y_ago::NUMERIC]) AS feat_pt_drift_n,
       pml.winsorise(pml.target_drift(
		                     ARRAY [last_price::NUMERIC, price_1w_ago::NUMERIC, price_1m_ago::NUMERIC, price_3m_ago::NUMERIC, price_6m_ago::NUMERIC, price_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_price_drift,
       pml.target_drift_n(ARRAY [last_price::NUMERIC, price_1w_ago::NUMERIC, price_1m_ago::NUMERIC, price_3m_ago::NUMERIC, price_6m_ago::NUMERIC, price_1y_ago::NUMERIC])                                      AS feat_price_drift_n,
       -- High / low / median analyst-target trails — capture skew in target drift
       pml.winsorise(pml.target_drift(
		                     ARRAY [price_target_high::NUMERIC, price_target_high_1w_ago::NUMERIC, price_target_high_1m_ago::NUMERIC, price_target_high_3m_ago::NUMERIC, price_target_high_6m_ago::NUMERIC, price_target_high_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_pt_high_drift,
       pml.winsorise(pml.target_drift(
		                     ARRAY [price_target_low::NUMERIC, price_target_low_1w_ago::NUMERIC, price_target_low_1m_ago::NUMERIC, price_target_low_3m_ago::NUMERIC, price_target_low_6m_ago::NUMERIC, price_target_low_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_pt_low_drift,
       pml.winsorise(pml.target_drift(
		                     ARRAY [price_target_median::NUMERIC, price_target_median_1w_ago::NUMERIC, price_target_median_1m_ago::NUMERIC, price_target_median_3m_ago::NUMERIC, price_target_median_6m_ago::NUMERIC, price_target_median_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_pt_median_drift,
       -- Analyst-coverage drift: rising / falling participation is a state signal
       pml.winsorise(pml.target_drift(
		                     ARRAY [price_target_num::NUMERIC, price_target_num_1w_ago::NUMERIC, price_target_num_1m_ago::NUMERIC, price_target_num_3m_ago::NUMERIC, price_target_num_6m_ago::NUMERIC, price_target_num_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_coverage_drift,
       -- Drift in analyst-stddev tells how noise itself is evolving (state-space Q)
       pml.winsorise(pml.target_drift(
		                     ARRAY [price_target_stddev::NUMERIC, price_target_stddev_1w_ago::NUMERIC, price_target_stddev_1m_ago::NUMERIC, price_target_stddev_3m_ago::NUMERIC, price_target_stddev_6m_ago::NUMERIC, price_target_stddev_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_pt_noise_drift,
       price_target_stddev                                                                                                                                                                                     AS feat_pt_noise_sigma,
       -- ---- Lagged analyst-stddev trail (kalman_pt observed noise sequence) -----
       -- Consensus-dispersion *_ago snapshots the panel observes as the evolving
       -- measurement-noise level (pymc_role 'observed'); emitted un-prefixed so
       -- feature_alias (== column_name) resolves in the notebook present-check.
       price_target_stddev_1w_ago,
       price_target_stddev_mtd_ago,
       price_target_stddev_1m_ago,
       price_target_stddev_qtd_ago,
       price_target_stddev_3m_ago,
       price_target_stddev_6m_ago,
       price_target_stddev_ytd_ago,
       price_target_stddev_1y_ago,
       -- Inter-analyst range (high - low) normalised by mean target
       pml.safe_divide(price_target_high - price_target_low,
                       NULLIF(price_target, 0))                                                                                                                                                                AS feat_pt_range_norm,
       -- Short-term momentum: last day's price change (mutable_predictor).
       one_day_pct                                                                                                                                                                                             AS feat_one_day_return,
       price_chg_pct_3m                                                                                                                                                                                             AS feat_price_chg_pct_3m,
       -- Drift across the realized-vol term structure (1m -> 1y) tells how price
       -- volatility itself is evolving (the sigma_obs widener analogue of
       -- feat_pt_noise_drift's state-space Q), with the matching valid-pair count.
       pml.winsorise(pml.target_drift(
		                     ARRAY [volatility_1m::NUMERIC, volatility_3m::NUMERIC, volatility_6m::NUMERIC, volatility_1y::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_vol_drift,
       pml.target_drift_n(ARRAY [volatility_1m::NUMERIC, volatility_3m::NUMERIC, volatility_6m::NUMERIC, volatility_1y::NUMERIC])                                                                              AS feat_vol_drift_n,
       -- ---------------------------------------------------------------------
       -- OBSERVATION-SCALE drivers (they belong in sigma_isin, NOT in the drift
       -- design matrix -- see KALMAN_DRIFT_EXCLUDED_FEATURES).
       --
       -- Measured 2026-08-16 on this MV: correlation of each candidate with
       -- log|residual| after an OLS fit of the 17 drift features (n = 6,533).
       --
       --     cv = pt_stddev/price   +0.2245   already in sigma_isin
       --     log market_cap         -0.2100   <- feat_log_mcap below
       --     volatility_1m          +0.1924   <- feat_vol_level below
       --     volatility_1y          +0.1897   <- feat_vol_level below
       --     log n_analysts         -0.1696   already in sigma_isin
       --     feat_vol_drift         -0.0349   <- the DRIFT is nearly useless
       --
       -- That last line is the point. 0.9.9.6 replaced the absolute
       -- feat_vol_{1m,3m,6m,1y} LEVELS with feat_vol_drift on the reasoning that
       -- the drift is "the sigma_obs widener analogue of feat_pt_noise_drift".
       -- It traded a +0.19 driver for a -0.03 one: how fast volatility is
       -- CHANGING says almost nothing about how dispersed a name's implied
       -- upside is, whereas how volatile it IS says a great deal. The levels are
       -- restored here as ONE composite rather than four columns because they are
       -- 0.53-0.94 correlated with each other (1m~3m 0.88, 3m~6m 0.94).
       --
       -- The median (not the mean) is used so a single missing or spiking window
       -- cannot move the composite; pml.winsorise bounds the residual tail.
       pml.winsorise((SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY v)
                      FROM unnest(ARRAY [volatility_1m, volatility_3m,
                                         volatility_6m, volatility_1y]) AS t(v)
                      WHERE v IS NOT NULL)::NUMERIC, 0, 300)::DOUBLE PRECISION                                                                                                                                   AS feat_vol_level,
       -- Size. Logged because market_cap spans ~7 orders of magnitude, so the
       -- raw level would be a near-binary indicator after z-scoring. Small names
       -- carry materially wider analyst dispersion; this is the second strongest
       -- driver available and was not previously reachable by the model at all
       -- (market_cap carries no kalman_pt tag in pml_df_metadata).
       ln(NULLIF(GREATEST(market_cap, 0::double precision), 0))                                                                                                                                                  AS feat_log_mcap,
       -- Raw beta windows (systematic-risk inputs to feat_avg_beta below).
       beta_1y,
       beta_2y,
       beta_5y,
       -- feat_avg_beta: NULL-aware mean of the available beta windows. The fused
       -- panel keys its risk adjustment on this systematic-risk (CAPM) driver,
       -- risk_adj_return = expected_return * exp(-risk_penalty * z(feat_avg_beta));
       -- realized vol enters only via the feat_vol_drift sigma_obs widener above.
       ((COALESCE(beta_1y, 0::double precision) + COALESCE(beta_2y, 0::double precision) +
         COALESCE(beta_5y, 0::double precision)) /
        NULLIF((beta_1y IS NOT NULL)::int + (beta_2y IS NOT NULL)::int + (beta_5y IS NOT NULL)::int,
               0))                                                                                                                                                                                             AS feat_avg_beta,
       total_return_ytd                                                                                                                                                                                        AS feat_total_return_ytd,
       total_return_5y                                                                                                                                                                                         AS feat_total_return_5y,
       total_return_10y                                                                                                                                                                                        AS feat_total_return_10y,
       tot_return_pct_cagr_3y                                                                                                                                                                                  AS feat_tr_cagr_3y,
       tot_return_pct_cagr_10y                                                                                                                                                                                 AS feat_tr_cagr_10y,
       tot_return_pct_cagr_5y                                                                                                                                                                                  AS feat_tr_cagr_5y,
       tot_return_pct_cagr_1y                                                                                                                                                                                  AS feat_tr_cagr_1y,
       total_return_1d                                                                                                                                                                                         AS feat_total_return_1d,
       total_return_5d                                                                                                                                                                                         AS feat_total_return_5d,
       total_return_1w                                                                                                                                                                                         AS feat_total_return_1w,
       total_return_1m                                                                                                                                                                                         AS feat_total_return_1m,
       total_return_3m                                                                                                                                                                                         AS feat_total_return_3m,
       total_return_6m                                                                                                                                                                                         AS feat_total_return_6m,
       total_return_1y                                                                                                                                                                                         AS feat_total_return_1y,
       total_return_3y                                                                                                                                                                                         AS feat_total_return_3y,
       total_return_mtd                                                                                                                                                                                        AS feat_total_return_mtd,
       total_return_qtd                                                                                                                                                                                        AS feat_total_return_qtd,
       total_return_2025                                                                                                                                                                                       AS feat_total_return_2025,
       total_return_2024                                                                                                                                                                                       AS feat_total_return_2024,
       total_return_2023                                                                                                                                                                                       AS feat_total_return_2023,
       total_return_2022                                                                                                                                                                                       AS feat_total_return_2022,
       total_return_2021                                                                                                                                                                                       AS feat_total_return_2021,
       -- ---- Market-cap rank ratios: (100 - rank) / 100, ~0 = LARGEST ----
       -- Six nested scopes (global / region / country, each also sector-relative).
       -- Easy to invert by mistake: the vendor rank is 100 for the biggest name, so
       -- the ratio runs the OTHER way — 0.0 = largest, 1.0 = smallest.
       --
       -- feat_mcap_country_r is the size discount driver for the fused panel's
       -- additive size tilt (NOT a drift predictor — KALMAN_TILT_FEATURE_ORDER
       -- routes it to its own pm.Data container). The five siblings are 0.78-0.98
       -- correlated with it and with each other, so they are deliberately NOT tilt
       -- drivers and NOT drift predictors: KALMAN_SIZE_RANK_SIBLING_FEATURES bars
       -- them from the drift design matrix. They are emitted for EDA, the screen
       -- and the analytics export, which is why they keep a catalogue row rather
       -- than a pymc_role='excluded' flip (that would raise MISSING_FROM_CATALOGUE
       -- while the MV still emits them).
       pml.safe_divide(100-market_cap_global_r, 100)                                                                                                                                                            AS feat_mcap_global_r,
       pml.safe_divide(100-market_cap_global_sec_r, 100)                                                                                                                                                        AS feat_mcap_global_sec_r,
       pml.safe_divide(100-market_cap_region_r, 100)                                                                                                                                                            AS feat_mcap_region_r,
       pml.safe_divide(100-market_cap_region_sec_r, 100)                                                                                                                                                        AS feat_mcap_region_sec_r,
       pml.safe_divide(100-market_cap_country_r, 100)                                                                                                                                                           AS feat_mcap_country_r,
       pml.safe_divide(100-market_cap_country_sec_r, 100)                                                                                                                                                       AS feat_mcap_country_sec_r,
       -- ---- EPS trend / surprise / beat frequency ----
       -- Replaces the market-cap & EV trend/drift feats (feat_mcap_trend_1y,
       -- feat_mcap_vs_3yavg, feat_ev_vs_3yavg, feat_mv_ev_drift + the market_cap_ev*
       -- trail they were built from). Those were PRICE-derived — market_cap is
       -- last_price * shrs_out — so they restated price history the drift matrix
       -- already carries via feat_price_drift / feat_price_chg_pct_3m /
       -- feat_one_day_return / feat_total_return_*. The EPS family below is
       -- earnings-derived and orthogonal to the price trails: it measures what
       -- analysts are actually revising their targets ABOUT. The trio remains in the
       -- other six mv_pymc_* views; only kalman_pt drops it.
       --
       -- feat_net_eps_drift: per-step drift of the net basic EPS fiscal-year trail
       -- (fy -> neg5fy), min-points-guarded (>=2 valid pairs) and winsorised to
       -- [-1, 1] like the analyst-target drifts. Uses pml.signed_drift, NOT
       -- pml.target_drift: EPS crosses zero, and the raw denominator would score a
       -- loss narrowing from -2.00 to -1.00 as -0.5 (see the signed_drift header).
       pml.winsorise(pml.signed_drift(
		                     ARRAY [net_eps_basic_fy, net_eps_basic_neg1fy, net_eps_basic_neg2fy, net_eps_basic_neg3fy, net_eps_basic_neg4fy, net_eps_basic_neg5fy],
		                     2), (-1)::double precision,
                     1::double precision)                                                                                                                                                                      AS feat_net_eps_drift,
       pml.target_drift_n(ARRAY [net_eps_basic_fy, net_eps_basic_neg1fy, net_eps_basic_neg2fy, net_eps_basic_neg3fy, net_eps_basic_neg4fy, net_eps_basic_neg5fy])                                               AS feat_net_eps_drift_n,
       -- Most recent realised EPS surprise, quarterly and annual. Same aliases
       -- mv_pymc_earnings_beat uses for the same source columns.
       eps_neg0fqsurprise_pct                                                                                                                                                                                  AS feat_last_q_surprise,
       eps_neg0fysurprise_pct                                                                                                                                                                                  AS feat_last_y_surprise,
       -- Beat FREQUENCY over the surprise trail (5 quarterly / 6 annual; there is no
       -- eps_neg5fqsurprise_pct). Deliberately the RAW [0, 1] rate rather than
       -- mv_pymc_earnings_beat's pml.safe_logit form: a 0/5 or 5/5 name pins the
       -- logit at +/-13.8, which becomes a fat tail once the fused model z-scores the
       -- drift design. pml.safe_divide returns NULL when n_total = 0, so a name with
       -- no surprise history is NULL rather than a fabricated 0.0.
       pml.safe_divide(bc_q.n_beats::double precision, bc_q.n_total::double precision)                                                                                                                          AS feat_eps_beat_rate,
       pml.safe_divide(bc_y.n_beats::double precision, bc_y.n_total::double precision)                                                                                                                          AS feat_eps_beat_rate_annual,
       -- ---- Piotroski F-score fundamental-quality trail ----
       -- Four per-fiscal-year 9-signal composites (pml.piotroski_f_score over the
       -- ROA / CFO / leverage / liquidity / share-count / margin / asset-turnover
       -- lag pairs, computed once in the pio LATERAL below) plus their median.
       -- Only feat_median_piotroski_f_score enters the fused drift design matrix
       -- (KALMAN_PIOTROSKI_COMPONENT_FEATURES bars the collinear per-year
       -- components in KalmanFilterModel.select_drift_features); the component
       -- scores are emitted for EDA / analytics.
       pio.feat_piotroski_f_score_fy,
       pio.feat_piotroski_f_score_neg1fy,
       pio.feat_piotroski_f_score_neg2fy,
       pio.feat_piotroski_f_score_neg3fy,
       -- Exact median of the four never-NULL scores: (sum - max - min) / 2.
       ((pio.feat_piotroski_f_score_fy + pio.feat_piotroski_f_score_neg1fy +
         pio.feat_piotroski_f_score_neg2fy + pio.feat_piotroski_f_score_neg3fy
	       - GREATEST(pio.feat_piotroski_f_score_fy, pio.feat_piotroski_f_score_neg1fy,
	                  pio.feat_piotroski_f_score_neg2fy, pio.feat_piotroski_f_score_neg3fy)
	       - LEAST(pio.feat_piotroski_f_score_fy, pio.feat_piotroski_f_score_neg1fy,
	               pio.feat_piotroski_f_score_neg2fy, pio.feat_piotroski_f_score_neg3fy)) /
        2.0)::double precision                                                                                                                                                                                 AS feat_median_piotroski_f_score
FROM pml.pml_df
	     CROSS JOIN LATERAL (SELECT pml.piotroski_f_score(return_on_assets_roa_pct_fy, return_on_assets_roa_pct_neg1fy,
	                                                      cfo_fy, net_income_fy,
	                                                      long_term_debt_equity_fy, long_term_debt_equity_neg1fy,
	                                                      current_ratio_fy, current_ratio_neg1fy,
	                                                      shrs_out, shrs_out_neg1fy,
	                                                      gross_profit_margin_pct_fy, gross_profit_margin_pct_neg1fy,
	                                                      asset_turnover_fy, asset_turnover_neg1fy)     AS feat_piotroski_f_score_fy,
	                                pml.piotroski_f_score(return_on_assets_roa_pct_neg1fy, return_on_assets_roa_pct_neg2fy,
	                                                      cfo_neg1fy, net_income_neg1fy,
	                                                      long_term_debt_equity_neg1fy, long_term_debt_equity_neg2fy,
	                                                      current_ratio_neg1fy, current_ratio_neg2fy,
	                                                      shrs_out_neg1fy, shrs_out_neg2fy,
	                                                      gross_profit_margin_pct_neg1fy, gross_profit_margin_pct_neg2fy,
	                                                      asset_turnover_neg1fy, asset_turnover_neg2fy) AS feat_piotroski_f_score_neg1fy,
	                                pml.piotroski_f_score(return_on_assets_roa_pct_neg2fy, return_on_assets_roa_pct_neg3fy,
	                                                      cfo_neg2fy, net_income_neg2fy,
	                                                      long_term_debt_equity_neg2fy, long_term_debt_equity_neg3fy,
	                                                      current_ratio_neg2fy, current_ratio_neg3fy,
	                                                      shrs_out_neg2fy, shrs_out_neg3fy,
	                                                      gross_profit_margin_pct_neg2fy, gross_profit_margin_pct_neg3fy,
	                                                      asset_turnover_neg2fy, asset_turnover_neg3fy) AS feat_piotroski_f_score_neg2fy,
	                                pml.piotroski_f_score(return_on_assets_roa_pct_neg3fy, return_on_assets_roa_pct_neg4fy,
	                                                      cfo_neg3fy, net_income_neg3fy,
	                                                      long_term_debt_equity_neg3fy, long_term_debt_equity_neg4fy,
	                                                      current_ratio_neg3fy, current_ratio_neg4fy,
	                                                      shrs_out_neg3fy, shrs_out_neg4fy,
	                                                      gross_profit_margin_pct_neg3fy, gross_profit_margin_pct_neg4fy,
	                                                      asset_turnover_neg3fy, asset_turnover_neg4fy) AS feat_piotroski_f_score_neg3fy) pio,
	     -- EPS beat counts backing feat_eps_beat_rate{,_annual}. Same
	     -- pml.beat_counts LATERAL idiom as mv_pymc_earnings_beat.
	     LATERAL pml.beat_counts(ARRAY [eps_neg0fqsurprise_pct, eps_neg1fqsurprise_pct, eps_neg2fqsurprise_pct, eps_neg3fqsurprise_pct, eps_neg4fqsurprise_pct]::NUMERIC[])                        bc_q(n_total, n_beats),
	     LATERAL pml.beat_counts(ARRAY [eps_neg0fysurprise_pct, eps_neg1fysurprise_pct, eps_neg2fysurprise_pct, eps_neg3fysurprise_pct, eps_neg4fysurprise_pct, eps_neg5fysurprise_pct]::NUMERIC[]) bc_y(n_total, n_beats);

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_pymc_kalman_pt_isin ON pml.mv_pymc_kalman_pt (isin);

-- ---- 4. DCFPriceTarget -------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS pml.mv_pymc_dcf_pt AS
SELECT isin,
       ticker,
       trading_region,
       region,
       country,
       trading_country,
       exchange,
       unit,
       style_class,
       size_class,
       sector,
       industry,
       price_target                                                                AS observed_pt,
       last_price                                                                  AS observed_price,
       market_cap,
       enterprise_value,
       shrs_out,
       fcf_ltm                                                                     AS feat_fcf_ltm,
       fcf_est_avg_fy1e                                                            AS feat_fcf_fy1e,
       fcf_est_avg_fy2e                                                            AS feat_fcf_fy2e,
       fcf_est_avg_fy3e                                                            AS feat_fcf_fy3e,
       fcf_est_avg_fy4e                                                            AS feat_fcf_fy4e,
       fcf_est_avg_fy5e                                                            AS feat_fcf_fy5e,
       pml.calc_change_ratio(fcf_est_avg_fy1e::NUMERIC, fcf_ltm::NUMERIC)          AS feat_fcf_growth_1y,
       pml.calc_change_ratio(fcf_est_avg_fy3e::NUMERIC, fcf_est_avg_fy1e::NUMERIC) AS feat_fcf_growth_2y,
       -- Terminal-period growth proxy (FY5 vs FY3) – informs DCF terminal value
       pml.calc_change_ratio(fcf_est_avg_fy5e::NUMERIC, fcf_est_avg_fy3e::NUMERIC) AS feat_fcf_terminal_growth,
       -- Re-investment rate proxy (capex / cfo) and capex burden vs FCF
       pml.safe_divide(capital_expenditure_ltm, cfo_ltm)                           AS feat_reinvest_rate,
       pml.safe_divide(capital_expenditure_ltm, fcf_ltm)                           AS feat_capex_to_fcf,
       cfo_ltm                                                                     AS feat_cfo_ltm,
       -- Historical realised CAGRs anchor priors on long-run growth
       tot_return_pct_cagr_3y                                                      AS feat_tr_cagr_3y,
       tot_return_pct_cagr_10y                                                     AS feat_tr_cagr_10y,
       peg_ntm                                                                     AS feat_peg_ntm,
       ev_sales_ltm                                                                AS feat_ev_sales_ltm,
       ev_ebitda_ntm                                                               AS feat_ev_ebitda_ntm,
       return_on_assets_roa_pct_ltm                                                AS feat_roa_ltm,
       gross_profit_margin_pct_ltm                                                 AS feat_gpm_ltm,
       beta_5y                                                                     AS feat_beta_5y,
       -- ---- Cross-cutting market-cap / EV size & trend feats ----
       pml.calc_change_ratio(market_cap, market_cap_neg1fy)                        AS feat_mcap_trend_1y,
       pml.safe_divide(market_cap, market_cap_3yavg)                               AS feat_mcap_vs_3yavg,
       pml.safe_divide(enterprise_value, enterprise_value_3yavg)                   AS feat_ev_vs_3yavg
FROM pml.pml_df;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_pymc_dcf_pt_isin ON pml.mv_pymc_dcf_pt (isin);

-- ---- 5. DividendSafetyBayesian -----------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS pml.mv_pymc_dividend_safety AS
SELECT isin,
       ticker,
       trading_region,
       region,
       country,
       trading_country,
       exchange,
       unit,
       style_class,
       size_class,
       sector,
       industry,
       div_yield_ltm                                                                              AS observed_div_yield,
       dividend_streak                                                                            AS n_streak,
       dividend_record_frequency                                                                  AS feat_div_frequency,
       pml.fcf_dividend_coverage(fcf_ltm::NUMERIC, common_dividends_paid_ltm::NUMERIC)            AS feat_fcf_coverage,
       -- Cash-flow (not FCF) coverage – complementary cushion signal
       pml.fcf_dividend_coverage(cfo_ltm::NUMERIC, common_dividends_paid_ltm::NUMERIC)            AS feat_cfo_coverage,
       -- Earnings payout ratio (DPS / EPS) – classic safety gauge
       pml.safe_divide(dividend_per_share_ltm,
                       NULLIF(net_eps_basic_ltm, 0))                                              AS feat_eps_payout_ratio,
       pml.calc_change_ratio(dividend_per_share_ltm::NUMERIC, dividend_per_share_neg1fy::NUMERIC) AS feat_dps_growth_1y,
       -- Longer-run DPS CAGRs reveal sustainability beyond a 1y change
       pml.calc_change_ratio(dividend_per_share_ltm::NUMERIC, dividend_per_share_neg3fy::NUMERIC) AS feat_dps_growth_3y,
       pml.calc_change_ratio(dividend_per_share_ltm::NUMERIC, dividend_per_share_neg5fy::NUMERIC) AS feat_dps_growth_5y,
       buyback_yield_ltm                                                                          AS feat_buyback_yield,
       -- Total shareholder yield (dividends + buybacks)
       (div_yield_ltm + COALESCE(buyback_yield_ltm, 0))                                           AS feat_total_yield,
       repurchase_common_stock_ltm                                                                AS feat_repurchases_ltm,
       altman_z_score_ltm                                                                         AS feat_altman_z,
       return_on_assets_roa_pct_ltm                                                               AS feat_roa_ltm,
       div_yield_ltm - div_yield_5yavgltm                                                         AS feat_yield_spread_vs_5y,
       -- ---- Cross-cutting market-cap / EV size & trend feats ----
       pml.calc_change_ratio(market_cap, market_cap_neg1fy)                                       AS feat_mcap_trend_1y,
       pml.safe_divide(market_cap, market_cap_3yavg)                                              AS feat_mcap_vs_3yavg,
       pml.safe_divide(enterprise_value, enterprise_value_3yavg)                                  AS feat_ev_vs_3yavg
FROM pml.pml_df;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_pymc_dividend_safety_isin ON pml.mv_pymc_dividend_safety (isin);

-- ---- 6. CreditRiskBayesian ---------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS pml.mv_pymc_credit_risk AS
SELECT isin,
       ticker,
       trading_region,
       region,
       country,
       trading_country,
       exchange,
       unit,
       style_class,
       size_class,
       sector,
       industry,
       altman_z_score_ltm                                                                              AS observed_altman_z,
       pml.altman_zone(altman_z_score_ltm)                                                             AS feat_distress_zone,
       pml.calc_change_ratio(altman_z_score_ltm, altman_z_score_neg1fy)                                AS feat_z_trend_1y,
       -- Multi-year Z-score trajectory: persistent deterioration is a strong default signal
       pml.calc_change_ratio(altman_z_score_ltm, altman_z_score_neg3fy)                                AS feat_z_trend_3y,
       pml.safe_divide(cfo_ltm, NULLIF(capital_expenditure_ltm, 0))                                    AS feat_cfo_capex_cov,
       pml.safe_divide(fcf_ltm, enterprise_value)                                                      AS feat_fcf_yield,
       -- Net financing footprint: negative cff_ltm means firm is repaying debt / returning capital
       pml.safe_divide(cff_ltm, enterprise_value)                                                      AS feat_cff_to_ev,
       -- Equity issuance vs buybacks: dilution signals distress, buybacks signal cushion
       pml.safe_divide(issuance_common_stock_ltm - repurchase_common_stock_ltm,
                       NULLIF(market_cap, 0))                                                          AS feat_net_equity_issuance,
       -- Headcount change as a leading operational-distress indicator
       pml.calc_change_ratio(full_time_employees_fy,
                             full_time_employees_neg1fy)                                               AS feat_employee_growth_1y,
       p_b_ltm                                                                                         AS feat_pb_ltm,
       beta_2y                                                                                         AS feat_beta_2y,
       volatility_6m                                                                                   AS feat_vol_6m,
       volatility_1y                                                                                   AS feat_vol_1y,
       -- ---- Cross-cutting market-cap / EV size & trend feats ----
       pml.calc_change_ratio(market_cap, market_cap_neg1fy)                                            AS feat_mcap_trend_1y,
       pml.safe_divide(market_cap, market_cap_3yavg)                                                   AS feat_mcap_vs_3yavg,
       pml.safe_divide(enterprise_value, enterprise_value_3yavg)                                       AS feat_ev_vs_3yavg
FROM pml.pml_df;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_pymc_credit_risk_isin ON pml.mv_pymc_credit_risk (isin);

-- ---- 7. AccountingAnomalyBayesian --------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS pml.mv_pymc_accounting_anomaly AS
SELECT isin,
       ticker,
       trading_region,
       region,
       country,
       trading_country,
       exchange,
       unit,
       style_class,
       size_class,
       sector,
       industry,
       eps_adj_ltm                                                                        AS observed_eps_adj,
       pml.accruals_ratio(net_eps_basic_ltm * shrs_out, cfo_ltm, enterprise_value)        AS feat_accruals_ratio,
       pml.calc_change_ratio(gross_profit_margin_pct_ltm, gross_profit_margin_pct_neg1fy) AS feat_gpm_change_1y,
       pml.calc_change_ratio(sales_neg0fyactual, sales_neg1fyactual)                      AS feat_sales_growth_1y,
       pml.calc_change_ratio(ebit_neg0fyactual, ebit_neg1fyactual)                        AS feat_ebit_growth_1y,
       pml.calc_change_ratio(ebitda_neg0fyactual, ebitda_neg1fyactual)                    AS feat_ebitda_growth_1y,
       pml.safe_divide(capital_expenditure_ltm, cfo_ltm)                                  AS feat_capex_intensity,
       pml.safe_divide(eps_adj_ltm - net_eps_basic_ltm, NULLIF(net_eps_basic_ltm, 0))     AS feat_eps_adj_gap,
       -- Cash flow composition: heavy CFI/CFF use vs CFO can mask earnings quality issues
       pml.safe_divide(cfi_ltm, NULLIF(cfo_ltm, 0))                                       AS feat_cfi_to_cfo,
       pml.safe_divide(cff_ltm, NULLIF(cfo_ltm, 0))                                       AS feat_cff_to_cfo,
       -- Beneish-style equity inflation (share count growth net of buybacks)
       pml.calc_change_ratio(shrs_out, shrs_out_neg1fy)                                   AS feat_share_inflation_1y,
       pml.safe_divide(issuance_common_stock_ltm, NULLIF(market_cap, 0))                  AS feat_issuance_intensity,
       -- Sales-vs-employee productivity (operational consistency check)
       pml.calc_change_ratio(full_time_employees_fy, full_time_employees_neg1fy)          AS feat_employee_growth_1y,
       -- FCF-per-share divergence from EPS – classic earnings-quality red flag
       pml.calc_change_ratio(fcf_per_share_ltm, net_eps_basic_ltm)                        AS feat_fcfps_vs_eps_gap,
       peg_ntm                                                                            AS feat_peg_ntm,
       -- ---- Cross-cutting market-cap / EV size & trend feats ----
       pml.calc_change_ratio(market_cap, market_cap_neg1fy)                               AS feat_mcap_trend_1y,
       pml.safe_divide(market_cap, market_cap_3yavg)                                      AS feat_mcap_vs_3yavg,
       pml.safe_divide(enterprise_value, enterprise_value_3yavg)                          AS feat_ev_vs_3yavg
FROM pml.pml_df;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_pymc_accounting_anomaly_isin ON pml.mv_pymc_accounting_anomaly (isin);

-- =============================================================================
-- UNIFIED CATALOGUE VIEWS  (drive notebook MODEL_FEATURE_CONTAINERS registry)
-- =============================================================================
-- One row per (model_target, pymc_role, column_name). The notebook's
-- `attach_features(...)` helper consumes this view directly so each PyMC model
-- gets its coords/observed/mutable_predictor/constant_data lists from SQL.
CREATE OR REPLACE VIEW pml.vw_pymc_feature_catalogue AS
SELECT m.model_name                                                 AS model_target,
       COALESCE(fa.pymc_role, md.pymc_role)                         AS pymc_role,
       md.column_name,
       md.category,
       md.feature_role,
       COALESCE(fa.feature_alias, md.feature_alias, md.column_name) AS feature_alias,
       md.data_type,
       md.description
FROM pml.pml_df_metadata                                md
	     CROSS JOIN LATERAL UNNEST(md.model_targets) AS m(model_name)
	     LEFT JOIN  pml.pml_df_feature_alias            fa
	                ON fa.column_name = md.column_name AND fa.model_target = m.model_name
WHERE COALESCE(fa.pymc_role, md.pymc_role) <> 'excluded';

-- Per-model feature alias list (used by `_resolve_<model>_feature_aliases`).
CREATE OR REPLACE VIEW pml.vw_pymc_feature_aliases AS
SELECT model_target,
       ARRAY_AGG(feature_alias ORDER BY feature_alias)
       FILTER (WHERE pymc_role = 'mutable_predictor')                                             AS feature_aliases,
       ARRAY_AGG(feature_alias ORDER BY feature_alias) FILTER (WHERE pymc_role = 'observed')      AS observed_aliases,
       ARRAY_AGG(feature_alias ORDER BY feature_alias)
       FILTER (WHERE pymc_role = 'constant_data')                                                 AS constant_data_aliases,
       ARRAY_AGG(feature_alias ORDER BY feature_alias) FILTER (WHERE pymc_role = 'coord')         AS coord_aliases
FROM pml.vw_pymc_feature_catalogue
GROUP BY model_target;

-- Coverage diagnostic: counts of non-null mutable_predictor columns per model.
CREATE OR REPLACE VIEW pml.vw_pymc_feature_coverage AS
SELECT model_target, pymc_role, COUNT(*) AS n_columns
FROM pml.vw_pymc_feature_catalogue
GROUP BY model_target, pymc_role
ORDER BY model_target, pymc_role;

-- =============================================================================
-- COVERAGE REGRESSION CHECK  (Findings 1/3/4 fail loudly on refresh)
-- =============================================================================
-- Contract: every feat_* / observed_* / n_* column emitted by each mv_pymc_*
-- must map to exactly ONE pml.vw_pymc_feature_catalogue row whose feature_alias
-- equals that MV column name, for the MV's model_target. Anything else means a
-- model would silently reindex the column to 0.0 (Finding 1) or list an alias
-- the MV never emits (Findings 3/4).
--
-- This view reconciles the live MV output columns (pg_catalog.pg_attribute)
-- against the catalogue, in BOTH directions:
--   * MISSING_FROM_CATALOGUE  : MV emits the column but the catalogue has no
--                               matching feature_alias for that model.
--   * DUPLICATE_CATALOGUE_ALIAS: more than one catalogue row claims the alias.
--   * PHANTOM_CATALOGUE_ALIAS  : catalogue lists a feat_*/observed_*/n_* alias
--                               the MV never emits (over-registration).
CREATE OR REPLACE VIEW pml.vw_pymc_catalogue_coverage_check AS
WITH mv_map(mv_name, model_target) AS (VALUES ('mv_pymc_earnings_beat', 'earnings_beat'),
                                              ('mv_pymc_price_target', 'price_target'),
                                              ('mv_pymc_kalman_pt', 'kalman_pt'),
                                              ('mv_pymc_dcf_pt', 'dcf_pt'),
                                              ('mv_pymc_dividend_safety', 'dividend_safety'),
                                              ('mv_pymc_credit_risk', 'credit_risk'),
                                              ('mv_pymc_accounting_anomaly', 'accounting_anomaly')
                                      ),
     -- NOTE: pg_class/pg_attribute (not information_schema.columns) — the SQL
     -- standard information_schema does NOT expose materialized-view columns,
     -- which silently emptied mv_cols and flagged every alias as PHANTOM.
     mv_cols                       AS (SELECT mm.model_target, a.attname::TEXT AS feat_name
                                       FROM mv_map                        mm
	                                            JOIN pg_catalog.pg_class     cl
	                                                 ON cl.relname = mm.mv_name AND cl.relkind = 'm'
	                                            JOIN pg_catalog.pg_namespace ns
	                                                 ON ns.oid = cl.relnamespace AND ns.nspname = 'pml'
	                                            JOIN pg_catalog.pg_attribute a
	                                                 ON a.attrelid = cl.oid AND a.attnum > 0 AND NOT a.attisdropped
                                       WHERE a.attname LIKE 'feat\_%'
	                                      OR a.attname LIKE 'observed\_%'
	                                      OR a.attname LIKE 'n\_%'
                                      ),
     cat                           AS (SELECT model_target, feature_alias, COUNT(*) AS n_rows
                                       FROM pml.vw_pymc_feature_catalogue
                                       WHERE feature_alias LIKE 'feat\_%'
	                                      OR feature_alias LIKE 'observed\_%'
	                                      OR feature_alias LIKE 'n\_%'
                                       GROUP BY model_target, feature_alias
                                      )
-- MV side: every emitted feat_/observed_/n_ column must resolve to one alias.
SELECT mc.model_target,
       mc.feat_name            AS feat_name,
       COALESCE(cat.n_rows, 0) AS catalogue_rows,
       CASE
	       WHEN cat.n_rows IS NULL THEN 'MISSING_FROM_CATALOGUE'
	       WHEN cat.n_rows > 1 THEN 'DUPLICATE_CATALOGUE_ALIAS'
	       ELSE 'OK' END       AS status
FROM mv_cols mc
	     LEFT JOIN cat ON cat.model_target = mc.model_target AND cat.feature_alias = mc.feat_name
UNION ALL
-- Catalogue side: feat_/observed_/n_ aliases the MV never emits (phantoms).
SELECT c.model_target, c.feature_alias AS feat_name, 0 AS catalogue_rows, 'PHANTOM_CATALOGUE_ALIAS' AS status
FROM cat                   c
	     LEFT JOIN mv_cols mc ON mc.model_target = c.model_target AND mc.feat_name = c.feature_alias
WHERE mc.feat_name IS NULL;

-- Fail-fast assertion (mirrors the PML_STRICT_STREAK_MERGE convention). Raises
-- if any MV column is unregistered / duplicated / phantom in the catalogue.
CREATE OR REPLACE FUNCTION pml.assert_pymc_catalogue_coverage() RETURNS VOID
	LANGUAGE plpgsql AS
$$
DECLARE
	v_count      INT;
	v_violations TEXT;
BEGIN
	SELECT COUNT(*),
	       string_agg(format('%s.%s [%s]', model_target, feat_name, status), ', ' ORDER BY model_target, feat_name)
	INTO v_count, v_violations
	FROM pml.vw_pymc_catalogue_coverage_check
	WHERE status <> 'OK';

	IF v_count > 0 THEN
		RAISE EXCEPTION 'PyMC catalogue coverage check failed for % column(s): %', v_count, v_violations USING HINT =
				'Every feat_/observed_/n_ column emitted by each mv_pymc_* must have exactly one pml.vw_pymc_feature_catalogue row with a matching feature_alias for its model_target (see pml.vw_pymc_catalogue_coverage_check).';
	END IF;
END;
$$;

-- =============================================================================
-- REFRESH HELPER  (refresh all per-model MVs in one call)
-- =============================================================================
-- Drop the previous single-arg signature so adding `assert_coverage` does not
-- create an ambiguous overload for `CALL pml.refresh_pymc_materialized_views();`.
-- `assert_coverage` is opt-in (default FALSE) mirroring the env-gated
-- PML_STRICT_STREAK_MERGE fail-fast convention: pass TRUE (e.g. in CI /
-- regression) to make Findings 1/3/4 raise on refresh. Call
-- pml.assert_pymc_catalogue_coverage() directly for an ad-hoc gate.
DROP PROCEDURE IF EXISTS pml.refresh_pymc_materialized_views(BOOLEAN);
CREATE OR REPLACE PROCEDURE pml.refresh_pymc_materialized_views(use_concurrently BOOLEAN DEFAULT TRUE,
                                                                assert_coverage  BOOLEAN DEFAULT FALSE)
	LANGUAGE plpgsql AS
$$
DECLARE
	mv          TEXT;
	schema_part TEXT;
	table_part  TEXT;
	mvs         TEXT[] := ARRAY [ 'pml.mv_pymc_earnings_beat', 'pml.mv_pymc_price_target', 'pml.mv_pymc_kalman_pt', 'pml.mv_pymc_dcf_pt', 'pml.mv_pymc_dividend_safety', 'pml.mv_pymc_credit_risk', 'pml.mv_pymc_accounting_anomaly' ];
BEGIN
	FOREACH mv IN ARRAY mvs
		LOOP
			-- Split "schema.table" into its two identifier parts so %I quotes correctly
			schema_part := split_part(mv, '.', 1);
			table_part := split_part(mv, '.', 2);

			IF use_concurrently THEN
				EXECUTE format('REFRESH MATERIALIZED VIEW CONCURRENTLY %I.%I', schema_part, table_part);
				ELSE
					EXECUTE format('REFRESH MATERIALIZED VIEW %I.%I', schema_part, table_part);
			END IF;
			END LOOP;

	-- Fail loudly if the MV feature surface and the catalogue have diverged.
	IF assert_coverage THEN PERFORM pml.assert_pymc_catalogue_coverage(); END IF;
END;
$$;

-- =============================================================================
-- USAGE
-- =============================================================================
-- 1. Refresh all PyMC feature MVs:
--      CALL pml.refresh_pymc_materialized_views();
--
--    Full signature (both arguments are frequently overlooked):
--      CALL pml.refresh_pymc_materialized_views(
--               use_concurrently => TRUE,   -- REFRESH ... CONCURRENTLY
--               assert_coverage  => FALSE); -- run the catalogue coverage gate
--
--    `assert_coverage => TRUE` runs pml.assert_pymc_catalogue_coverage() after
--    the refresh, which RAISES if any mv_pymc_* feat_/observed_/n_ column is
--    unregistered, duplicated or phantom in the catalogue. It defaults to
--    FALSE because the non-kalman models still carry known violations; the
--    kalman_pt path is clean. Enumerate what is outstanding with:
--      SELECT model_target, status, count(*),
--             string_agg(feat_name, ', ' ORDER BY feat_name)
--      FROM pml.vw_pymc_catalogue_coverage_check
--      WHERE status <> 'OK' GROUP BY 1, 2 ORDER BY 1, 2;
--    Fix MISSING_FROM_CATALOGUE / PHANTOM_CATALOGUE_ALIAS /
--    DUPLICATE_CATALOGUE_ALIAS in pml_df_metadata_populate.sql, then flip this
--    default to TRUE so refreshes stay honest.
--
-- 2. Drive the notebook's MODEL_FEATURE_CONTAINERS from SQL:
--      SELECT * FROM pml.vw_pymc_feature_aliases WHERE model_target = 'earnings_beat';
--
-- 3. Load a model's ready-to-use feature matrix in Python:
--      SELECT * FROM pml.mv_pymc_price_target WHERE isin = ANY(:isins);
--
-- 4. Filter columns by pymc role for a specific model:
--      SELECT column_name FROM pml.vw_pymc_feature_catalogue
--      WHERE model_target = 'dcf_pt' AND pymc_role = 'mutable_predictor';
-- =============================================================================