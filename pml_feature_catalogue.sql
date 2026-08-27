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

-- ---- 3b. KalmanFilterPriceTarget v2 (correlated trail) -----------------------
-- =============================================================================
-- pml.mv_pymc_kalman_pt_v2 -- feature matrix for the v2 correlated-trail model
-- =============================================================================
--
-- WHY A v2 MV EXISTS
-- ------------------
-- The v2 model (probabilistic_ml_model/pymc_models/KalmanFilterModel_v2.py)
-- treats a name lookback trail as ONE correlated observation vector rather than
-- T independent reads. That changes what the feature layer has to supply:
--
--   1. The RESPONSE ITSELF, not its ingredients. v1 rebuilt log(pt/px) per
--      lookback in Python from eight raw columns, so the definition of the
--      modelled quantity lived in a Python helper rather than in SQL. It is a
--      derived feature and belongs here, next to every other feat_ column.
--   2. CALENDAR OFFSETS. The OU kernel is exp(-gap_days / ell); it needs real
--      day gaps, not a time index. v1 standardised the time axis and threw the
--      gaps away, which is why one length-scale could not reproduce a
--      correlation of 0.97 at 7 days and 0.43 at 365 days.
--   3. PER-LOOKBACK COVERAGE. price_target_num_{lb}_ago already exists but was
--      never surfaced, so v1 used one precision weight for all T -- charging
--      a 30-analyst consensus today and a 4-analyst consensus last year the
--      same measurement precision.
--
-- The empirical basis, measured on this MV (2026-08-18, n ~ 6.5k). Correlation
-- between the log-uplift at two lookbacks against the calendar gap between them
-- is fit by a two-parameter kernel over all 15 available pairs:
--
--     r(delta) = rho_inf + (1 - rho_inf) * exp(-delta / ell)
--     rho_inf = 0.423, ell =  95.2 d   RMSE 0.033   (raw)
--     rho_inf = 0.334, ell = 104.7 d   RMSE 0.026   (after removing the crossed
--                                                    group means the model fits)
--
-- so ~33% of within-name response variance is a permanent per-name level and
-- ~67% decays with a 73-day half-life. Those are the two latent blocks v2
-- estimates; this MV exists to make them estimable.
--
-- DESIGN: A DERIVED MV, NOT A FORK
-- --------------------------------
-- This is built ON TOP OF pml.mv_pymc_kalman_pt rather than forking its ~420
-- lines. Consequences, all deliberate:
--   * the raw feature definitions keep exactly one home (this file);
--   * the v2 diff is reviewable -- everything below is genuinely new;
--   * refresh ORDER MATTERS: v1 must be refreshed before v2. Both
--     pml.refresh_pymc_materialized_views (array order) and
--     pml.refresh_kalman_pt_v2 (explicit parent leg) enforce it;
--   * every kalman_pt catalogue row must ALSO carry the kalman_pt_v2 tag,
--     because `SELECT b.*` re-emits every parent column under the child MV.
--     Section 7k of pml_df_metadata_populate.sql does that sweep.
-- The cost is one extra materialisation of ~6.5k rows, negligible next to the
-- 400-line duplication it avoids.
--
-- IF NOT EXISTS, like its seven siblings in this file -- which means re-running
-- this script does NOT pick up a definition change. Editing the body below
-- requires an explicit `DROP MATERIALIZED VIEW pml.mv_pymc_kalman_pt_v2;`
-- first. (True of every MV here; stated once, on the newest one.)
--
-- REPRODUCIBILITY
-- ---------------
-- v1 computes seven days_* horizons against CURRENT_DATE, so it is not
-- reproducible across refresh dates. v2 cannot fix that from here, but it stops
-- the problem being SILENT: built_at stamps every row with the refresh moment,
-- so a run artifacts record the as-of date they were built against and two runs
-- can be told apart. A genuine point-in-time backtest needs an as-of parameter,
-- which an MV cannot take -- see pml.kalman_pt_v2_asof() below for the function
-- form that can.
--
-- UNITS: all feat_* are raw decimals (0.25 = +25%), matching the 0.9.9.7
-- convention. Day offsets are integers. No percent scaling anywhere -- which is
-- exactly why the EPS block below divides the two *_surprise legs by 100 before
-- averaging; see CONSOLIDATED EPS BLOCK.
-- =============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS pml.mv_pymc_kalman_pt_v2 AS
WITH base AS (SELECT * FROM pml.mv_pymc_kalman_pt),

     -- ---------------------------------------------------------------------
     -- The response trail, in log space.
     --
     -- log_uplift(lb) = ln(price_target_at_lb / price_at_lb). Both legs are
     -- taken AT THE SAME LOOKBACK -- that is the whole point of the trail. A
     -- stale target over TODAY spot price is not a historical observation of
     -- the state, it is today price with extra noise, and mixing the two is
     -- how a momentum identity leaks into a price-target model.
     --
     -- pml.safe_divide guards the zero denominator; the ratio > 0 test guards
     -- ln() of a non-positive number (negative prices do not occur but a NULL
     -- masquerading as 0 has).
     -- ---------------------------------------------------------------------
     uplift AS (SELECT b.isin,
                       CASE
	                       WHEN pml.safe_divide(b.observed_pt, b.last_price) > 0
		                       THEN ln(pml.safe_divide(b.observed_pt, b.last_price))
	                       END AS lu_now,
                       CASE
	                       WHEN pml.safe_divide(b.price_target_1w_ago, b.price_1w_ago) > 0
		                       THEN ln(pml.safe_divide(b.price_target_1w_ago, b.price_1w_ago))
	                       END AS lu_1w,
                       CASE
	                       WHEN pml.safe_divide(b.price_target_1m_ago, b.price_1m_ago) > 0
		                       THEN ln(pml.safe_divide(b.price_target_1m_ago, b.price_1m_ago))
	                       END AS lu_1m,
                       CASE
	                       WHEN pml.safe_divide(b.price_target_3m_ago, b.price_3m_ago) > 0
		                       THEN ln(pml.safe_divide(b.price_target_3m_ago, b.price_3m_ago))
	                       END AS lu_3m,
                       CASE
	                       WHEN pml.safe_divide(b.price_target_6m_ago, b.price_6m_ago) > 0
		                       THEN ln(pml.safe_divide(b.price_target_6m_ago, b.price_6m_ago))
	                       END AS lu_6m,
                       CASE
	                       WHEN pml.safe_divide(b.price_target_1y_ago, b.price_1y_ago) > 0
		                       THEN ln(pml.safe_divide(b.price_target_1y_ago, b.price_1y_ago))
	                       END AS lu_1y
                FROM base b),

     -- ---------------------------------------------------------------------
     -- Consolidated EPS block, SPLIT BY QUANTITY, each leg with its coverage.
     --
     -- v1 carried five raw EPS columns into the drift matrix. Measured on the
     -- 2026-08-18 fit every one of them has |beta| <= 0.0115 and three straddle
     -- zero -- while feat_last_q_surprise is 52.1% NULL and feat_eps_beat_rate
     -- 45.4% NULL. Because the Python alignment layer zero-fills a missing
     -- column, the model was reading "no data" as "no surprise" for half the
     -- universe, which is not a small distinction: it is the difference between
     -- an informative zero and an absent measurement.
     --
     -- The first fix collapsed all five into ONE average. That was wrong for a
     -- different reason: the five legs are on three incompatible scales.
     --   feat_last_{q,y}_surprise are eps_neg0f{q,y}surprise_pct -- PERCENT,
     --       where 5.2 means +5.2%;
     --   feat_eps_beat_rate{,_annual} are n_beats / n_total -- shares in [0,1];
     --   feat_net_eps_drift is a pml.target_drift ratio -- a raw decimal.
     -- Averaging them makes the percent legs ~100x everything else, so the
     -- "consolidated signal" was the surprise legs wearing a different name,
     -- and it violated the raw-decimal convention this header claims two
     -- paragraphs up.
     --
     -- So: one column per quantity, each on one scale.
     --   feat_eps_signal_surprise -- mean of whichever surprise legs EXIST,
     --                               rescaled /100 to a signed raw decimal
     --   feat_eps_signal_beat     -- mean of whichever beat-rate legs EXIST,
     --                               already a share in [0,1]
     --   feat_eps_signal_coverage -- share of all five legs that existed, so
     --                               the model can learn the interaction rather
     --                               than being handed a counterfeit zero
     -- The TREND leg needs no new column: the parent feat_net_eps_drift is
     -- already a raw decimal ratio and is re-emitted by `SELECT b.*` above. It
     -- is re-admitted to the drift matrix in pymc_kalman_filter_pt_v2.py.
     --
     -- The `avg(v) ... WHERE v IS NOT NULL` idiom is what makes "mean of
     -- whichever legs exist, NULL when none does" correct: avg() over an empty
     -- set already yields NULL, no special case needed.
     -- ---------------------------------------------------------------------
     eps AS (SELECT b.isin,
                    (SELECT avg(v)
                     FROM unnest(ARRAY [b.feat_last_q_surprise / 100.0,
	                     b.feat_last_y_surprise / 100.0]) AS v
                     WHERE v IS NOT NULL)                          AS eps_surprise,
                    (SELECT avg(v)
                     FROM unnest(ARRAY [b.feat_eps_beat_rate,
	                     b.feat_eps_beat_rate_annual]) AS v
                     WHERE v IS NOT NULL)                          AS eps_beat,
                    (SELECT count(v)::DOUBLE PRECISION / 5.0
                     FROM unnest(ARRAY [b.feat_net_eps_drift, b.feat_last_q_surprise,
	                     b.feat_last_y_surprise, b.feat_eps_beat_rate,
	                     b.feat_eps_beat_rate_annual]) AS v
                     WHERE v IS NOT NULL)                          AS eps_coverage
             FROM base b)

SELECT b.*,

       -- ===================================================================
       -- v2 RESPONSE TRAIL (pymc_role = 'observed')
       -- ===================================================================
       u.lu_now                                                       AS feat_log_uplift_now,
       u.lu_1w                                                        AS feat_log_uplift_1w,
       u.lu_1m                                                        AS feat_log_uplift_1m,
       u.lu_3m                                                        AS feat_log_uplift_3m,
       u.lu_6m                                                        AS feat_log_uplift_6m,
       u.lu_1y                                                        AS feat_log_uplift_1y,

       -- Non-NULL cells in the trail. The v2 likelihood masks the rest, so this
       -- is the per-name T actually contributing -- and a name with 1 is a pure
       -- cross-section rider, not a panel member. Surface it so the workflow can
       -- report the distribution instead of discovering it in a coverage plot.
       (CASE WHEN u.lu_now IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN u.lu_1w IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN u.lu_1m IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN u.lu_3m IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN u.lu_6m IS NOT NULL THEN 1 ELSE 0 END +
        CASE WHEN u.lu_1y IS NOT NULL THEN 1 ELSE 0 END)::INT         AS n_trail_obs,

       -- ===================================================================
       -- CALENDAR OFFSETS -- REMOVED 2026-08-19, kept here as a tombstone
       -- ===================================================================
       -- trail_days_{now,1w,1m,3m,6m,1y} were emitted here as SQL LITERALS
       -- (0/7/30/91/182/365), identical on all ~6,500 rows, while the model
       -- built the same grid from DEFAULT_LOOKBACK_DAYS in Python and never
       -- read the columns. Two sources of truth for the OU kernel's x-axis, and
       -- the one the model used was the one the database could not see.
       --
       -- The x-axis now has one home: pml.vw_pymc_trail_days (defined below),
       -- read by KalmanFilterModel_v2.load_trail_days_map() and tied to this
       -- MV's feat_log_uplift_* columns by pml.assert_pymc_trail_days_map() --
       -- the foreign key a view cannot declare. Their metadata rows are retired
       -- in pml_df_metadata_populate.sql section 7l.
       --
       -- DO NOT RE-ADD THEM HERE. This block survived in this file until
       -- 2026-08-21 while the deployed MV had already dropped them, which was
       -- harmless only because `CREATE MATERIALIZED VIEW IF NOT EXISTS` silently
       -- no-ops: anyone following the documented DROP-and-recreate path would
       -- have resurrected six columns that section 7l then de-registers, landing
       -- them as MISSING_FROM_CATALOGUE -- the status where the alignment layer
       -- zero-fills in silence.

       -- ===================================================================
       -- PER-LOOKBACK ANALYST COVERAGE (pymc_role = 'constant_data')
       -- ===================================================================
       -- v1 applied ONE precision weight to every time step. A consensus built
       -- from 4 analysts a year ago and one built from 30 today are not equally
       -- precise measurements of the same latent, and the model had no way to
       -- say so. These feed the per-cell measurement scale in v2.
       --
       -- Registered as engineered self-rows rather than as aliases of
       -- price_target_num_{lb}_ago: `SELECT b.*` already emits those columns
       -- under their own names, and an alias row can only claim one name per
       -- (column_name, model_target), so aliasing would leave whichever name it
       -- did not claim MISSING_FROM_CATALOGUE.
       --
       -- The consequence is that the exported panel frame carries each of these
       -- five quantities under two names, which the export_duplicate_content gate
       -- correctly notices. It is declared there rather than "fixed" here --
       -- EXPORT_DECLARED_ALIASES['04_panel_frame_v2'] in pymc_kalman_filter_pt_v2.py
       -- cites this comment as the reason -- so the gate re-verifies the equality
       -- every run instead of re-reporting a settled trade-off as a finding.
       b.price_target_num_1w_ago                                      AS n_analysts_1w,
       b.price_target_num_1m_ago                                      AS n_analysts_1m,
       b.price_target_num_3m_ago                                      AS n_analysts_3m,
       b.price_target_num_6m_ago                                      AS n_analysts_6m,
       b.price_target_num_1y_ago                                      AS n_analysts_1y,

       -- ===================================================================
       -- CONSOLIDATED EPS BLOCK (pymc_role = 'mutable_predictor')
       -- ===================================================================
       e.eps_surprise                                                 AS feat_eps_signal_surprise,
       e.eps_beat                                                     AS feat_eps_signal_beat,
       COALESCE(e.eps_coverage, 0.0)                                  AS feat_eps_signal_coverage,

       -- ===================================================================
       -- PROVENANCE (pymc_role = 'derived_input')
       -- ===================================================================
       -- Stamps the refresh moment on every row. The parent days_* horizons are
       -- CURRENT_DATE-relative, so two refreshes of "the same" MV are different
       -- datasets; this makes that visible in the exported panel frame instead
       -- of being something you have to remember.
       --
       -- derived_input, NOT constant_data: constant_data would land a
       -- timestamptz in vw_pymc_feature_aliases.constant_data_aliases, and
       -- coerce_by_data_type() casts every alias handed to it to float64.
       now()                                                          AS built_at

FROM base b
	     JOIN uplift u ON u.isin = b.isin
	     JOIN eps    e ON e.isin = b.isin
-- The snapshot leg must exist: it is the anchor of the OU grid and the column
-- every downstream decision reads. A name without it is not a panel member.
WHERE u.lu_now IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_pymc_kalman_pt_v2_isin
	ON pml.mv_pymc_kalman_pt_v2 (isin);

-- Supports the trail-coverage filter the workflow applies when choosing a grid.
CREATE INDEX IF NOT EXISTS idx_mv_pymc_kalman_pt_v2_trail
	ON pml.mv_pymc_kalman_pt_v2 (n_trail_obs);

ALTER MATERIALIZED VIEW pml.mv_pymc_kalman_pt_v2 OWNER TO postgres;

COMMENT ON MATERIALIZED VIEW pml.mv_pymc_kalman_pt_v2 IS
	'v2 Kalman price-target feature matrix. Derived from mv_pymc_kalman_pt; adds '
		'the log-uplift response trail, its calendar offsets, per-lookback analyst '
		'coverage and a consolidated EPS block. Refresh AFTER mv_pymc_kalman_pt.';

COMMENT ON COLUMN pml.mv_pymc_kalman_pt_v2.feat_log_uplift_now IS
	'ln(price_target / last_price). Raw decimal log ratio. The snapshot response '
		'and the anchor (offset 0) of the OU time grid.';
COMMENT ON COLUMN pml.mv_pymc_kalman_pt_v2.n_trail_obs IS
	'Count of non-NULL trail cells, 1..6. The per-name T actually contributing to '
		'the likelihood; cells outside it are masked, not imputed.';
-- (No COMMENT for trail_days_1y: the six trail_days_* columns were retired on
--  2026-08-19 -- see the tombstone in the SELECT list above. This statement
--  survived the removal and would have failed with "column does not exist" the
--  first time anyone actually recreated the view, which is the usual first
--  symptom of the IF NOT EXISTS no-op.)
COMMENT ON COLUMN pml.mv_pymc_kalman_pt_v2.n_analysts_1y IS
	'Analyst count behind the 1y consensus (price_target_num_1y_ago). Per-cell '
		'measurement precision; v1 had one weight for all T.';
-- PROVENANCE CONTAINER since 2026-08-21, not a drift predictor. Kept in the MV
-- and in the catalogue -- withholding either would be MISSING_FROM_CATALOGUE,
-- not an exclusion -- but barred from the v2 drift design matrix in Python by
-- DRIFT_EXCLUSIONS in pymc_kalman_filter_pt_v2.py. Same treatment, and the same
-- reason, as feat_vol_drift on the parent MV.
COMMENT ON COLUMN pml.mv_pymc_kalman_pt_v2.feat_eps_signal_surprise IS
	'Mean of whichever of feat_last_{q,y}_surprise is non-NULL, divided by 100 so '
		'it is a signed RAW DECIMAL like every other feat_ column. NULL when neither '
		'leg exists. RETAINED FOR PROVENANCE AND EDA, excluded from the v2 drift '
		'matrix: on run 37e6d8966250 (n = 6,533) it measured 79.1% coverage against '
		'a next-thinnest 86.3%, |r(feat_log_uplift_now)| 0.0071 against a '
		'next-weakest 0.0862, and trail-contrast dominance 4.53 against a '
		'next-highest 0.93 -- the only drift column failing any admission test, and '
		'it failed all three. At 79.1% coverage a fifth of the universe enters '
		'mean-imputed, which attenuates the slope before the sampler sees it.';
COMMENT ON COLUMN pml.mv_pymc_kalman_pt_v2.feat_eps_signal_beat IS
	'Mean of whichever of feat_eps_beat_rate{,_annual} is non-NULL. A frequency in '
		'[0, 1] -- a different quantity from feat_eps_signal_surprise, which is a '
		'magnitude, which is why they are separate columns.';
COMMENT ON COLUMN pml.mv_pymc_kalman_pt_v2.feat_eps_signal_coverage IS
	'Share of all five EPS legs (net_eps_drift, last_{q,y}_surprise, '
		'eps_beat_rate{,_annual}) that were non-NULL, in [0, 1]. Lets the model '
		'distinguish an informative zero from an absent measurement.';
COMMENT ON COLUMN pml.mv_pymc_kalman_pt_v2.built_at IS
	'Refresh timestamp. The parent MV computes its days_* horizons against '
		'CURRENT_DATE, so this is what tells two refreshes apart.';


-- =============================================================================
-- pml.kalman_pt_v2_asof(p_asof) -- the point-in-time form
-- =============================================================================
-- An MV cannot take a parameter, which is why mv_pymc_kalman_pt days_* horizons
-- are pinned to CURRENT_DATE and why the whole feature matrix is unusable for a
-- backtest. This function is the escape hatch: it recomputes the date-relative
-- horizons against an arbitrary as-of date, leaving every price/target-derived
-- feature untouched (those are already point-in-time by construction).
--
-- STABLE, not IMMUTABLE: it reads the MV.
--
-- SIGN CONVENTIONS ARE THE PARENT CONVENTIONS, DELIBERATELY. In particular
-- days_since_fy_end is (fy_end_date - p_asof), which is NEGATIVE for a past
-- fiscal-year end -- the same expression mv_pymc_kalman_pt uses. It reads like
-- a sign bug and is not one: a function that silently disagreed with the column
-- it shadows would be far worse. Fix both together or neither.
--
-- COVERAGE: six of the parent SEVEN days_* horizons.
-- days_to_next_fiscal_quarter is omitted because the parent computes it as
-- (next_fiscal_quarter - CURRENT_DATE) where next_fiscal_quarter is a 1-4
-- quarter ORDINAL, not a date; reproducing that here would propagate the
-- defect rather than the convention. Left out knowingly, not overlooked.
--
-- SCOPE, stated plainly: this fixes the HORIZON columns only. The underlying
-- price and target trails are still whatever the last vendor load contained, so
-- this supports "what would the model have said about the calendar on date X",
-- not a full historical replay. A real replay needs versioned vendor snapshots,
-- which pml.staging does not retain.
CREATE OR REPLACE FUNCTION pml.kalman_pt_v2_asof(p_asof DATE DEFAULT CURRENT_DATE)
	RETURNS TABLE
	        (
		        isin                    TEXT,
		        days_to_next_earnings   INT,
		        days_since_last_report  INT,
		        days_to_next_fy_end     INT,
		        days_to_next_report     INT,
		        days_to_expected_report INT,
		        days_since_fy_end       INT,
		        asof_date               DATE
	        )
	LANGUAGE sql
	STABLE
	PARALLEL SAFE
AS
$$
SELECT v.isin,
       (v.next_earnings - p_asof)::INT,
       (p_asof - v.income_statement_report_date)::INT,
       (v.next_fy_end_date - p_asof)::INT,
       (v.next_income_statement_report_date - p_asof)::INT,
       (v.expected_report_date - p_asof)::INT,
       (v.fy_end_date - p_asof)::INT,
       p_asof
FROM pml.mv_pymc_kalman_pt_v2 v;
$$;

COMMENT ON FUNCTION pml.kalman_pt_v2_asof(DATE) IS
	'Recompute the date-relative horizon columns against an arbitrary as-of date. '
		'Horizons only -- the price/target trails are not versioned, so this is not '
		'a full historical replay. Sign conventions match mv_pymc_kalman_pt.';


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
-- TRAIL-DAYS MAP  (SSOT for the Kalman v2 OU kernel's x-axis)
-- =============================================================================
-- The calendar offset of each response column used to live in TWO places: as
-- six literal columns on mv_pymc_kalman_pt_v2 (0/7/30/91/182/365, identical on
-- every one of ~6,500 rows, so zero information and 6,500x the storage) and
-- again as DEFAULT_LOOKBACK_DAYS in KalmanFilterModel_v2.py, which is what the
-- model actually read. The MV columns were never consumed.
--
-- This view is the single source. It is deliberately a standalone VALUES list
-- rather than a SELECT over the MV: it has to survive
-- `DROP MATERIALIZED VIEW pml.mv_pymc_kalman_pt_v2` (the only way to change that
-- MV, since it is CREATE ... IF NOT EXISTS), and the offsets are metadata about
-- the grid rather than data about any name.
--
-- `response_column` is the mapping to the trail the offset describes. A view
-- cannot carry a real foreign key, so the equivalent guarantee is enforced by
-- pml.assert_pymc_trail_days_map() below, which is called from
-- pml.assert_pymc_catalogue_coverage() -- the same place every other
-- MV<->catalogue contract is checked.
--
-- Adding a lookback is a three-line change: a row here, the feat_log_uplift_*
-- column on the MV, and its catalogue row in pml_df_metadata_populate.sql.
CREATE OR REPLACE VIEW pml.vw_pymc_trail_days AS
SELECT v.lookback_key,
       v.response_column,
       v.trail_days,
       v.trail_rank
FROM (VALUES ('now', 'feat_log_uplift_now', 0, 0),
             ('1w', 'feat_log_uplift_1w', 7, 1),
             ('1m', 'feat_log_uplift_1m', 30, 2),
             ('3m', 'feat_log_uplift_3m', 91, 3),
             ('6m', 'feat_log_uplift_6m', 182, 4),
             ('1y', 'feat_log_uplift_1y', 365, 5)) AS v(lookback_key, response_column, trail_days,
                                                        trail_rank);

COMMENT ON VIEW pml.vw_pymc_trail_days IS
	'SSOT for the Kalman v2 OU kernel x-axis: lookback key -> response column -> nominal calendar offset in days. Read by KalmanFilterModel_v2.load_trail_days_map(); replaces the per-row trail_days_* columns dropped from mv_pymc_kalman_pt_v2 in favour of one metadata row per lookback.';

-- The foreign-key stand-in: every response_column must exist on the MV, and
-- every feat_log_uplift_* the MV emits must appear here. Materialized-view
-- columns are NOT listed in information_schema.columns, so resolve them through
-- pg_attribute.
CREATE OR REPLACE FUNCTION pml.assert_pymc_trail_days_map() RETURNS VOID
	LANGUAGE plpgsql AS
$$
DECLARE
	v_missing TEXT;
	v_extra   TEXT;
BEGIN
	WITH mv_cols AS (SELECT a.attname::TEXT AS col
	                 FROM pg_attribute a
		                      JOIN pg_class c ON c.oid = a.attrelid
		                      JOIN pg_namespace n ON n.oid = c.relnamespace
	                 WHERE n.nspname = 'pml'
		               AND c.relname = 'mv_pymc_kalman_pt_v2'
		               AND a.attnum > 0
		               AND NOT a.attisdropped)
	SELECT string_agg(m.response_column, ', ' ORDER BY m.trail_rank)
	INTO v_missing
	FROM pml.vw_pymc_trail_days m
	WHERE NOT EXISTS (SELECT 1 FROM mv_cols WHERE col = m.response_column);

	WITH mv_cols AS (SELECT a.attname::TEXT AS col
	                 FROM pg_attribute a
		                      JOIN pg_class c ON c.oid = a.attrelid
		                      JOIN pg_namespace n ON n.oid = c.relnamespace
	                 WHERE n.nspname = 'pml'
		               AND c.relname = 'mv_pymc_kalman_pt_v2'
		               AND a.attnum > 0
		               AND NOT a.attisdropped
		               AND a.attname::TEXT LIKE 'feat\_log\_uplift\_%')
	SELECT string_agg(col, ', ' ORDER BY col)
	INTO v_extra
	FROM mv_cols
	WHERE NOT EXISTS (SELECT 1 FROM pml.vw_pymc_trail_days m WHERE m.response_column = col);

	IF v_missing IS NOT NULL THEN
		RAISE EXCEPTION 'vw_pymc_trail_days maps response column(s) mv_pymc_kalman_pt_v2 does not emit: %', v_missing USING HINT =
				'Add the feat_log_uplift_* column to the MV, or drop the row from the trail-days map. The model builds its OU kernel x-axis from this view.';
	END IF;

	IF v_extra IS NOT NULL THEN
		RAISE EXCEPTION 'mv_pymc_kalman_pt_v2 emits response column(s) absent from vw_pymc_trail_days: %', v_extra USING HINT =
				'Every feat_log_uplift_* trail needs a calendar offset in pml.vw_pymc_trail_days or the model cannot place it on the OU grid.';
	END IF;
END;
$$;

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
                                              -- v2 is DERIVED from kalman_pt, so it re-emits every
                                              -- parent feat_/observed_/n_ column. Omitting it here
                                              -- would leave all of them unverified -- which reads as
                                              -- passing, not as failing.
                                              ('mv_pymc_kalman_pt_v2', 'kalman_pt_v2'),
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

	-- The trail-days map is the same class of contract -- a Python-visible
	-- surface that must agree with what the MV emits -- so it is asserted from
	-- the same entry point rather than needing its own call site.
	PERFORM pml.assert_pymc_trail_days_map();
END;
$$;

-- =============================================================================
-- REFRESH HELPER  (refresh all per-model MVs in one call)
-- =============================================================================
-- Drop the previous single-arg signature so adding `assert_coverage` does not
-- create an ambiguous overload for `CALL pml.refresh_pymc_materialized_views();`.
--
-- `assert_coverage` DEFAULTS TO TRUE since 2026-08-24. It was opt-in for one
-- reason only -- the non-kalman models carried ~50 real violations, so a strict
-- default would have failed every refresh on a defect nobody was about to fix.
-- That reason is gone: `pml.vw_pymc_catalogue_coverage_check` returns ZERO rows
-- database-wide and `pml.assert_pymc_catalogue_coverage()` passes, once
-- pml_df_metadata_populate.sql §7i.3 canonicalises the feat_pt_achievement_1y
-- self-row (verified against the live database on 2026-08-24).
--
-- Defaulting it on is the point of having built it. MISSING_FROM_CATALOGUE is
-- the dangerous status -- the MV emits a column, no catalogue row claims it, and
-- the alignment layer silently reindexes it to 0.0 -- and a check that is off by
-- default catches it only when someone remembers to ask. Pass FALSE explicitly
-- to refresh a database that has not had the §7i reconciliation applied yet.
DROP PROCEDURE IF EXISTS pml.refresh_pymc_materialized_views(BOOLEAN);
CREATE OR REPLACE PROCEDURE pml.refresh_pymc_materialized_views(use_concurrently BOOLEAN DEFAULT TRUE,
                                                                assert_coverage  BOOLEAN DEFAULT TRUE)
	LANGUAGE plpgsql AS
$$
DECLARE
	mv          TEXT;
	schema_part TEXT;
	table_part  TEXT;
	-- ORDER IS LOAD-BEARING, not alphabetical or historical: FOREACH walks the
	-- array in sequence, and mv_pymc_kalman_pt_v2 SELECTs from
	-- mv_pymc_kalman_pt. Refreshing the child first rebuilds it against a stale
	-- parent -- the mixed-vintage failure the analytics export already learned
	-- the hard way. Keep v2 immediately after v1.
	mvs         TEXT[] := ARRAY [ 'pml.mv_pymc_earnings_beat', 'pml.mv_pymc_price_target', 'pml.mv_pymc_kalman_pt', 'pml.mv_pymc_kalman_pt_v2', 'pml.mv_pymc_dcf_pt', 'pml.mv_pymc_dividend_safety', 'pml.mv_pymc_credit_risk', 'pml.mv_pymc_accounting_anomaly' ];
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

-- ---- Targeted two-MV refresh: pml.refresh_kalman_pt_v2 -----------------------
-- The full refresher above rebuilds all eight MVs. This is the narrow path for
-- iterating on the Kalman pair alone, and it exists mainly to make the
-- parent-then-child order impossible to get wrong: mv_pymc_kalman_pt_v2 SELECTs
-- from mv_pymc_kalman_pt, so a child-only refresh silently rebuilds v2 against
-- a stale v1. refresh_parent therefore defaults to TRUE -- pass FALSE only when
-- you have just refreshed the parent yourself.
--
-- pymc_kalman_filter_pt_v2.py names this procedure in its "MV returned no rows"
-- error, so keep the name stable.
CREATE OR REPLACE PROCEDURE pml.refresh_kalman_pt_v2(use_concurrently BOOLEAN DEFAULT TRUE,
                                                     refresh_parent   BOOLEAN DEFAULT TRUE)
	LANGUAGE plpgsql AS
$$
BEGIN
	IF refresh_parent THEN
		IF use_concurrently THEN
			REFRESH MATERIALIZED VIEW CONCURRENTLY pml.mv_pymc_kalman_pt;
			ELSE
				REFRESH MATERIALIZED VIEW pml.mv_pymc_kalman_pt;
		END IF;
		RAISE NOTICE 'refreshed pml.mv_pymc_kalman_pt';
	END IF;

	IF use_concurrently THEN
		REFRESH MATERIALIZED VIEW CONCURRENTLY pml.mv_pymc_kalman_pt_v2;
		ELSE
			REFRESH MATERIALIZED VIEW pml.mv_pymc_kalman_pt_v2;
	END IF;
	RAISE NOTICE 'refreshed pml.mv_pymc_kalman_pt_v2';
END;
$$;

COMMENT ON PROCEDURE pml.refresh_kalman_pt_v2(BOOLEAN, BOOLEAN) IS
	'Refresh the Kalman MV pair in dependency order (parent mv_pymc_kalman_pt, '
		'then child mv_pymc_kalman_pt_v2). Narrow alternative to '
		'refresh_pymc_materialized_views when iterating on the v2 model.';

-- =============================================================================
-- USAGE
-- =============================================================================
-- 1. Refresh all PyMC feature MVs:
--      CALL pml.refresh_pymc_materialized_views();
--
--    Full signature (both arguments are frequently overlooked):
--      CALL pml.refresh_pymc_materialized_views(
--               use_concurrently => TRUE,   -- REFRESH ... CONCURRENTLY
--               assert_coverage  => TRUE);  -- run the catalogue coverage gate
--
--    `assert_coverage` runs pml.assert_pymc_catalogue_coverage() after the
--    refresh, which RAISES if any mv_pymc_* feat_/observed_/n_ column is
--    unregistered, duplicated or phantom in the catalogue. It now defaults to
--    TRUE: the violations that kept it off are cleared database-wide. Pass
--    FALSE for a database without the §7i reconciliation. Enumerate what is
--    outstanding with:
--      SELECT model_target, status, count(*),
--             string_agg(feat_name, ', ' ORDER BY feat_name)
--      FROM pml.vw_pymc_catalogue_coverage_check
--      WHERE status <> 'OK' GROUP BY 1, 2 ORDER BY 1, 2;
--    Fix MISSING_FROM_CATALOGUE / PHANTOM_CATALOGUE_ALIAS /
--    DUPLICATE_CATALOGUE_ALIAS in pml_df_metadata_populate.sql, then flip this
--    default to TRUE so refreshes stay honest.
--
-- 1b. Refresh only the Kalman pair, in dependency order:
--      CALL pml.refresh_kalman_pt_v2();                       -- v1 then v2
--      CALL pml.refresh_kalman_pt_v2(refresh_parent => FALSE); -- v2 only
--
--    mv_pymc_kalman_pt_v2 SELECTs from mv_pymc_kalman_pt, so refresh_parent
--    defaults to TRUE. Pass FALSE only when you have just refreshed the parent
--    yourself -- otherwise v2 is rebuilt against a stale v1 and nothing says so.
--
-- 1c. Point-in-time calendar horizons (the only part of the Kalman feature
--     surface that is not already point-in-time by construction):
--      SELECT * FROM pml.kalman_pt_v2_asof(DATE '2026-06-30');
--
--    Horizons only. The price / target trails are not versioned, so this is not
--    a historical replay -- see the function header.
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
--
-- 5. The eight model targets, and which MV backs each:
--      earnings_beat      -> mv_pymc_earnings_beat
--      price_target       -> mv_pymc_price_target
--      kalman_pt          -> mv_pymc_kalman_pt
--      kalman_pt_v2       -> mv_pymc_kalman_pt_v2   (derived from kalman_pt)
--      dcf_pt             -> mv_pymc_dcf_pt
--      dividend_safety    -> mv_pymc_dividend_safety
--      credit_risk        -> mv_pymc_credit_risk
--      accounting_anomaly -> mv_pymc_accounting_anomaly
--    The allow-list is CHECK-enforced in pml_df_metadata.sql (from-scratch) and
--    re-applied idempotently at the top of pml_df_metadata_populate.sql (live).
-- =============================================================================