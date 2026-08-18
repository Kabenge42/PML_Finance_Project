-- ===========================================================================
-- pml.mv_pymc_kalman_pt_v2 -- feature matrix for the v2 correlated-trail model
-- ===========================================================================
--
-- WHY A v2 MV EXISTS
-- ------------------
-- The v2 model (probabilistic_ml_model/pymc_models/KalmanFilterModel_v2.py)
-- treats a name's lookback trail as ONE correlated observation vector rather
-- than T independent reads. That changes what the feature layer has to supply:
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
--      today's 30-analyst consensus and last year's 4-analyst consensus the
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
--   * the raw feature definitions keep exactly one home (pml_feature_catalogue.sql);
--   * the v2 diff is reviewable -- everything below is genuinely new;
--   * refresh ORDER MATTERS: v1 must be refreshed before v2. The procedure at
--     the bottom enforces it.
-- The cost is one extra materialisation of ~6.5k rows, which is negligible next
-- to the 400-line duplication it avoids.
--
-- SSOT NOTE
-- ---------
-- Per CLAUDE.md the authoritative MV definitions live in pml_feature_catalogue.sql,
-- and sql_scripts/pml/*.sql are pg_dump-style extracts (mostly stubs). This file
-- is real DDL and is intended to be FOLDED INTO pml_feature_catalogue.sql when
-- v2 is promoted; until then it is applied on its own and the catalogue rows are
-- registered by mv_pymc_kalman_pt_v2_metadata.sql, which must be run with it.
--
-- REPRODUCIBILITY
-- ---------------
-- v1 computes seven days_* horizons against CURRENT_DATE, so it is not
-- reproducible across refresh dates. v2 cannot fix that from here, but it stops
-- the problem being SILENT: built_at stamps every row with the refresh moment,
-- so a run's artifacts record the as-of date they were built against and two
-- runs can be told apart. A genuine point-in-time backtest needs an as-of
-- parameter, which an MV cannot take -- see pml.kalman_pt_v2_asof() at the
-- bottom for the function form that can.
--
-- UNITS: all feat_* are raw decimals (0.25 = +25%), matching the 0.9.9.7
-- convention. Day offsets are integers. No percent scaling anywhere.
-- ===========================================================================

DROP MATERIALIZED VIEW IF EXISTS pml.mv_pymc_kalman_pt_v2 CASCADE;

CREATE MATERIALIZED VIEW pml.mv_pymc_kalman_pt_v2 AS
WITH base AS (SELECT * FROM pml.mv_pymc_kalman_pt),

     -- ---------------------------------------------------------------------
     -- The response trail, in log space.
     --
     -- log_uplift(lb) = ln(price_target_at_lb / price_at_lb). Both legs are
     -- taken AT THE SAME LOOKBACK -- that is the whole point of the trail. A
     -- stale target over TODAY's price is not a historical observation of the
     -- state, it is today's price with extra noise, and mixing the two is how a
     -- momentum identity leaks into a price-target model.
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
     -- Consolidated EPS-surprise signal, with its own coverage.
     --
     -- v1 carried five raw EPS columns into the drift matrix. Measured on the
     -- 2026-08-18 fit every one of them has |beta| <= 0.0115 and three straddle
     -- zero -- while feat_last_q_surprise is 52.1% NULL and feat_eps_beat_rate
     -- 45.4% NULL. Because the Python alignment layer zero-fills a missing
     -- column, the model was reading "no data" as "no surprise" for half the
     -- universe, which is not a small distinction: it is the difference between
     -- an informative zero and an absent measurement.
     --
     -- v2 replaces the five with two columns that say different things:
     --   feat_eps_signal   -- the average of whichever legs EXIST (NULL if none)
     --   feat_eps_coverage -- what share of the five legs existed, in [0, 1]
     -- so the model can learn an interaction rather than being handed a
     -- counterfeit zero. The raw five are still emitted below for EDA and for
     -- the analytics export; they are simply no longer the drift inputs.
     -- ---------------------------------------------------------------------
     eps AS (SELECT b.isin,
                    (SELECT avg(v)
                     FROM unnest(ARRAY [b.feat_net_eps_drift, b.feat_last_q_surprise,
                         b.feat_last_y_surprise, b.feat_eps_beat_rate,
                         b.feat_eps_beat_rate_annual]) AS v
                     WHERE v IS NOT NULL)                           AS eps_signal,
                    (SELECT count(v)::DOUBLE PRECISION / 5.0
                     FROM unnest(ARRAY [b.feat_net_eps_drift, b.feat_last_q_surprise,
                         b.feat_last_y_surprise, b.feat_eps_beat_rate,
                         b.feat_eps_beat_rate_annual]) AS v
                     WHERE v IS NOT NULL)                           AS eps_coverage
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
       -- CALENDAR OFFSETS (pymc_role = 'constant_data')
       -- ===================================================================
       -- Nominal day offsets of each trail column. Emitted as columns rather
       -- than hard-coded in Python so the OU kernel's x-axis has an SSOT: if a
       -- lookback's meaning ever changes, the model's gaps change with it
       -- instead of silently disagreeing.
       --
       -- These are NOMINAL, not the exact as-of dates of each vendor snapshot,
       -- which the source does not carry. The approximation is second-order:
       -- the fitted kernel has RMSE 0.026 against nominal offsets, so any
       -- date-alignment error is already inside the residual.
       0::INT                                                         AS trail_days_now,
       7::INT                                                         AS trail_days_1w,
       30::INT                                                        AS trail_days_1m,
       91::INT                                                        AS trail_days_3m,
       182::INT                                                       AS trail_days_6m,
       365::INT                                                       AS trail_days_1y,

       -- ===================================================================
       -- PER-LOOKBACK ANALYST COVERAGE (pymc_role = 'constant_data')
       -- ===================================================================
       -- v1 applied ONE precision weight to every time step. A consensus built
       -- from 4 analysts a year ago and one built from 30 today are not equally
       -- precise measurements of the same latent, and the model had no way to
       -- say so. These feed the per-cell measurement scale in v2.
       b.price_target_num_1w_ago                                      AS n_analysts_1w,
       b.price_target_num_1m_ago                                      AS n_analysts_1m,
       b.price_target_num_3m_ago                                      AS n_analysts_3m,
       b.price_target_num_6m_ago                                      AS n_analysts_6m,
       b.price_target_num_1y_ago                                      AS n_analysts_1y,

       -- ===================================================================
       -- CONSOLIDATED EPS BLOCK (pymc_role = 'mutable_predictor')
       -- ===================================================================
       e.eps_signal                                                   AS feat_eps_signal,
       COALESCE(e.eps_coverage, 0.0)                                  AS feat_eps_coverage,

       -- ===================================================================
       -- PROVENANCE (pymc_role = 'constant_data')
       -- ===================================================================
       -- Stamps the refresh moment on every row. v1's days_* horizons are
       -- CURRENT_DATE-relative, so two refreshes of "the same" MV are different
       -- datasets; this makes that visible in the exported panel frame instead
       -- of being something you have to remember.
       now()                                                          AS built_at

FROM base b
         JOIN uplift u ON u.isin = b.isin
         JOIN eps e ON e.isin = b.isin
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
COMMENT ON COLUMN pml.mv_pymc_kalman_pt_v2.trail_days_1y IS
    'Nominal calendar offset in days of the 1y trail column. Feeds the OU kernel '
        'exp(-gap/ell); emitted so the gaps have one source of truth.';
COMMENT ON COLUMN pml.mv_pymc_kalman_pt_v2.feat_eps_signal IS
    'Mean of whichever of the five EPS surprise/drift legs is non-NULL, or NULL '
        'when none is. Replaces five separately zero-filled columns.';
COMMENT ON COLUMN pml.mv_pymc_kalman_pt_v2.feat_eps_coverage IS
    'Share of the five EPS legs that were non-NULL, in [0, 1]. Lets the model '
        'distinguish an informative zero from an absent measurement.';
COMMENT ON COLUMN pml.mv_pymc_kalman_pt_v2.built_at IS
    'Refresh timestamp. The parent MV computes its days_* horizons against '
        'CURRENT_DATE, so this is what tells two refreshes apart.';


-- ===========================================================================
-- Refresh procedure -- enforces the parent-then-child order
-- ===========================================================================
CREATE OR REPLACE PROCEDURE pml.refresh_kalman_pt_v2(
    use_concurrently BOOLEAN DEFAULT TRUE,
    refresh_parent BOOLEAN DEFAULT TRUE
)
    LANGUAGE plpgsql
AS
$$
BEGIN
    -- The child selects from the parent, so a child-only refresh silently
    -- rebuilds v2 against a stale v1 -- the mixed-vintage failure mode the
    -- analytics export already learned the hard way. Default to doing both.
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


-- ===========================================================================
-- pml.kalman_pt_v2_asof(p_asof) -- the point-in-time form
-- ===========================================================================
-- An MV cannot take a parameter, which is why mv_pymc_kalman_pt's days_*
-- horizons are pinned to CURRENT_DATE and why the whole feature matrix is
-- unusable for a backtest. This function is the escape hatch: it recomputes the
-- date-relative horizons against an arbitrary as-of date, leaving every
-- price/target-derived feature untouched (those are already point-in-time by
-- construction).
--
-- STABLE, not IMMUTABLE: it reads the MV.
--
-- SCOPE, stated plainly: this fixes the HORIZON columns only. The underlying
-- price and target trails are still whatever the last vendor load contained, so
-- this supports "what would the model have said about the calendar on date X",
-- not a full historical replay. A real replay needs versioned vendor snapshots,
-- which pml.staging does not retain.
CREATE OR REPLACE FUNCTION pml.kalman_pt_v2_asof(p_asof DATE DEFAULT CURRENT_DATE)
    RETURNS TABLE
            (
                isin                        TEXT,
                days_to_next_earnings       INT,
                days_since_last_report      INT,
                days_to_next_fy_end         INT,
                days_to_next_report         INT,
                days_to_expected_report     INT,
                days_since_fy_end           INT,
                asof_date                   DATE
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
        'a full historical replay.';
