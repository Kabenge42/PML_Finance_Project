-- ===========================================================================
-- analytics.panel_vintage_v2
-- ===========================================================================
-- The point-in-time store the Kalman v2 workflow has never had.
--
-- WHY IT EXISTS
-- -------------
-- Every gate in the v2 workflow scores the model against the ANALYST TRAIL:
-- ppc_coverage, ppc_decay, ppc_t_spread, mean_calibration and shrinkage_slope
-- all ask "does the model reproduce, or sensibly revise, the price-target
-- series it was fitted to?". None of them asks "was the model RIGHT?".
--
-- That is why run 49e84d7e9d59 could reproduce analyst consensus at Spearman
-- 0.999995 and still clear 19 of 21 gates. A pass-through is, by construction,
-- a perfect model of its own input.
--
-- Scoring against outcomes needs vintages, and vintages are exactly what the
-- feature surface cannot reconstruct after the fact:
--
--   * pml.mv_pymc_kalman_pt computes seven days_* horizons against
--     CURRENT_DATE, so refreshing on a different day silently shifts every one.
--     pml.kalman_pt_v2_asof(p_asof) recovers six of those for an arbitrary date.
--   * The price and price-target TRAILS are not versioned at all. There is no
--     as-of function for them and no way to recover them later, which is the
--     gap this table closes.
--
-- So: capture the trail as it stood, keyed by the date it stood that way, and
-- let scripts/score_panel_vintages.py join two captures once they are far
-- enough apart to have an outcome between them.
--
-- APPEND-ONLY. Unlike the seven pipeline tables, which are DROP-and-RECREATE on
-- every export, a vintage that is overwritten is a vintage that never existed.
-- scripts/capture_panel_vintage.py refuses to overwrite an existing asof_date
-- unless --replace is passed.
--
-- UNITS: prices and price targets are in the security's own currency; every
-- return-like column is a RAW DECIMAL (0.25 = +25%), per the 0.9.9.7 convention.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS analytics.panel_vintage_v2
(
    -- ---- vintage key -------------------------------------------------------
    "asof_date"               DATE             NOT NULL,
    "isin"                    TEXT             NOT NULL,
    "run_id"                  TEXT,
    "captured_at"             TIMESTAMPTZ      NOT NULL DEFAULT now(),

    -- ---- identity ----------------------------------------------------------
    "ticker"                  TEXT,
    "name"                    TEXT,
    "sector"                  TEXT,
    "industry"                TEXT,
    "trading_region"          TEXT,
    "country"                 TEXT,
    "style_class"             TEXT,
    "size_class"              TEXT,
    "market_cap"              DOUBLE PRECISION,

    -- ---- the anchor, as it stood ------------------------------------------
    "last_price"              DOUBLE PRECISION,
    "observed_pt"             DOUBLE PRECISION,
    "n_analysts"              DOUBLE PRECISION,

    -- ---- the trail, as it stood -------------------------------------------
    "price_1w_ago"            DOUBLE PRECISION,
    "price_1m_ago"            DOUBLE PRECISION,
    "price_3m_ago"            DOUBLE PRECISION,
    "price_6m_ago"            DOUBLE PRECISION,
    "price_1y_ago"            DOUBLE PRECISION,
    "price_target_1w_ago"     DOUBLE PRECISION,
    "price_target_1m_ago"     DOUBLE PRECISION,
    "price_target_3m_ago"     DOUBLE PRECISION,
    "price_target_6m_ago"     DOUBLE PRECISION,
    "price_target_1y_ago"     DOUBLE PRECISION,
    "n_analysts_1w"           DOUBLE PRECISION,
    "n_analysts_1m"           DOUBLE PRECISION,
    "n_analysts_3m"           DOUBLE PRECISION,
    "n_analysts_6m"           DOUBLE PRECISION,
    "n_analysts_1y"           DOUBLE PRECISION,

    -- ---- what the model said at that date ---------------------------------
    -- Carried so a vintage is self-sufficient for scoring: the run that
    -- produced them is DROP-and-RECREATEd out of existence on the next export.
    "implied_upside"          DOUBLE PRECISION,
    "expected_return_kalman"  DOUBLE PRECISION,
    "expected_upside_sd"      DOUBLE PRECISION,
    "shrink_gain"             DOUBLE PRECISION,
    "er_mean"                 DOUBLE PRECISION,
    "er_sd"                   DOUBLE PRECISION,
    "er_p05"                  DOUBLE PRECISION,
    "er_p50"                  DOUBLE PRECISION,
    "er_p95"                  DOUBLE PRECISION,
    "mc_prob_pos"             DOUBLE PRECISION,
    "p_upside_pos_cond"       DOUBLE PRECISION,
    "cvar_5pct_kalman"        DOUBLE PRECISION,
    "out_of_support"          BOOLEAN,

    CONSTRAINT pk_panel_vintage_v2 PRIMARY KEY ("asof_date", "isin")
);

COMMENT ON TABLE analytics.panel_vintage_v2 IS
    'Append-only point-in-time capture of the Kalman v2 panel and its decision '
    'outputs. Two captures separated in time are what scripts/score_panel_vintages.py '
    'needs to score the model against realised returns rather than against the '
    'analyst trail it was fitted to.';

COMMENT ON COLUMN analytics.panel_vintage_v2."asof_date" IS
    'The date this row describes. Supplied by the capture script, defaulting to '
    'CURRENT_DATE -- NOT derived from the data, because the trail columns carry '
    'no date of their own, which is the whole reason this table exists.';

COMMENT ON COLUMN analytics.panel_vintage_v2."expected_return_kalman" IS
    'The model expected upside as of asof_date. Raw decimal. Scored later '
    'against (last_price at a subsequent vintage / last_price here - 1).';

COMMENT ON COLUMN analytics.panel_vintage_v2."shrink_gain" IS
    'Weight the forecast-error update put on the name own smoothed observation. '
    'Captured because calibrating forecast_error_multiplier against realised '
    'outcomes is the point of the exercise -- a vintage without it cannot tell '
    'you whether the shrinkage helped.';

CREATE INDEX IF NOT EXISTS idx_panel_vintage_v2_asof
    ON analytics.panel_vintage_v2 ("asof_date");
CREATE INDEX IF NOT EXISTS idx_panel_vintage_v2_isin
    ON analytics.panel_vintage_v2 ("isin");
