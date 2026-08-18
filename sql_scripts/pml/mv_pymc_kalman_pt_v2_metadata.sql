-- ===========================================================================
-- Catalogue registration for pml.mv_pymc_kalman_pt_v2
-- ===========================================================================
--
-- RUN THIS WITH mv_pymc_kalman_pt_v2.sql, NOT AFTER IT AND NOT INSTEAD OF IT.
--
-- The resolution chain Python depends on is:
--
--     pml_df_metadata            (global pymc_role, model_targets[])
--       -> pml_df_feature_alias  (per-model feature_alias AND pymc_role override)
--       -> vw_pymc_feature_catalogue     COALESCE(fa.pymc_role, md.pymc_role)
--       -> vw_pymc_feature_aliases       the arrays the resolvers consume
--
-- and every link fails SILENTLY. An MV column with no catalogue row does not
-- raise; it simply vanishes from the model's feature list and is later
-- zero-filled by the alignment layer. That is why pml.assert_pymc_catalogue_coverage()
-- exists and why this file is mandatory rather than tidy-up.
--
-- FOUR THINGS HAVE TO HAPPEN, IN THIS ORDER
-- -----------------------------------------
--   1. Widen the model_targets CHECK. It is a hard allow-list of seven values;
--      inserting 'kalman_pt_v2' without this step fails the constraint. This is
--      the step that is easy to miss because nothing else in the pipeline
--      references it.
--   2. Register the new columns in pml_df_metadata.
--   3. Teach the coverage check about the new MV, or it reports every v2 column
--      as MISSING_FROM_CATALOGUE and every v2 alias as PHANTOM.
--   4. Verify. The last statement in this file is the check; it should return
--      zero rows.
--
-- WHY A SEPARATE MODEL TARGET
-- ---------------------------
-- v2 gets 'kalman_pt_v2' rather than extending 'kalman_pt' because the two
-- models want different feature LISTS from overlapping columns: v2 drops the
-- five raw EPS legs from its drift matrix in favour of feat_eps_signal /
-- feat_eps_coverage, and adds the response trail as 'observed'. Sharing one
-- target would force one of them to carry the other's exclusions in Python,
-- which is exactly the coupling the catalogue exists to avoid. It also means v1
-- keeps running untouched while v2 is evaluated -- the two can be compared on
-- the same database.
-- ===========================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. Widen the model_targets allow-list
-- ---------------------------------------------------------------------------
ALTER TABLE pml.pml_df_metadata
    DROP CONSTRAINT IF EXISTS ck_pml_df_metadata_model_targets;

ALTER TABLE pml.pml_df_metadata
    ADD CONSTRAINT ck_pml_df_metadata_model_targets
        CHECK (model_targets <@ ARRAY [
            'earnings_beat', 'price_target', 'kalman_pt', 'kalman_pt_v2',
            'dcf_pt', 'dividend_safety', 'credit_risk', 'accounting_anomaly'
            ]::TEXT[]);

-- The alias table carries the same vocabulary; widen it too if it is constrained.
ALTER TABLE pml.pml_df_feature_alias
    DROP CONSTRAINT IF EXISTS ck_pml_df_feature_alias_model_target;

ALTER TABLE pml.pml_df_feature_alias
    ADD CONSTRAINT ck_pml_df_feature_alias_model_target
        CHECK (model_target = ANY (ARRAY [
            'earnings_beat', 'price_target', 'kalman_pt', 'kalman_pt_v2',
            'dcf_pt', 'dividend_safety', 'credit_risk', 'accounting_anomaly'
            ]::TEXT[]));


-- ---------------------------------------------------------------------------
-- 2a. v2 inherits every kalman_pt column it still uses
-- ---------------------------------------------------------------------------
-- Rather than re-listing ~140 rows, tag the existing kalman_pt registrations
-- with the v2 target as well. The columns v2 does NOT want in its drift matrix
-- are excluded in Python (DRIFT_EXCLUSIONS in pymc_kalman_filter_pt_v2.py), NOT
-- by withholding a catalogue row -- withholding the row would make the column
-- invisible to the catalogue while the MV still emits it, which is the
-- MISSING_FROM_CATALOGUE failure, the dangerous one.
UPDATE pml.pml_df_metadata
SET model_targets = array_append(model_targets, 'kalman_pt_v2'),
    updated_at    = now()
WHERE 'kalman_pt' = ANY (model_targets)
  AND NOT ('kalman_pt_v2' = ANY (model_targets));

INSERT INTO pml.pml_df_feature_alias (column_name, model_target, feature_alias, pymc_role)
SELECT fa.column_name, 'kalman_pt_v2', fa.feature_alias, fa.pymc_role
FROM pml.pml_df_feature_alias fa
WHERE fa.model_target = 'kalman_pt'
ON CONFLICT DO NOTHING;


-- ---------------------------------------------------------------------------
-- 2b. Register the genuinely new v2 columns
-- ---------------------------------------------------------------------------
-- data_type drives coerce_by_data_type() on the Python side, so it is a
-- behavioural field, not documentation: 'double precision' is passed through,
-- and a wrong value silently rescales a feature.
INSERT INTO pml.pml_df_metadata
    (column_name, category, feature_role, feature_alias, description,
     data_type, pymc_role, model_targets, updated_at)
VALUES
    -- ---- the response trail (pymc_role = observed) -------------------------
    ('feat_log_uplift_now', 'analyst_targets', 'target', 'feat_log_uplift_now',
     'ln(price_target / last_price). Snapshot response and OU grid anchor.',
     'double precision', 'observed', ARRAY ['kalman_pt_v2'], now()),
    ('feat_log_uplift_1w', 'analyst_targets', 'historical', 'feat_log_uplift_1w',
     'ln(price_target_1w_ago / price_1w_ago). Trail response at 7 days.',
     'double precision', 'observed', ARRAY ['kalman_pt_v2'], now()),
    ('feat_log_uplift_1m', 'analyst_targets', 'historical', 'feat_log_uplift_1m',
     'ln(price_target_1m_ago / price_1m_ago). Trail response at 30 days.',
     'double precision', 'observed', ARRAY ['kalman_pt_v2'], now()),
    ('feat_log_uplift_3m', 'analyst_targets', 'historical', 'feat_log_uplift_3m',
     'ln(price_target_3m_ago / price_3m_ago). Trail response at 91 days.',
     'double precision', 'observed', ARRAY ['kalman_pt_v2'], now()),
    ('feat_log_uplift_6m', 'analyst_targets', 'historical', 'feat_log_uplift_6m',
     'ln(price_target_6m_ago / price_6m_ago). Trail response at 182 days.',
     'double precision', 'observed', ARRAY ['kalman_pt_v2'], now()),
    ('feat_log_uplift_1y', 'analyst_targets', 'historical', 'feat_log_uplift_1y',
     'ln(price_target_1y_ago / price_1y_ago). Trail response at 365 days.',
     'double precision', 'observed', ARRAY ['kalman_pt_v2'], now()),

    -- ---- trail support ------------------------------------------------------
    ('n_trail_obs', 'analyst_targets', 'count', 'n_trail_obs',
     'Count of non-NULL trail cells (1..6). Per-name T contributing to the likelihood.',
     'integer', 'constant_data', ARRAY ['kalman_pt_v2'], now()),

    -- ---- calendar offsets (the OU kernel x-axis) ---------------------------
    ('trail_days_now', 'fiscal_calendar', 'metadata', 'trail_days_now',
     'Nominal calendar offset in days of the snapshot column (0).',
     'integer', 'constant_data', ARRAY ['kalman_pt_v2'], now()),
    ('trail_days_1w', 'fiscal_calendar', 'metadata', 'trail_days_1w',
     'Nominal calendar offset in days of the 1w trail column (7).',
     'integer', 'constant_data', ARRAY ['kalman_pt_v2'], now()),
    ('trail_days_1m', 'fiscal_calendar', 'metadata', 'trail_days_1m',
     'Nominal calendar offset in days of the 1m trail column (30).',
     'integer', 'constant_data', ARRAY ['kalman_pt_v2'], now()),
    ('trail_days_3m', 'fiscal_calendar', 'metadata', 'trail_days_3m',
     'Nominal calendar offset in days of the 3m trail column (91).',
     'integer', 'constant_data', ARRAY ['kalman_pt_v2'], now()),
    ('trail_days_6m', 'fiscal_calendar', 'metadata', 'trail_days_6m',
     'Nominal calendar offset in days of the 6m trail column (182).',
     'integer', 'constant_data', ARRAY ['kalman_pt_v2'], now()),
    ('trail_days_1y', 'fiscal_calendar', 'metadata', 'trail_days_1y',
     'Nominal calendar offset in days of the 1y trail column (365).',
     'integer', 'constant_data', ARRAY ['kalman_pt_v2'], now()),

    -- ---- per-lookback analyst coverage --------------------------------------
    ('n_analysts_1w', 'analyst_targets', 'count', 'n_analysts_1w',
     'Analyst count behind the 1w consensus. Per-cell measurement precision.',
     'double precision', 'constant_data', ARRAY ['kalman_pt_v2'], now()),
    ('n_analysts_1m', 'analyst_targets', 'count', 'n_analysts_1m',
     'Analyst count behind the 1m consensus.',
     'double precision', 'constant_data', ARRAY ['kalman_pt_v2'], now()),
    ('n_analysts_3m', 'analyst_targets', 'count', 'n_analysts_3m',
     'Analyst count behind the 3m consensus.',
     'double precision', 'constant_data', ARRAY ['kalman_pt_v2'], now()),
    ('n_analysts_6m', 'analyst_targets', 'count', 'n_analysts_6m',
     'Analyst count behind the 6m consensus.',
     'double precision', 'constant_data', ARRAY ['kalman_pt_v2'], now()),
    ('n_analysts_1y', 'analyst_targets', 'count', 'n_analysts_1y',
     'Analyst count behind the 1y consensus.',
     'double precision', 'constant_data', ARRAY ['kalman_pt_v2'], now()),

    -- ---- consolidated EPS block ---------------------------------------------
    ('feat_eps_signal', 'eps', 'predictor', 'feat_eps_signal',
     'Mean of the non-NULL EPS surprise/drift legs, or NULL when none exists. '
         'Replaces five separately zero-filled columns.',
     'double precision', 'mutable_predictor', ARRAY ['kalman_pt_v2'], now()),
    ('feat_eps_coverage', 'eps', 'predictor', 'feat_eps_coverage',
     'Share of the five EPS legs that were non-NULL, in [0,1]. Distinguishes an '
         'informative zero from an absent measurement.',
     'double precision', 'mutable_predictor', ARRAY ['kalman_pt_v2'], now()),

    -- ---- provenance -----------------------------------------------------------
    ('built_at', 'identifier', 'metadata', 'built_at',
     'MV refresh timestamp. The parent computes days_* against CURRENT_DATE, so '
         'this is what tells two refreshes apart.',
     'timestamp with time zone', 'constant_data', ARRAY ['kalman_pt_v2'], now())

ON CONFLICT (column_name) DO UPDATE
    SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(
                                          pml.pml_df_metadata.model_targets ||
                                          EXCLUDED.model_targets))),
        description   = EXCLUDED.description,
        pymc_role     = EXCLUDED.pymc_role,
        data_type     = EXCLUDED.data_type,
        updated_at    = now();


-- ---------------------------------------------------------------------------
-- 2c. Per-model role overrides
-- ---------------------------------------------------------------------------
-- The trail columns are 'observed' for v2 but must never be offered as drift
-- predictors -- feeding the response back in as a feature is the leakage that
-- would make every diagnostic look excellent and every forecast worthless.
-- The global role above already says 'observed'; these rows make it explicit at
-- the model level so a future global change cannot quietly re-admit them.
INSERT INTO pml.pml_df_feature_alias (column_name, model_target, feature_alias, pymc_role)
VALUES ('feat_log_uplift_now', 'kalman_pt_v2', 'feat_log_uplift_now', 'observed'),
       ('feat_log_uplift_1w', 'kalman_pt_v2', 'feat_log_uplift_1w', 'observed'),
       ('feat_log_uplift_1m', 'kalman_pt_v2', 'feat_log_uplift_1m', 'observed'),
       ('feat_log_uplift_3m', 'kalman_pt_v2', 'feat_log_uplift_3m', 'observed'),
       ('feat_log_uplift_6m', 'kalman_pt_v2', 'feat_log_uplift_6m', 'observed'),
       ('feat_log_uplift_1y', 'kalman_pt_v2', 'feat_log_uplift_1y', 'observed')
ON CONFLICT (column_name, model_target) DO UPDATE
    SET feature_alias = EXCLUDED.feature_alias,
        pymc_role     = EXCLUDED.pymc_role;


-- ---------------------------------------------------------------------------
-- 3. Teach the coverage check about the v2 MV
-- ---------------------------------------------------------------------------
-- The mv_map CTE in vw_pymc_catalogue_coverage_check is a hard-coded VALUES
-- list. A new MV that is not in it is invisible to the check, so its columns are
-- never verified -- which is worse than failing, because it looks like passing.
--
-- DROP + CREATE, not CREATE OR REPLACE: replacing a view cannot change its
-- column list, and the deployed definition exposes four columns
-- (model_target, feat_name, catalogue_rows, status). The body below reproduces
-- that contract EXACTLY -- same columns, same order, same three statuses, same
-- UNION ALL arm for phantoms -- and differs from the deployed view in one line,
-- the v2 entry in mv_map. Anything else here is a regression in a check the
-- whole catalogue depends on.
--
-- Plain DROP, never DROP ... CASCADE: pml.assert_pymc_catalogue_coverage()
-- resolves this view by name at execution time (plpgsql late binding), so it
-- survives the drop, whereas CASCADE would silently delete it.
DROP VIEW IF EXISTS pml.vw_pymc_catalogue_coverage_check;

CREATE VIEW pml.vw_pymc_catalogue_coverage_check AS
WITH mv_map(mv_name, model_target) AS (VALUES ('mv_pymc_earnings_beat', 'earnings_beat'),
                                              ('mv_pymc_price_target', 'price_target'),
                                              ('mv_pymc_kalman_pt', 'kalman_pt'),
                                              ('mv_pymc_kalman_pt_v2', 'kalman_pt_v2'),
                                              ('mv_pymc_dcf_pt', 'dcf_pt'),
                                              ('mv_pymc_dividend_safety', 'dividend_safety'),
                                              ('mv_pymc_credit_risk', 'credit_risk'),
                                              ('mv_pymc_accounting_anomaly', 'accounting_anomaly')),
     -- pg_class/pg_attribute, NOT information_schema.columns: the SQL standard
     -- information_schema does not expose materialized-view columns, which once
     -- silently emptied mv_cols and flagged every alias as PHANTOM.
     mv_cols AS (SELECT mm.model_target, a.attname::TEXT AS feat_name
                 FROM mv_map mm
                          JOIN pg_catalog.pg_class cl
                               ON cl.relname = mm.mv_name AND cl.relkind = 'm'
                          JOIN pg_catalog.pg_namespace ns
                               ON ns.oid = cl.relnamespace AND ns.nspname = 'pml'
                          JOIN pg_catalog.pg_attribute a
                               ON a.attrelid = cl.oid AND a.attnum > 0 AND NOT a.attisdropped
                 WHERE a.attname LIKE 'feat\_%'
                    OR a.attname LIKE 'observed\_%'
                    OR a.attname LIKE 'n\_%'),
     cat AS (SELECT model_target, feature_alias, count(*) AS n_rows
             FROM pml.vw_pymc_feature_catalogue
             WHERE feature_alias LIKE 'feat\_%'
                OR feature_alias LIKE 'observed\_%'
                OR feature_alias LIKE 'n\_%'
             GROUP BY 1, 2)
SELECT mc.model_target,
       mc.feat_name,
       COALESCE(cat.n_rows, 0::BIGINT) AS catalogue_rows,
       CASE
           WHEN cat.n_rows IS NULL THEN 'MISSING_FROM_CATALOGUE'
           WHEN cat.n_rows > 1 THEN 'DUPLICATE_CATALOGUE_ALIAS'
           ELSE 'OK'
           END                         AS status
FROM mv_cols mc
         LEFT JOIN cat
                   ON cat.model_target = mc.model_target AND cat.feature_alias = mc.feat_name
UNION ALL
SELECT c.model_target,
       c.feature_alias           AS feat_name,
       0                         AS catalogue_rows,
       'PHANTOM_CATALOGUE_ALIAS' AS status
FROM cat c
         LEFT JOIN mv_cols mc
                   ON mc.model_target = c.model_target AND mc.feat_name = c.feature_alias
WHERE mc.feat_name IS NULL;

COMMIT;


-- ===========================================================================
-- 4. VERIFY -- this must return zero rows
-- ===========================================================================
-- Run it now, and again after every refresh. A non-empty result means a v2
-- column is silently zero-filled (MISSING_FROM_CATALOGUE), registered but not
-- emitted (PHANTOM_CATALOGUE_ALIAS), or claimed twice so alias resolution is
-- order-dependent (DUPLICATE_CATALOGUE_ALIAS).
SELECT model_target,
       status,
       count(*)                                       AS n,
       string_agg(feat_name, ', ' ORDER BY feat_name) AS columns
FROM pml.vw_pymc_catalogue_coverage_check
WHERE status <> 'OK'
  AND model_target = 'kalman_pt_v2'
GROUP BY 1, 2
ORDER BY 1, 2;
