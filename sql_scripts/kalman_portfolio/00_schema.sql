-- ===========================================================================
--  kalman_portfolio -- the decision layer's own schema
-- ===========================================================================
--
--  Applied by `scripts/apply_kalman_portfolio_schema.py`.
--
--  WHY A SECOND SCHEMA. `analytics` is written by the FIT with
--  `if_exists='replace'`: every table there is dropped and recreated once per
--  export. `kalman_portfolio.py` replays that one fit many times -- three ranking
--  arms, two prior sweeps, a mean-model contrast -- and each replay is a distinct
--  observation about the same posterior. Writing them into DROP-and-RECREATE
--  tables would mean each replay destroyed the last, so the sweep whose whole
--  purpose is comparison could only ever be compared by re-deriving it.
--
--  APPEND-ONLY, keyed by run_id. Same policy as `analytics.panel_vintage_v2`,
--  and for a related reason: the thing being accumulated cannot be recovered
--  later. A replay is cheap to re-run only while the handoff it replayed still
--  exists and still matches; the moment a new fit lands, the old replay is not
--  reproducible at all.
--
--  `vw_latest_*` views resolve the most recent run per table, so a consumer that
--  wants "the current book" reads a view and a consumer that wants the history
--  reads the table. The tables themselves are created by the export via
--  `to_sql(..., if_exists='append')` on first write; this file creates the
--  SCHEMA and the grants, which nothing else in this repository does -- there is
--  no `CREATE SCHEMA` anywhere, and `analytics` has always been assumed to
--  pre-exist.
--
--  UNITS. Raw decimals throughout (0.25 = +25%), matching the analytics schema.
--  Percent scaling happens at visualization boundaries only.

CREATE SCHEMA IF NOT EXISTS kalman_portfolio;

COMMENT ON SCHEMA kalman_portfolio IS
    'Decision-layer exports from kalman_portfolio.py. APPEND-ONLY, keyed by '
    'run_id: one replay is one observation about a fit, and a replay is not '
    'reproducible once its handoff is superseded. Contrast the analytics schema, '
    'which the fit DROPs and RECREATEs each export. Raw decimal returns.';

-- The owner the generated DDL assumes, matching the analytics convention.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = current_setting('pml.analytics_owner', true)) THEN
        EXECUTE format('ALTER SCHEMA kalman_portfolio OWNER TO %I',
                       current_setting('pml.analytics_owner', true));
    END IF;
END
$$;
