-- =============================================================================
-- Helper Functions for Feature Registry
-- Must be executed BEFORE feature_registry.sql
-- =============================================================================

-- Safe division helper (avoids division by zero)
CREATE OR REPLACE FUNCTION safe_divide(
    numerator   NUMERIC,
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
CREATE OR REPLACE FUNCTION pct_change(current_val NUMERIC, previous_val NUMERIC)
    RETURNS NUMERIC
    IMMUTABLE
    PARALLEL SAFE
AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) * 100 AS result;
$$ LANGUAGE SQL;

-- Momentum/change ratio helper (without percentage multiplier)
CREATE OR REPLACE FUNCTION calc_change_ratio(current_val NUMERIC, previous_val NUMERIC)
    RETURNS NUMERIC
    IMMUTABLE
    PARALLEL SAFE
AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) AS result;
$$ LANGUAGE SQL;

-- Score clamping helper (constrains value between 0 and 100)
CREATE OR REPLACE FUNCTION clamp_score(val NUMERIC, min_val NUMERIC DEFAULT 0, max_val NUMERIC DEFAULT 100)
    RETURNS NUMERIC
    IMMUTABLE
    PARALLEL SAFE
AS
$$
SELECT GREATEST(min_val, LEAST(max_val, val)) AS result;
$$ LANGUAGE SQL;

-- EMA crossover signal helper
CREATE OR REPLACE FUNCTION ema_crossover_signal(fast_ema NUMERIC, slow_ema NUMERIC)
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

-- Verify functions were created
DO
$$
    BEGIN
        RAISE NOTICE 'Helper functions created successfully:';
        RAISE NOTICE '  - safe_divide(numeric, numeric)';
        RAISE NOTICE '  - pct_change(numeric, numeric)';
        RAISE NOTICE '  - calc_change_ratio(numeric, numeric)';
        RAISE NOTICE '  - clamp_score(numeric, numeric, numeric)';
        RAISE NOTICE '  - ema_crossover_signal(numeric, numeric)';
    END
$$;
