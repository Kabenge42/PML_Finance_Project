CREATE MATERIALIZED VIEW mv_pymc_earnings_beat AS
-- missing source code
;

ALTER MATERIALIZED VIEW mv_pymc_earnings_beat OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_pymc_earnings_beat_isin ON mv_pymc_earnings_beat (isin);