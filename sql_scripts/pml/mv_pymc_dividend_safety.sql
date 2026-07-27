CREATE MATERIALIZED VIEW mv_pymc_dividend_safety AS
-- missing source code
;

ALTER MATERIALIZED VIEW mv_pymc_dividend_safety OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_pymc_dividend_safety_isin ON mv_pymc_dividend_safety (isin);