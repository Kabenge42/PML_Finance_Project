CREATE MATERIALIZED VIEW mv_pymc_price_target AS
-- missing source code
;

ALTER MATERIALIZED VIEW mv_pymc_price_target OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_pymc_price_target_isin ON mv_pymc_price_target (isin);