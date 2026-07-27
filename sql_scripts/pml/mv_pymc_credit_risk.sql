CREATE MATERIALIZED VIEW mv_pymc_credit_risk AS
-- missing source code
;

ALTER MATERIALIZED VIEW mv_pymc_credit_risk OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_pymc_credit_risk_isin ON mv_pymc_credit_risk (isin);