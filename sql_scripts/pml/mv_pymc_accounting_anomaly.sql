CREATE MATERIALIZED VIEW mv_pymc_accounting_anomaly AS
-- missing source code
;

ALTER MATERIALIZED VIEW mv_pymc_accounting_anomaly OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_pymc_accounting_anomaly_isin ON mv_pymc_accounting_anomaly (isin);