CREATE MATERIALIZED VIEW mv_pymc_dcf_pt AS
-- missing source code
;

ALTER MATERIALIZED VIEW mv_pymc_dcf_pt OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_pymc_dcf_pt_isin ON mv_pymc_dcf_pt (isin);