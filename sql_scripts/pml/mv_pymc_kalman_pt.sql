create materialized view mv_pymc_kalman_pt
as
-- missing source code
;

alter materialized view mv_pymc_kalman_pt owner to postgres
;

create unique index idx_mv_pymc_kalman_pt_isin
	on mv_pymc_kalman_pt (isin)
;