create materialized view mv_pymc_price_target
as
-- missing source code
;

alter materialized view mv_pymc_price_target owner to postgres
;

create unique index idx_mv_pymc_price_target_isin
	on mv_pymc_price_target (isin)
;