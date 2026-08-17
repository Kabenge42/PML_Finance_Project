create materialized view mv_pymc_dividend_safety
as
-- missing source code
;

alter materialized view mv_pymc_dividend_safety owner to postgres
;

create unique index idx_mv_pymc_dividend_safety_isin
	on mv_pymc_dividend_safety (isin)
;