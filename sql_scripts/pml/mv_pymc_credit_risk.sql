create materialized view mv_pymc_credit_risk
as
-- missing source code
;

alter materialized view mv_pymc_credit_risk owner to postgres
;

create unique index idx_mv_pymc_credit_risk_isin
	on mv_pymc_credit_risk (isin)
;