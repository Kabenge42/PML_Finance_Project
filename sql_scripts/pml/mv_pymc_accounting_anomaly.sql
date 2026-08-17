create materialized view mv_pymc_accounting_anomaly
as
-- missing source code
;

alter materialized view mv_pymc_accounting_anomaly owner to postgres
;

create unique index idx_mv_pymc_accounting_anomaly_isin
	on mv_pymc_accounting_anomaly (isin)
;