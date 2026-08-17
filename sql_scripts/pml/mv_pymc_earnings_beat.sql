create materialized view mv_pymc_earnings_beat
as
-- missing source code
;

alter materialized view mv_pymc_earnings_beat owner to postgres
;

create unique index idx_mv_pymc_earnings_beat_isin
	on mv_pymc_earnings_beat (isin)
;