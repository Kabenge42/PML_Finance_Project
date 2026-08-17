create materialized view public.mv_equities
as
-- missing source code
;

comment on materialized view public.mv_equities is 'Aliased snapshot of the equities table. Column names sourced from equities_schema_metadata.column_alias. Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY mv_equities;'
;

alter materialized view public.mv_equities owner to postgres
;

create unique index idx_mv_equities_isin
	on public.mv_equities (isin)
;

create index idx_mv_equities_ticker
	on public.mv_equities (ticker)
;

create index idx_mv_equities_geography
	on public.mv_equities (region, country, exchange)
;

create index idx_mv_equities_class
	on public.mv_equities (sector, industry, size_class, style_class)
;

create index idx_mv_equities_market_cap
	on public.mv_equities (market_cap desc)
;

create index idx_mv_equities_fiscal
	on public.mv_equities (fiscal_year, fiscal_quarter, income_statement_report_date)
;