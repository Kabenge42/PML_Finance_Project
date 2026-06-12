CREATE MATERIALIZED VIEW public.mv_equities AS
-- missing source code
;

COMMENT ON MATERIALIZED VIEW public.mv_equities IS 'Aliased snapshot of the equities table. Column names sourced from equities_schema_metadata.column_alias. Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY mv_equities;';

ALTER MATERIALIZED VIEW public.mv_equities OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_equities_isin ON public.mv_equities (isin);

CREATE INDEX idx_mv_equities_ticker ON public.mv_equities (ticker);

CREATE INDEX idx_mv_equities_geography ON public.mv_equities (region, country, exchange);

CREATE INDEX idx_mv_equities_class ON public.mv_equities (sector, industry, size_class, style_class);

CREATE INDEX idx_mv_equities_market_cap ON public.mv_equities (market_cap DESC);

CREATE INDEX idx_mv_equities_fiscal ON public.mv_equities (fiscal_year, fiscal_quarter, income_statement_report_date);