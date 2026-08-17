create function public.calc_temporal_features(p_isin text default NULL::text)
	returns table("isin" text, "fiscal_quarter" integer, "fiscal_month" integer, "fiscal_year" integer, "days_to_earnings" integer, "earnings_report_recency" integer, "reporting_lag" numeric, "fiscal_year_progress" numeric)
	stable
	parallel safe
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function public.calc_temporal_features(text) owner to postgres
;