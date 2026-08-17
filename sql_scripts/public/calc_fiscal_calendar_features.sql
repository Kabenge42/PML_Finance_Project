create function public.calc_fiscal_calendar_features(p_isin text default NULL::text)
	returns table("isin" text, "days_since_last_report" integer, "days_to_fy_end" integer, "is_quarter_end_month" integer, "is_fy_end_month" integer, "earnings_season_flag" integer, "pre_earnings_window" integer, "post_earnings_window" integer, "reporting_freshness_score" numeric, "fiscal_quarter_progress" numeric)
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

alter function public.calc_fiscal_calendar_features(text) owner to postgres
;