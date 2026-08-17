create function public.calc_working_capital_deep_features(p_isin text default NULL::text)
	returns table("isin" text, "working_capital_ltm" numeric, "working_capital_fq" numeric, "working_capital_fy" numeric, "wc_to_revenue" numeric, "wc_to_assets" numeric, "wc_change_qoq" numeric, "wc_change_yoy" numeric, "days_working_capital" numeric, "wc_efficiency_score" numeric, "negative_wc_flag" integer, "wc_improvement_flag" integer)
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

alter function public.calc_working_capital_deep_features(text) owner to postgres
;