create function public.calc_margin_trends(p_isin text default NULL::text)
	returns table("isin" text, "gross_margin_trend_yoy" numeric, "operating_margin_trend" numeric, "net_margin_trend_yoy" numeric, "ebitda_margin_trend" numeric, "margin_expansion_flag" integer, "margin_stability_score" numeric)
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

alter function public.calc_margin_trends(text) owner to postgres
;