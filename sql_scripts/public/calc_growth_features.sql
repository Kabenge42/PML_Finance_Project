create function public.calc_growth_features(p_isin text default NULL::text)
	returns table("isin" text, "revenue_growth_yoy" numeric, "ebitda_growth_yoy" numeric, "operating_income_growth" numeric, "fcf_growth" numeric, "revenue_cagr_5y" numeric, "forward_revenue_growth" numeric, "revenue_vs_5y_avg" numeric)
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

alter function public.calc_growth_features(text) owner to postgres
;