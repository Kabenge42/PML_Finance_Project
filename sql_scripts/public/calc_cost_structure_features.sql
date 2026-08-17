create function public.calc_cost_structure_features(p_isin text default NULL::text)
	returns table("isin" text, "cogs_to_revenue" numeric, "opex_to_revenue" numeric, "sga_to_revenue" numeric, "rnd_to_revenue" numeric, "interest_to_revenue" numeric, "sga_trend_yoy" numeric, "operating_leverage_proxy" numeric, "cost_efficiency_score" numeric, "marketing_to_revenue" numeric, "marketing_trend_yoy" numeric, "marketing_vs_5y_avg" numeric, "sga_vs_5y_avg" numeric, "sga_efficiency_trend" numeric)
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

alter function public.calc_cost_structure_features(text) owner to postgres
;