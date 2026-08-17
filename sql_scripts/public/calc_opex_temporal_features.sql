create function public.calc_opex_temporal_features(p_isin text default NULL::text)
	returns table("isin" text, "opex_fq" numeric, "opex_ltm" numeric, "opex_fy" numeric, "opex_qoq_growth" numeric, "opex_yoy_growth" numeric, "opex_vs_revenue_trend" numeric, "sga_qoq_growth" numeric, "sga_yoy_growth" numeric, "operating_leverage_score" numeric)
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

alter function public.calc_opex_temporal_features(text) owner to postgres
;