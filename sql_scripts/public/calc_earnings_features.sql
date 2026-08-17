create function public.calc_earnings_features(p_isin text default NULL::text)
	returns table("isin" text, "eps_surprise_pct" numeric, "revenue_surprise_pct" numeric, "eps_adjustment_ratio" numeric, "gaap_adj_eps_gap_pct" numeric, "ebitda_adjustment_ratio" numeric, "eps_quarterly_trend" numeric, "eps_yoy_growth" numeric)
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

alter function public.calc_earnings_features(text) owner to postgres
;