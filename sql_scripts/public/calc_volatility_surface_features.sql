create function public.calc_volatility_surface_features(p_isin text default NULL::text)
	returns table("isin" text, "vol_1m" numeric, "vol_3m" numeric, "vol_6m" numeric, "vol_1y" numeric, "vol_term_spread_short" numeric, "vol_term_spread_long" numeric, "vol_ratio_3m_1y" numeric, "vol_hump" numeric, "beta_1y" numeric, "beta_2y" numeric, "beta_5y" numeric, "beta_term_structure" numeric, "beta_convexity" numeric, "realized_vs_implied_proxy" numeric)
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

alter function public.calc_volatility_surface_features(text) owner to postgres
;