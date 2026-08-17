create function public.calc_size_liquidity_features(p_isin text default NULL::text)
	returns table("isin" text, "market_cap" numeric, "market_cap_country_r" numeric, "log_market_cap" numeric, "volume_shrs" numeric, "relative_volume" numeric, "shares_outstanding" numeric, "daily_turnover_ratio" numeric, "size_class" text, "style_class" text, "liquidity_score" numeric)
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

alter function public.calc_size_liquidity_features(text) owner to postgres
;