create function public.calc_momentum_features(p_isin text default NULL::text)
	returns table("isin" text, "price_momentum_1m" numeric, "price_momentum_3m" numeric, "price_momentum_6m" numeric, "price_momentum_1y" numeric, "price_momentum_5d" numeric, "ema_crossover_20_50" integer, "ema_crossover_50_250" integer, "price_vs_ema_20d" numeric, "price_vs_ema_250d" numeric, "pct_off_52w_high" numeric, "pct_above_52w_low" numeric, "range_52w_position" numeric, "beta_momentum" numeric, "volatility_regime" numeric)
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

alter function public.calc_momentum_features(text) owner to postgres
;