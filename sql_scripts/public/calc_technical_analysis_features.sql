create function public.calc_technical_analysis_features(p_isin text default NULL::text)
	returns table("isin" text, "ema_slope_20d" numeric, "ema_trend_consistency" integer, "price_vs_ema_100d" numeric, "near_52w_high_flag" integer, "near_52w_low_flag" integer, "volume_momentum_score" numeric, "breakout_signal" integer, "high_volume_flag" integer, "low_volume_flag" integer, "volatility_compression" numeric, "volatility_term_structure" numeric)
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

alter function public.calc_technical_analysis_features(text) owner to postgres
;