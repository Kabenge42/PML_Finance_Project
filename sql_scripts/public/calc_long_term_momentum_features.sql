create function public.calc_long_term_momentum_features(p_isin text default NULL::text)
	returns table("isin" text, "price_momentum_1y" numeric, "price_momentum_3y" numeric, "price_momentum_5y" numeric, "long_term_trend_score" numeric, "price_vs_ema_250d" numeric, "multi_year_high_flag" integer, "secular_trend_flag" integer, "total_return_ytd" numeric, "total_return_5y" numeric, "total_return_10y" numeric, "return_cagr_3y" numeric, "return_cagr_10y" numeric, "return_vs_price_momentum" numeric, "return_consistency_score" numeric)
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

alter function public.calc_long_term_momentum_features(text) owner to postgres
;