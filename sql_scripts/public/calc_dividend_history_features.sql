create function public.calc_dividend_history_features(p_isin text default NULL::text)
	returns table("isin" text, "div_yield_2fy" numeric, "div_yield_3fy" numeric, "div_yield_4fy" numeric, "div_yield_5fy" numeric, "div_yield_trend_3y" numeric, "div_yield_volatility" numeric, "div_yield_declining_flag" integer, "div_yield_mean_5y" numeric, "div_yield_vs_5y_mean" numeric)
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

alter function public.calc_dividend_history_features(text) owner to postgres
;