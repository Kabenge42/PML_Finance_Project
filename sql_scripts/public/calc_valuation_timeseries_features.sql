create function public.calc_valuation_timeseries_features(p_isin text default NULL::text)
	returns table("isin" text, "ev_sales_trend_1y" numeric, "ev_ebitda_momentum" numeric, "p_e_momentum_yoy" numeric, "p_e_momentum_qoq" numeric, "ev_sales_vs_3y_avg" numeric, "ev_ebitda_vs_3y_avg" numeric, "p_e_vs_3y_avg" numeric, "ev_sales_forward_discount" numeric, "ev_ebitda_forward_discount" numeric, "p_e_forward_discount" numeric, "p_b_vs_5y_avg" numeric)
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

alter function public.calc_valuation_timeseries_features(text) owner to postgres
;