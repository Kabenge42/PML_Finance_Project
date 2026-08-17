create function public.calc_extended_valuation_timeseries(p_isin text default NULL::text)
	returns table("isin" text, "ev_sales_qoq_1q" numeric, "ev_sales_qoq_2q" numeric, "ev_sales_qoq_3q" numeric, "ev_sales_qoq_4q" numeric, "p_e_vs_5y_avg" numeric, "p_e_percentile_proxy" numeric, "valuation_mean_reversion" numeric, "ev_ebitda_qoq_trend" numeric, "p_b_momentum_yoy" numeric, "valuation_compression" numeric, "forward_pe_premium" numeric)
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

alter function public.calc_extended_valuation_timeseries(text) owner to postgres
;