CREATE FUNCTION public.calc_extended_valuation_timeseries(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                     text,
		        "ev_sales_qoq_1q"          numeric,
		        "ev_sales_qoq_2q"          numeric,
		        "ev_sales_qoq_3q"          numeric,
		        "ev_sales_qoq_4q"          numeric,
		        "p_e_vs_5y_avg"            numeric,
		        "p_e_percentile_proxy"     numeric,
		        "valuation_mean_reversion" numeric,
		        "ev_ebitda_qoq_trend"      numeric,
		        "p_b_momentum_yoy"         numeric,
		        "valuation_compression"    numeric,
		        "forward_pe_premium"       numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_extended_valuation_timeseries(text) OWNER TO postgres;