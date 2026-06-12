CREATE FUNCTION public.calc_valuation_timeseries_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                       text,
		        "ev_sales_trend_1y"          numeric,
		        "ev_ebitda_momentum"         numeric,
		        "p_e_momentum_yoy"           numeric,
		        "p_e_momentum_qoq"           numeric,
		        "ev_sales_vs_3y_avg"         numeric,
		        "ev_ebitda_vs_3y_avg"        numeric,
		        "p_e_vs_3y_avg"              numeric,
		        "ev_sales_forward_discount"  numeric,
		        "ev_ebitda_forward_discount" numeric,
		        "p_e_forward_discount"       numeric,
		        "p_b_vs_5y_avg"              numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_valuation_timeseries_features(text) OWNER TO postgres;