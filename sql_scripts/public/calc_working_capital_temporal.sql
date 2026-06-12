CREATE FUNCTION public.calc_working_capital_temporal(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                 text,
		        "wc_fq"                numeric,
		        "wc_fy"                numeric,
		        "wc_ltm"               numeric,
		        "wc_5yavgfy"           numeric,
		        "wc_1fq"               numeric,
		        "wc_2fq"               numeric,
		        "wc_3fq"               numeric,
		        "wc_4fq"               numeric,
		        "wc_1fy"               numeric,
		        "wc_2fy"               numeric,
		        "wc_3fy"               numeric,
		        "wc_4fy"               numeric,
		        "wc_qoq_change"        numeric,
		        "wc_yoy_change"        numeric,
		        "wc_4q_trend"          numeric,
		        "wc_vs_5y_avg"         numeric,
		        "wc_positive_quarters" integer,
		        "wc_improving_flag"    integer,
		        "wc_volatility"        numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_working_capital_temporal(text) OWNER TO postgres;