CREATE FUNCTION public.calc_working_capital_deep_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                 text,
		        "working_capital_ltm"  numeric,
		        "working_capital_fq"   numeric,
		        "working_capital_fy"   numeric,
		        "wc_to_revenue"        numeric,
		        "wc_to_assets"         numeric,
		        "wc_change_qoq"        numeric,
		        "wc_change_yoy"        numeric,
		        "days_working_capital" numeric,
		        "wc_efficiency_score"  numeric,
		        "negative_wc_flag"     integer,
		        "wc_improvement_flag"  integer
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_working_capital_deep_features(text) OWNER TO postgres;