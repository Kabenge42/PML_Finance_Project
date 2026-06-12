CREATE FUNCTION public.calc_growth_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                    text,
		        "revenue_growth_yoy"      numeric,
		        "ebitda_growth_yoy"       numeric,
		        "operating_income_growth" numeric,
		        "fcf_growth"              numeric,
		        "revenue_cagr_5y"         numeric,
		        "forward_revenue_growth"  numeric,
		        "revenue_vs_5y_avg"       numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_growth_features(text) OWNER TO postgres;