CREATE FUNCTION public.calc_net_income_comprehensive(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                       text,
		        "net_income_is_fq"           numeric,
		        "net_income_is_ltm"          numeric,
		        "net_income_is_fy"           numeric,
		        "net_income_adj_ltm"         numeric,
		        "normalized_ni_ltm"          numeric,
		        "net_income_is_1fqfq"        numeric,
		        "net_income_is_2fqfq"        numeric,
		        "net_income_is_3fqfq"        numeric,
		        "net_income_is_4fqfq"        numeric,
		        "net_income_is_1fy"          numeric,
		        "net_income_is_2fy"          numeric,
		        "net_income_is_3fy"          numeric,
		        "net_income_is_4fy"          numeric,
		        "net_income_is_5yavgfq"      numeric,
		        "net_income_is_5yavgltm"     numeric,
		        "normalized_ni_5yavgfq"      numeric,
		        "normalized_ni_5yavgltm"     numeric,
		        "net_income_growth_yoy"      numeric,
		        "net_income_margin_ltm"      numeric,
		        "ni_adjustment_ratio"        numeric,
		        "net_income_positive_years"  integer,
		        "earnings_quality_composite" numeric,
		        "net_income_qoq_growth"      numeric,
		        "net_income_yoy_quarterly"   numeric,
		        "net_income_vs_5y_avg"       numeric,
		        "normalized_ni_vs_5y_avg"    numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_net_income_comprehensive(text) OWNER TO postgres;