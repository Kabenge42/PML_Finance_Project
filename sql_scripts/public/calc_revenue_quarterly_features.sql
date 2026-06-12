CREATE FUNCTION public.calc_revenue_quarterly_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                        text,
		        "revenue_fq"                  numeric,
		        "revenue_fy"                  numeric,
		        "revenue_ltm"                 numeric,
		        "revenue_5y_avg"              numeric,
		        "revenue_1fqfq"               numeric,
		        "revenue_2fqfq"               numeric,
		        "revenue_3fqfq"               numeric,
		        "revenue_4fqfq"               numeric,
		        "revenue_1fy"                 numeric,
		        "revenue_2fy"                 numeric,
		        "revenue_3fy"                 numeric,
		        "revenue_4fy"                 numeric,
		        "revenue_yoy_growth"          numeric,
		        "revenue_vs_5y_avg"           numeric,
		        "revenue_ltm_vs_fy"           numeric,
		        "revenue_fq_vs_5y_avg_fq"     numeric,
		        "revenue_qoq_growth"          numeric,
		        "revenue_qoq_2q"              numeric,
		        "revenue_qoq_3q"              numeric,
		        "revenue_qoq_4q"              numeric,
		        "revenue_yoy_quarterly"       numeric,
		        "revenue_2y_growth"           numeric,
		        "revenue_3y_growth"           numeric,
		        "revenue_4y_growth"           numeric,
		        "revenue_cagr_3y"             numeric,
		        "revenue_cagr_4y"             numeric,
		        "revenue_4q_trend"            numeric,
		        "revenue_4q_avg"              numeric,
		        "revenue_fq_vs_4q_avg"        numeric,
		        "revenue_growth_flag"         integer,
		        "revenue_stability_score"     numeric,
		        "revenue_accelerating_flag"   integer,
		        "revenue_positive_qoq_streak" integer
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_revenue_quarterly_features(text) OWNER TO postgres;