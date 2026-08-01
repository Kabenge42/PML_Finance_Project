CREATE FUNCTION public.calc_cashflow_comprehensive(p_isin text default NULL::text)
	RETURNS table("isin" text, "cfo_fq" numeric, "cfo_ltm" numeric, "cfo_fy" numeric, "fcf_fq" numeric, "fcf_ltm" numeric, "fcf_fy" numeric, "cfo_growth_yoy" numeric, "fcf_growth_yoy" numeric, "cfo_to_net_income" numeric, "fcf_margin" numeric, "fcf_yield" numeric, "cfo_positive_years" integer, "fcf_positive_years" integer, "cash_flow_quality_score" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_cashflow_comprehensive(text) OWNER TO postgres;