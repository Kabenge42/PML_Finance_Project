CREATE FUNCTION public.calc_investment_income_temporal(p_isin text default NULL::text)
	RETURNS table("isin" text, "inv_income_ltm" numeric, "inv_income_fq" numeric, "inv_income_fy" numeric, "inv_income_qoq_growth" numeric, "inv_income_yoy_growth" numeric, "inv_income_to_revenue" numeric, "inv_income_trend_3y" numeric, "inv_income_positive_quarters" integer, "financial_company_proxy" integer)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_investment_income_temporal(text) OWNER TO postgres;