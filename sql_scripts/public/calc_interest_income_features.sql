CREATE FUNCTION public.calc_interest_income_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "interest_income_ltm" numeric, "interest_expense_ltm" numeric, "net_interest_income" numeric, "interest_coverage_ratio" numeric, "interest_income_to_revenue" numeric, "interest_expense_to_revenue" numeric, "net_interest_margin_proxy" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_interest_income_features(text) OWNER TO postgres;