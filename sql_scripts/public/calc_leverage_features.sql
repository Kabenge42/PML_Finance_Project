CREATE FUNCTION public.calc_leverage_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "debt_to_equity" numeric, "debt_to_assets" numeric, "equity_ratio" numeric, "interest_coverage" numeric, "current_ratio" numeric, "cash_ratio" numeric, "working_capital_ratio" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_leverage_features(text) OWNER TO postgres;