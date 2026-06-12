CREATE FUNCTION public.calc_efficiency_ratios(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                  text,
		        "asset_turnover"        numeric,
		        "inventory_turnover"    numeric,
		        "receivables_days"      numeric,
		        "working_capital_turns" numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_efficiency_ratios(text) OWNER TO postgres;