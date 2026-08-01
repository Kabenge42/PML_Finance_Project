CREATE FUNCTION public.calc_profitability_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "roe" numeric, "roa" numeric, "gross_margin_pct" numeric, "operating_margin_pct" numeric, "net_margin_pct" numeric, "ebitda_margin_pct" numeric, "roic" numeric, "rnd_intensity" numeric, "equity_multiplier" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_profitability_features(text) OWNER TO postgres;