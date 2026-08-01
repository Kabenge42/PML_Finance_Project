CREATE FUNCTION public.calc_cashflow_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "cfo_to_net_income" numeric, "fcf_to_net_income" numeric, "fcf_margin" numeric, "cfo_growth_yoy" numeric, "fcf_positive_ratio" numeric, "acquisition_intensity" numeric, "self_funding_ratio" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_cashflow_features(text) OWNER TO postgres;