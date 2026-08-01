CREATE FUNCTION public.calc_cashflow_temporal_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "cfo_quarterly_trend" numeric, "cfo_yoy_quarterly" numeric, "cfi_quarterly_trend" numeric, "cff_quarterly_trend" numeric, "fcf_quarterly_trend" numeric, "cfo_positive_quarters" integer, "cfi_negative_quarters" integer, "cff_pattern_score" numeric, "cash_burn_rate" numeric, "cf_volatility_score" numeric, "operating_cf_momentum" numeric, "financing_dependency" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_cashflow_temporal_features(text) OWNER TO postgres;