CREATE FUNCTION public.calc_volatility_surface_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                      text,
		        "vol_1m"                    numeric,
		        "vol_3m"                    numeric,
		        "vol_6m"                    numeric,
		        "vol_1y"                    numeric,
		        "vol_term_spread_short"     numeric,
		        "vol_term_spread_long"      numeric,
		        "vol_ratio_3m_1y"           numeric,
		        "vol_hump"                  numeric,
		        "beta_1y"                   numeric,
		        "beta_2y"                   numeric,
		        "beta_5y"                   numeric,
		        "beta_term_structure"       numeric,
		        "beta_convexity"            numeric,
		        "realized_vs_implied_proxy" numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_volatility_surface_features(text) OWNER TO postgres;