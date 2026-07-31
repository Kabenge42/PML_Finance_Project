CREATE FUNCTION public.calc_volatility_surface_features(p_isin text DEFAULT NULL::text)
	RETURNS TABLE(isin text, vol_1m numeric, vol_3m numeric, vol_6m numeric, vol_1y numeric, vol_term_spread_short numeric, vol_term_spread_long numeric, vol_ratio_3m_1y numeric, vol_hump numeric, beta_1y numeric, beta_2y numeric, beta_5y numeric, beta_term_structure numeric, beta_convexity numeric, realized_vs_implied_proxy numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT "ISIN",
       "Volatility (1M)",
       "Volatility (3M)",
       "Volatility (6M)",
       "Volatility (1Y)",
       "Volatility (3M)" - "Volatility (1M)"                             AS vol_term_spread_short,
       "Volatility (1Y)" - "Volatility (6M)"                             AS vol_term_spread_long,
       public.safe_divide("Volatility (3M)", "Volatility (1Y)")          AS vol_ratio_3m_1y,
       "Volatility (6M)" - ("Volatility (3M)" + "Volatility (1Y)") / 2.0 AS vol_hump,
       "Beta (1Y)",
       "Beta (2Y)",
       "Beta (5Y)",
       public.calc_change_ratio("Beta (1Y)", "Beta (5Y)")                AS beta_term_structure,
       "Beta (2Y)" - ("Beta (1Y)" + "Beta (5Y)") / 2.0                   AS beta_convexity,
       public.safe_divide("Volatility (1M)", "Volatility (1Y)")          AS realized_vs_implied_proxy
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

ALTER FUNCTION public.calc_volatility_surface_features(unknown) OWNER TO postgres;