CREATE FUNCTION calc_piotroski_f_score(p_isin text DEFAULT NULL::text)
	RETURNS TABLE(isin text, piotroski_f_score integer)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT isin                                                                                AS isin,
       (CASE WHEN return_on_assets_roa_pct_ltm > 0 THEN 1 ELSE 0 END + CASE WHEN cfo_ltm > 0 THEN 1 ELSE 0 END +
        CASE WHEN return_on_assets_roa_pct_ltm > return_on_assets_roa_pct_neg1fy THEN 1 ELSE 0 END +
        CASE WHEN cfo_ltm > net_income_ltm THEN 1 ELSE 0 END + CASE
	                                                                              WHEN long_term_debt_equity_ltm <
	                                                                                   long_term_debt_equity_neg1fy
		                                                                              THEN 1
	                                                                              ELSE 0 END +
        CASE WHEN current_ratio_ltm > current_ratio_neg1fy THEN 1 ELSE 0 END +
        CASE WHEN shrs_out <= shrs_out_neg1fy THEN 1 ELSE 0 END +
        CASE WHEN gross_profit_margin_pct_ltm > gross_profit_margin_pct_neg1fy THEN 1 ELSE 0 END +
        CASE WHEN asset_turnover_fq > asset_turnover_fy THEN 1 ELSE 0 END)::INTEGER AS piotroski_f_score
FROM postgres.pml.pml_df pd
WHERE p_isin IS NULL
   OR isin = p_isin;
$$;

ALTER FUNCTION calc_piotroski_f_score(text) OWNER TO postgres;