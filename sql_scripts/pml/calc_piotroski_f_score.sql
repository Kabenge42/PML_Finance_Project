-- piotroski_f_score LTM screener: thin set-returning wrapper over the scalar
-- pml.piotroski_f_score composite (see piotroski_f_score.sql).
-- LTM-era signal wiring: ROA/CFO/leverage/liquidity/margin compare LTM vs
-- neg1fy, share count compares shrs_out vs shrs_out_neg1fy, and asset turnover
-- uses the FQ-vs-FY proxy (no LTM asset-turnover column exists).
CREATE OR REPLACE FUNCTION calc_piotroski_f_score(p_isin text DEFAULT NULL::text)
	RETURNS TABLE(isin text, piotroski_f_score integer)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT isin AS isin,
       pml.piotroski_f_score(return_on_assets_roa_pct_ltm, return_on_assets_roa_pct_neg1fy,
                             cfo_ltm, net_income_ltm,
                             long_term_debt_equity_ltm, long_term_debt_equity_neg1fy,
                             current_ratio_ltm, current_ratio_neg1fy,
                             shrs_out, shrs_out_neg1fy,
                             gross_profit_margin_pct_ltm, gross_profit_margin_pct_neg1fy,
                             asset_turnover_fq, asset_turnover_fy) AS piotroski_f_score
FROM pml.pml_df pd
WHERE p_isin IS NULL
   OR isin = p_isin;
$$;

ALTER FUNCTION calc_piotroski_f_score(text) OWNER TO postgres;
