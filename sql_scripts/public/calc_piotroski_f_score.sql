CREATE FUNCTION public.calc_piotroski_f_score(p_isin text DEFAULT NULL::text)
	RETURNS TABLE(isin text, piotroski_f_score integer)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT "ISIN"                                                                                AS isin,
       (CASE WHEN "Return on Assets (ROA) % (LTM)" > 0 THEN 1 ELSE 0 END + CASE WHEN "CFO (LTM)" > 0 THEN 1 ELSE 0 END +
        CASE WHEN "Return on Assets (ROA) % (LTM)" > "Return on Assets (ROA) % (FY)" THEN 1 ELSE 0 END +
        CASE WHEN "CFO (LTM)" > "Net Income - (IS) (LTM)" THEN 1 ELSE 0 END + CASE
	                                                                              WHEN "Total Debt (LTM)" /
	                                                                                   NULLIF("Total Equity (LTM)", 0) <
	                                                                                   "Total Debt (FY)" /
	                                                                                   NULLIF("Total Equity (FY)", 0)
		                                                                              THEN 1
	                                                                              ELSE 0 END +
        CASE WHEN "Current Ratio (LTM)" > "Current Ratio (FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Shrs Out" <= "Shrs Out (-1FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Gross Profit Margin % (LTM)" > "Gross Profit Margin % (FY)" THEN 1 ELSE 0 END +
        CASE WHEN "Asset Turnover (LTM)" > "Asset Turnover (FY)" THEN 1 ELSE 0 END)::INTEGER AS piotroski_f_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

ALTER FUNCTION public.calc_piotroski_f_score(unknown) OWNER TO postgres;