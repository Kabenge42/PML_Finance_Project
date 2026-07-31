CREATE FUNCTION public.calc_shareholder_dilution_features(p_isin text DEFAULT NULL::text)
	RETURNS TABLE(isin text, dilution_score numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT "ISIN"                                                                                                AS isin,
       GREATEST(0, LEAST(100, 50 - (("Shrs Out" - "Shrs Out (-1FY)") / NULLIF("Shrs Out (-1FY)", 0)) *
                                   100))                                                                     AS dilution_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

ALTER FUNCTION public.calc_shareholder_dilution_features(unknown) OWNER TO postgres;