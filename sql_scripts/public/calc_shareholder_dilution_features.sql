create function calc_shareholder_dilution_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, dilution_score numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"         AS isin,
       GREATEST(0, LEAST(100,
                         50 - (("Shrs Out" - "Shrs Out (-1FY)") / NULLIF("Shrs Out (-1FY)", 0)) * 100
                   )) AS dilution_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_shareholder_dilution_features(text) owner to postgres;

