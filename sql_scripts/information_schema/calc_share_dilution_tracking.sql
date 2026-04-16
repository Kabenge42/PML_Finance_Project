create function information_schema.calc_share_dilution_tracking(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                  text,
                shrs_out_1fy          numeric,
                shares_yoy_change_pct numeric,
                net_buyback_flag      integer
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN",
       "Shrs Out (-1FY)",
       public.calc_change_ratio("Shrs Out", "Shrs Out (-1FY)")    AS shares_yoy_change_pct,
       CASE WHEN "Shrs Out" < "Shrs Out (-1FY)" THEN 1 ELSE 0 END AS net_buyback_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function information_schema.calc_share_dilution_tracking(unknown) owner to postgres;

