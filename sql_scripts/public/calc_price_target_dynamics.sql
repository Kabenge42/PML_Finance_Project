create function calc_price_target_dynamics(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, pt_momentum_1w numeric, pt_momentum_1m numeric, pt_momentum_3m numeric, pt_momentum_6m numeric, pt_momentum_1y numeric, pt_median_momentum_1m numeric, pt_median_momentum_3m numeric, pt_acceleration_short numeric, pt_acceleration_long numeric, pt_consensus_convergence numeric, analyst_coverage_change_1m integer, analyst_coverage_change_3m integer, analyst_coverage_change_1y integer, pt_vs_price_momentum numeric, analyst_coverage_trend numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                            AS isin,
       ("Price Target" - "Price Target (1W Ago)") / NULLIF("Price Target (1W Ago)", 0)   AS pt_momentum_1w,
       ("Price Target" - "Price Target (1M Ago)") / NULLIF("Price Target (1M Ago)", 0)   AS pt_momentum_1m,
       ("Price Target" - "Price Target (3M Ago)") / NULLIF("Price Target (3M Ago)", 0)   AS pt_momentum_3m,
       ("Price Target" - "Price Target (6M Ago)") / NULLIF("Price Target (6M Ago)", 0)   AS pt_momentum_6m,
       ("Price Target" - "Price Target (1Y Ago)") / NULLIF("Price Target (1Y Ago)", 0)   AS pt_momentum_1y,
       ("Price Target - Median" - "Price Target - Median (1M Ago)") /
       NULLIF("Price Target - Median (1M Ago)", 0)                                       AS pt_median_momentum_1m,
       ("Price Target - Median" - "Price Target - Median (3M Ago)") /
       NULLIF("Price Target - Median (3M Ago)", 0)                                       AS pt_median_momentum_3m,
       (("Price Target" - "Price Target (1M Ago)") / NULLIF("Price Target (1M Ago)", 0)) -
       (("Price Target" - "Price Target (3M Ago)") / NULLIF("Price Target (3M Ago)", 0)) AS pt_acceleration_short,
       (("Price Target" - "Price Target (3M Ago)") / NULLIF("Price Target (3M Ago)", 0)) -
       (("Price Target" - "Price Target (1Y Ago)") / NULLIF("Price Target (1Y Ago)", 0)) AS pt_acceleration_long,
       (("Price Target - High (3M Ago)" - "Price Target - Low (3M Ago)") /
        NULLIF("Price Target - Median (3M Ago)", 0)) -
       (("Price Target - High" - "Price Target - Low") /
        NULLIF("Price Target - Median", 0))                                              AS pt_consensus_convergence,
       ("Price Target - #" - "Price Target - # (1M Ago)")::INTEGER                       AS analyst_coverage_change_1m,
       ("Price Target - #" - "Price Target - # (3M Ago)")::INTEGER                       AS analyst_coverage_change_3m,
       ("Price Target - #" - "Price Target - # (1Y Ago)")::INTEGER                       AS analyst_coverage_change_1y,
       (("Price Target" / NULLIF("Last Price", 0)) -
        ("Price Target (3M Ago)" / NULLIF("Price (3M Ago)", 0))) /
       NULLIF(("Price Target (3M Ago)" / NULLIF("Price (3M Ago)", 0)), 0)                AS pt_vs_price_momentum,
       (COALESCE("Price Target - #" - "Price Target - # (1M Ago)", 0) * 0.40 +
        COALESCE("Price Target - #" - "Price Target - # (3M Ago)", 0) * 0.35 +
        COALESCE("Price Target - #" - "Price Target - # (6M Ago)", 0) * 0.25) /
       NULLIF("Price Target - #"::NUMERIC, 0)                                            AS analyst_coverage_trend
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_price_target_dynamics(text) owner to postgres;

