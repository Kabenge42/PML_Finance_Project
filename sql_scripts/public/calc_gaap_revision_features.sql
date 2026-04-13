create function calc_gaap_revision_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                         text,
                gaap_revision_momentum       numeric,
                gaap_revision_1m             numeric,
                gaap_revision_3m             numeric,
                gaap_revision_6m             numeric,
                gaap_revision_1y             numeric,
                gaap_vs_norm_revision_spread numeric,
                gaap_revision_acceleration   numeric,
                gaap_positive_revision_flag  integer,
                revision_quality_divergence  numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                                      AS isin,
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1M)", 0) * 0.35 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 3M)", 0) * 0.30 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 6M)", 0) * 0.20 +
       COALESCE("EPS GAAP Est Avg Rev % (FY1E - 1Y)", 0) * 0.15                    AS gaap_revision_momentum,
       "EPS GAAP Est Avg Rev % (FY1E - 1M)"                                        AS gaap_revision_1m,
       "EPS GAAP Est Avg Rev % (FY1E - 3M)"                                        AS gaap_revision_3m,
       "EPS GAAP Est Avg Rev % (FY1E - 6M)"                                        AS gaap_revision_6m,
       "EPS GAAP Est Avg Rev % (FY1E - 1Y)"                                        AS gaap_revision_1y,
       "EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)"      AS gaap_vs_norm_revision_spread,
       "EPS GAAP Est Avg Rev % (FY1E - 1M)" - "EPS GAAP Est Avg Rev % (FY1E - 6M)" AS gaap_revision_acceleration,
       CASE
           WHEN "EPS GAAP Est Avg Rev % (FY1E - 1M)" > 0
               AND "EPS GAAP Est Avg Rev % (FY1E - 3M)" > 0
               AND "EPS GAAP Est Avg Rev % (FY1E - 6M)" > 0
               THEN 1
           ELSE 0
           END                                                                     AS gaap_positive_revision_flag,
       ABS(("EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)") -
           ("EPS Est Avg Rev % (FY1E - 1M)" - "EPS GAAP Est Avg Rev % (FY1E - 1M)"))
                                                                                   AS revision_quality_divergence
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_gaap_revision_features(text) owner to postgres;

