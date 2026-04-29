create function calc_beta_risk_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                 text,
                beta_1y              numeric,
                beta_5y              numeric,
                beta_spread          numeric,
                beta_trend           numeric,
                high_beta_flag       integer,
                low_beta_flag        integer,
                beta_stability_score numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                        AS isin,
       "Beta (1Y)"                                   AS beta_1y,
       "Beta (5Y)"                                   AS beta_5y,
       "Beta (1Y)" - "Beta (5Y)"                     AS beta_spread,
       ("Beta (1Y)" - "Beta (5Y)") / NULLIF(ABS("Beta (5Y)"), 0) * 100
                                                     AS beta_trend,
       CASE WHEN "Beta (1Y)" > 1.5 THEN 1 ELSE 0 END AS high_beta_flag,
       CASE WHEN "Beta (1Y)" < 0.5 THEN 1 ELSE 0 END AS low_beta_flag,
       GREATEST(0, LEAST(100,
                         100 - ABS("Beta (1Y)" - "Beta (5Y)") * 50
                   ))                                AS beta_stability_score
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_beta_risk_features(text) owner to postgres;

