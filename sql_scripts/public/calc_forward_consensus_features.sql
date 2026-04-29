create function calc_forward_consensus_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, pe_ntm numeric, pe_est_fy1 numeric, pe_forward_discount numeric, eps_gaap_vs_norm_ntm numeric, eps_gaap_vs_norm_fy1e numeric, forward_adjustment_trend numeric, ebitda_est_ntm numeric, ebitda_est_fy1e numeric, ev_ebitda_est_fy1 numeric, ebitda_forward_growth numeric, earnings_revision_divergence numeric, forward_pe_vs_sector_proxy numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN",
       "P/E (NTM)",
       "P/E (EST FY1)",
       public.calc_change_ratio("P/E (NTM)", "P/E (LTM)")                       AS pe_forward_discount,
       "EPS GAAP - Est Avg (NTM)" - "EPS Norm - Est Avg (NTM)"                  AS eps_gaap_vs_norm_ntm,
       "EPS GAAP - Est Avg (FY1E)" - "EPS Norm - Est Avg (FY1E)"                AS eps_gaap_vs_norm_fy1e,
       ("EPS GAAP - Est Avg (FY1E)" - "EPS Norm - Est Avg (FY1E)") -
       ("EPS/Adj. (LTM)" - "Net EPS - Basic (LTM)")                             AS forward_adjustment_trend,
       "EBITDA - Est Avg (NTM)",
       "EBITDA - Est Avg (FY1E)",
       "EV/EBITDA (EST FY1)",
       public.calc_change_ratio("EBITDA - Est Avg (FY1E)", "EBITDA (LTM)")      AS ebitda_forward_growth,
       ("EPS Est Avg Rev % (FY1E - 3M)" - "EPS GAAP Est Avg Rev % (FY1E - 3M)") -
       ("EPS Est Avg Rev % (FY1E - 1M)" - "EPS GAAP Est Avg Rev % (FY1E - 1M)") AS earnings_revision_divergence,
       public.calc_change_ratio("P/E (NTM)", "P/E (3YAVGLTM)")                  AS forward_pe_vs_sector_proxy
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_forward_consensus_features(text) owner to postgres;

