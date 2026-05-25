CREATE MATERIALIZED VIEW pml.mv_pymc_dividend_safety AS
SELECT isin,
       ticker,
       region,
       country,
       trading_country,
       exchange,
       unit,
       style_class,
       size_class,
       sector,
       industry,
       div_yield_ltm                                                                          AS observed_div_yield,
       dividend_streak                                                                        AS n_streak,
       dividend_record_frequency                                                              AS feat_div_frequency,
       fcf_dividend_coverage(fcf_ltm::numeric, common_dividends_paid_ltm::numeric)            AS feat_fcf_coverage,
       fcf_dividend_coverage(cfo_ltm::numeric, common_dividends_paid_ltm::numeric)            AS feat_cfo_coverage,
       safe_divide(dividend_per_share_ltm, NULLIF(net_eps_basic_ltm, 0::double precision))    AS feat_eps_payout_ratio,
       calc_change_ratio(dividend_per_share_ltm::numeric, dividend_per_share_neg1fy::numeric) AS feat_dps_growth_1y,
       calc_change_ratio(dividend_per_share_ltm::numeric, dividend_per_share_neg3fy::numeric) AS feat_dps_growth_3y,
       calc_change_ratio(dividend_per_share_ltm::numeric, dividend_per_share_neg5fy::numeric) AS feat_dps_growth_5y,
       buyback_yield_ltm                                                                      AS feat_buyback_yield,
       div_yield_ltm + COALESCE(buyback_yield_ltm, 0::double precision)                       AS feat_total_yield,
       repurchase_common_stock_ltm                                                            AS feat_repurchases_ltm,
       altman_z_score_ltm                                                                     AS feat_altman_z,
       return_on_assets_roa_pct_ltm                                                           AS feat_roa_ltm,
       div_yield_ltm - div_yield_5yavgltm                                                     AS feat_yield_spread_vs_5y
FROM pml_df;

ALTER MATERIALIZED VIEW pml.mv_pymc_dividend_safety OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_pymc_dividend_safety_isin ON pml.mv_pymc_dividend_safety (isin);