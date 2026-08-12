create materialized view mv_pymc_dcf_pt as
SELECT isin,
       ticker,
       trading_region,
       region,
       country,
       trading_country,
       exchange,
       unit,
       style_class,
       size_class,
       sector,
       industry,
       price_target                                                            AS observed_pt,
       last_price                                                              AS observed_price,
       market_cap,
       enterprise_value,
       shrs_out,
       fcf_ltm                                                                 AS feat_fcf_ltm,
       fcf_est_avg_fy1e                                                        AS feat_fcf_fy1e,
       fcf_est_avg_fy2e                                                        AS feat_fcf_fy2e,
       fcf_est_avg_fy3e                                                        AS feat_fcf_fy3e,
       fcf_est_avg_fy4e                                                        AS feat_fcf_fy4e,
       fcf_est_avg_fy5e                                                        AS feat_fcf_fy5e,
       calc_change_ratio(fcf_est_avg_fy1e::numeric, fcf_ltm::numeric)          AS feat_fcf_growth_1y,
       calc_change_ratio(fcf_est_avg_fy3e::numeric, fcf_est_avg_fy1e::numeric) AS feat_fcf_growth_2y,
       calc_change_ratio(fcf_est_avg_fy5e::numeric, fcf_est_avg_fy3e::numeric) AS feat_fcf_terminal_growth,
       safe_divide(capital_expenditure_ltm, cfo_ltm)                           AS feat_reinvest_rate,
       safe_divide(capital_expenditure_ltm, fcf_ltm)                           AS feat_capex_to_fcf,
       cfo_ltm                                                                 AS feat_cfo_ltm,
       tot_return_pct_cagr_3y                                                  AS feat_tr_cagr_3y,
       tot_return_pct_cagr_10y                                                 AS feat_tr_cagr_10y,
       peg_ntm                                                                 AS feat_peg_ntm,
       ev_sales_ltm                                                            AS feat_ev_sales_ltm,
       ev_ebitda_ntm                                                           AS feat_ev_ebitda_ntm,
       return_on_assets_roa_pct_ltm                                            AS feat_roa_ltm,
       gross_profit_margin_pct_ltm                                             AS feat_gpm_ltm,
       beta_5y                                                                 AS feat_beta_5y,
       calc_change_ratio(market_cap, market_cap_neg1fy)                        AS feat_mcap_trend_1y,
       safe_divide(market_cap, market_cap_3yavg)                               AS feat_mcap_vs_3yavg,
       safe_divide(enterprise_value, enterprise_value_3yavg)                   AS feat_ev_vs_3yavg
FROM pml_df
;

alter materialized view mv_pymc_dcf_pt owner to postgres
;

create unique index idx_mv_pymc_dcf_pt_isin
	on mv_pymc_dcf_pt (isin)
;