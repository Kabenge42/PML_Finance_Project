CREATE MATERIALIZED VIEW mv_pymc_credit_risk AS
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
       altman_z_score_ltm                                                             AS observed_altman_z,
       pml.altman_zone(altman_z_score_ltm)                                            AS feat_distress_zone,
       pml.calc_change_ratio(altman_z_score_ltm, altman_z_score_neg1fy)               AS feat_z_trend_1y,
       pml.calc_change_ratio(altman_z_score_ltm, altman_z_score_neg3fy)               AS feat_z_trend_3y,
       pml.safe_divide(cfo_ltm, NULLIF(capital_expenditure_ltm, 0::double precision)) AS feat_cfo_capex_cov,
       pml.safe_divide(fcf_ltm, enterprise_value)                                     AS feat_fcf_yield,
       pml.safe_divide(cff_ltm, enterprise_value)                                     AS feat_cff_to_ev,
       pml.safe_divide(issuance_common_stock_ltm - repurchase_common_stock_ltm,
                       NULLIF(market_cap, 0::double precision))                       AS feat_net_equity_issuance,
       pml.calc_change_ratio(full_time_employees_fy, full_time_employees_neg1fy)      AS feat_employee_growth_1y,
       p_b_ltm                                                                        AS feat_pb_ltm,
       beta_2y                                                                        AS feat_beta_2y,
       volatility_6m                                                                  AS feat_vol_6m,
       volatility_1y                                                                  AS feat_vol_1y,
       pml.calc_change_ratio(market_cap, market_cap_neg1fy)                           AS feat_mcap_trend_1y,
       pml.safe_divide(market_cap, market_cap_3yavg)                                  AS feat_mcap_vs_3yavg,
       pml.safe_divide(enterprise_value, enterprise_value_3yavg)                      AS feat_ev_vs_3yavg
FROM pml.pml_df;

ALTER MATERIALIZED VIEW mv_pymc_credit_risk OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_pymc_credit_risk_isin ON mv_pymc_credit_risk (isin);