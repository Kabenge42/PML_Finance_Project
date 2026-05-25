CREATE MATERIALIZED VIEW pml.mv_pymc_credit_risk AS
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
       altman_z_score_ltm                                                                                            AS observed_altman_z,
       altman_zone(altman_z_score_ltm)                                                                               AS feat_distress_zone,
       calc_change_ratio(altman_z_score_ltm, altman_z_score_neg1fy)                                                  AS feat_z_trend_1y,
       calc_change_ratio(altman_z_score_ltm, altman_z_score_neg3fy)                                                  AS feat_z_trend_3y,
       safe_divide(cfo_ltm,
                   NULLIF(capital_expenditure_ltm, 0::double precision))                                             AS feat_cfo_capex_cov,
       safe_divide(fcf_ltm, enterprise_value)                                                                        AS feat_fcf_yield,
       safe_divide(cff_ltm, enterprise_value)                                                                        AS feat_cff_to_ev,
       safe_divide(issuance_common_stock_ltm - repurchase_common_stock_ltm,
                   NULLIF(market_cap, 0::double precision))                                                          AS feat_net_equity_issuance,
       calc_change_ratio(full_time_employees_fy, full_time_employees_neg1fy)                                         AS feat_employee_growth_1y,
       p_b_ltm                                                                                                       AS feat_pb_ltm,
       beta_2y                                                                                                       AS feat_beta_2y,
       volatility_6m                                                                                                 AS feat_vol_6m,
       volatility_1y                                                                                                 AS feat_vol_1y
FROM pml_df;

ALTER MATERIALIZED VIEW pml.mv_pymc_credit_risk OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_pymc_credit_risk_isin ON pml.mv_pymc_credit_risk (isin);