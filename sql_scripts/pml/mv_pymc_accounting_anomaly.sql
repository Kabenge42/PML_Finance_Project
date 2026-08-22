create materialized view pml.mv_pymc_accounting_anomaly
as
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
       eps_adj_ltm                                                                                  AS observed_eps_adj,
       accruals_ratio(net_eps_basic_ltm * shrs_out, cfo_ltm, enterprise_value)                      AS feat_accruals_ratio,
       calc_change_ratio(gross_profit_margin_pct_ltm,
                         gross_profit_margin_pct_neg1fy)                                            AS feat_gpm_change_1y,
       calc_change_ratio(sales_neg0fyactual, sales_neg1fyactual)                                    AS feat_sales_growth_1y,
       calc_change_ratio(ebit_neg0fyactual, ebit_neg1fyactual)                                      AS feat_ebit_growth_1y,
       calc_change_ratio(ebitda_neg0fyactual, ebitda_neg1fyactual)                                  AS feat_ebitda_growth_1y,
       safe_divide(capital_expenditure_ltm, cfo_ltm)                                                AS feat_capex_intensity,
       safe_divide(eps_adj_ltm - net_eps_basic_ltm, NULLIF(net_eps_basic_ltm, 0::double precision)) AS feat_eps_adj_gap,
       safe_divide(cfi_ltm, NULLIF(cfo_ltm, 0::double precision))                                   AS feat_cfi_to_cfo,
       safe_divide(cff_ltm, NULLIF(cfo_ltm, 0::double precision))                                   AS feat_cff_to_cfo,
       calc_change_ratio(shrs_out, shrs_out_neg1fy)                                                 AS feat_share_inflation_1y,
       safe_divide(issuance_common_stock_ltm,
                   NULLIF(market_cap, 0::double precision))                                         AS feat_issuance_intensity,
       calc_change_ratio(full_time_employees_fy, full_time_employees_neg1fy)                        AS feat_employee_growth_1y,
       calc_change_ratio(fcf_per_share_ltm, net_eps_basic_ltm)                                      AS feat_fcfps_vs_eps_gap,
       peg_ntm                                                                                      AS feat_peg_ntm,
       calc_change_ratio(market_cap, market_cap_neg1fy)                                             AS feat_mcap_trend_1y,
       safe_divide(market_cap, market_cap_3yavg)                                                    AS feat_mcap_vs_3yavg,
       safe_divide(enterprise_value, enterprise_value_3yavg)                                        AS feat_ev_vs_3yavg
FROM pml_df
;

alter materialized view pml.mv_pymc_accounting_anomaly owner to postgres
;

create unique index idx_mv_pymc_accounting_anomaly_isin
	on pml.mv_pymc_accounting_anomaly (isin)
;