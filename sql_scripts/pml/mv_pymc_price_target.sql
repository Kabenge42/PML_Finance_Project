CREATE MATERIALIZED VIEW mv_pymc_price_target AS
SELECT isin,
       ticker,
       trading_region,
       country,
       trading_country,
       exchange,
       unit,
       style_class,
       size_class,
       sector,
       industry,
       income_statement_report_date,
       next_earnings,
       next_earnings_when,
       next_earnings_status,
       fy_end_date,
       next_income_statement_report_date,
       next_fy_end_date,
       expected_report_date,
       next_earnings - CURRENT_DATE                                                                       AS days_to_next_earnings,
       CURRENT_DATE - income_statement_report_date                                                        AS days_since_last_report,
       next_fy_end_date - CURRENT_DATE                                                                    AS days_to_next_fy_end,
       next_income_statement_report_date - CURRENT_DATE                                                   AS days_to_next_report,
       expected_report_date - CURRENT_DATE                                                                AS days_to_expected_report,
       fy_end_date - CURRENT_DATE                                                                         AS days_since_fy_end,
       target_pct_avg                                                                                     AS observed_target_pct,
       target_pct_med                                                                                     AS observed_target_pct_med,
       price_target,
       price_target_median,
       price_target_low,
       price_target_high,
       price_target_stddev,
       last_price,
       total_return_ytd,
       price_target_num                                                                                   AS n_analysts,
       num_strong_buys_ratings + num_buys_ratings - num_sell_ratings -
       num_strong_sell_ratings                                                                            AS feat_net_buy_sentiment,
       pml.safe_divide((num_strong_buys_ratings + num_strong_sell_ratings)::numeric, NULLIF(
		       num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_sell_ratings +
		       num_strong_sell_ratings + num_no_opinion_ratings,
		       0)::numeric)                                                                               AS feat_conviction_ratio,
       num_hold_ratings                                                                                   AS feat_holds,
       num_strong_buys_ratings + num_buys_ratings                                                         AS feat_buys,
       num_strong_sell_ratings + num_sell_ratings                                                         AS feat_sells,
       num_no_opinion_ratings                                                                             AS feat_no_opinion,
       pml.safe_divide((num_strong_buys_ratings + num_buys_ratings)::numeric,
                       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
                        num_sell_ratings +
                        num_strong_sell_ratings)::numeric)                                                AS feat_analyst_bullish_pct,
       pml.safe_divide((num_sell_ratings + num_strong_sell_ratings)::numeric,
                       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
                        num_sell_ratings +
                        num_strong_sell_ratings)::numeric)                                                AS feat_analyst_bearish_pct,
       pml.safe_divide(num_hold_ratings::numeric,
                       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
                        num_sell_ratings +
                        num_strong_sell_ratings)::numeric)                                                AS feat_analyst_neutral_pct,
       abs(pml.safe_divide(
		       (num_strong_buys_ratings + num_buys_ratings - (num_sell_ratings + num_strong_sell_ratings))::numeric,
		       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
		        num_sell_ratings +
		        num_strong_sell_ratings)::numeric))                                                       AS feat_analyst_conviction,
       pml.calc_change_ratio(price_target::numeric, last_price::numeric)                                  AS feat_implied_upside,
       pml.calc_change_ratio(target_pct_high::numeric,
                             target_pct_low::numeric)                                                     AS feat_target_range_width,
       pml.calc_change_ratio(price_target::numeric,
                             price_target_3m_ago::numeric)                                                AS feat_pt_momentum_3m,
       pml.calc_change_ratio(price_target_num::numeric,
                             price_target_num_3m_ago::numeric)                                            AS feat_coverage_change_3m,
       pml.coef_var(price_target::numeric, price_target_stddev::numeric)                                  AS feat_target_dispersion_cv,
       pml.safe_divide(last_price - w_52low_adj,
                       NULLIF(w_52high_adj - w_52low_adj, 0::double precision))                           AS feat_52w_range_position,
       p_e_ntm                                                                                            AS feat_pe_ntm,
       ev_ebitda_ntm                                                                                      AS feat_ev_ebitda_ntm,
       volatility_3m                                                                                      AS feat_vol_3m,
       analyst_rating                                                                                     AS feat_analyst_rating,
       CASE
	       WHEN price_target_1y_ago > 0::double precision AND last_price >= price_target_1y_ago
		       THEN 1.0::double precision
	       WHEN price_target_1y_ago > 0::double precision THEN pml.safe_divide(last_price, price_target_1y_ago)
	       ELSE NULL::double precision END                                                                AS feat_pt_achievement_1y,
       pml.safe_divide(abs(last_price - price_target_1y_ago),
                       abs(price_target_1y_ago))                                                          AS feat_pt_accuracy_1y,
       pml.safe_divide(price_target_1y_ago - last_price,
                       abs(price_target_1y_ago))                                                          AS feat_pt_optimism_bias,
       CASE
	       WHEN last_price >= price_target_low_1y_ago AND last_price <= price_target_high_1y_ago THEN 1.0
	       ELSE 0.0 END                                                                                   AS feat_pt_range_hit_rate,
       pml.safe_divide(price_target - price_target_median,
                       price_target_median)                                                               AS feat_pt_median_vs_mean_spread,
       pml.safe_divide(price_target_high - price_target_low, price_target_median) -
       pml.safe_divide(price_target_high_1y_ago - price_target_low_1y_ago,
                       price_target_median_1y_ago)                                                        AS feat_pt_high_low_convergence_1y,
       pml.safe_divide(price_target_num, (price_target_num_1y_ago + price_target_num_6m_ago + price_target_num_3m_ago) /
                                         3.0::double precision)                                           AS feat_analyst_count_stability,
       pml.calc_change_ratio(market_cap, market_cap_neg1fy)                                               AS feat_mcap_trend_1y,
       pml.safe_divide(market_cap, market_cap_3yavg)                                                      AS feat_mcap_vs_3yavg,
       pml.safe_divide(enterprise_value, enterprise_value_3yavg)                                          AS feat_ev_vs_3yavg
FROM pml.pml_df;

ALTER MATERIALIZED VIEW mv_pymc_price_target OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_pymc_price_target_isin ON mv_pymc_price_target (isin);