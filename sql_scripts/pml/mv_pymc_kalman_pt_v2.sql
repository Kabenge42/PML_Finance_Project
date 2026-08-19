create materialized view pml.mv_pymc_kalman_pt_v2
as
WITH base AS (SELECT mv_pymc_kalman_pt.isin,
                     mv_pymc_kalman_pt.ticker,
                     mv_pymc_kalman_pt.name,
                     mv_pymc_kalman_pt.trading_region,
                     mv_pymc_kalman_pt.region,
                     mv_pymc_kalman_pt.country,
                     mv_pymc_kalman_pt.country_name,
                     mv_pymc_kalman_pt.trading_country,
                     mv_pymc_kalman_pt.trading_country_name,
                     mv_pymc_kalman_pt.exchange,
                     mv_pymc_kalman_pt.exchange_name,
                     mv_pymc_kalman_pt.unit,
                     mv_pymc_kalman_pt.unit_name,
                     mv_pymc_kalman_pt.style_class,
                     mv_pymc_kalman_pt.size_class,
                     mv_pymc_kalman_pt.sector,
                     mv_pymc_kalman_pt.industry,
                     mv_pymc_kalman_pt.last_updated,
                     mv_pymc_kalman_pt.income_statement_report_date,
                     mv_pymc_kalman_pt.next_earnings,
                     mv_pymc_kalman_pt.next_earnings_when,
                     mv_pymc_kalman_pt.next_earnings_status,
                     mv_pymc_kalman_pt.fy_end_date,
                     mv_pymc_kalman_pt.next_fiscal_quarter,
                     mv_pymc_kalman_pt.next_income_statement_report_date,
                     mv_pymc_kalman_pt.next_fy_end_date,
                     mv_pymc_kalman_pt.expected_report_date,
                     mv_pymc_kalman_pt.days_to_next_earnings,
                     mv_pymc_kalman_pt.days_since_last_report,
                     mv_pymc_kalman_pt.days_to_next_fy_end,
                     mv_pymc_kalman_pt.days_to_next_fiscal_quarter,
                     mv_pymc_kalman_pt.days_to_next_report,
                     mv_pymc_kalman_pt.days_to_expected_report,
                     mv_pymc_kalman_pt.days_since_fy_end,
                     mv_pymc_kalman_pt.market_cap,
                     mv_pymc_kalman_pt.enterprise_value,
                     mv_pymc_kalman_pt.market_cap_global_r,
                     mv_pymc_kalman_pt.market_cap_global_sec_r,
                     mv_pymc_kalman_pt.market_cap_region_r,
                     mv_pymc_kalman_pt.market_cap_region_sec_r,
                     mv_pymc_kalman_pt.market_cap_country_r,
                     mv_pymc_kalman_pt.market_cap_country_sec_r,
                     mv_pymc_kalman_pt.feat_rel_volume,
                     mv_pymc_kalman_pt.observed_pt,
                     mv_pymc_kalman_pt.last_price,
                     mv_pymc_kalman_pt.price_target_median,
                     mv_pymc_kalman_pt.price_target_high,
                     mv_pymc_kalman_pt.price_target_low,
                     mv_pymc_kalman_pt.n_analysts,
                     mv_pymc_kalman_pt.price_target_1w_ago,
                     mv_pymc_kalman_pt.price_target_mtd_ago,
                     mv_pymc_kalman_pt.price_target_1m_ago,
                     mv_pymc_kalman_pt.price_target_qtd_ago,
                     mv_pymc_kalman_pt.price_target_3m_ago,
                     mv_pymc_kalman_pt.price_target_6m_ago,
                     mv_pymc_kalman_pt.price_target_ytd_ago,
                     mv_pymc_kalman_pt.price_target_1y_ago,
                     mv_pymc_kalman_pt.price_target_low_1w_ago,
                     mv_pymc_kalman_pt.price_target_low_mtd_ago,
                     mv_pymc_kalman_pt.price_target_low_1m_ago,
                     mv_pymc_kalman_pt.price_target_low_qtd_ago,
                     mv_pymc_kalman_pt.price_target_low_3m_ago,
                     mv_pymc_kalman_pt.price_target_low_6m_ago,
                     mv_pymc_kalman_pt.price_target_low_ytd_ago,
                     mv_pymc_kalman_pt.price_target_low_1y_ago,
                     mv_pymc_kalman_pt.price_target_high_1w_ago,
                     mv_pymc_kalman_pt.price_target_high_mtd_ago,
                     mv_pymc_kalman_pt.price_target_high_1m_ago,
                     mv_pymc_kalman_pt.price_target_high_qtd_ago,
                     mv_pymc_kalman_pt.price_target_high_3m_ago,
                     mv_pymc_kalman_pt.price_target_high_6m_ago,
                     mv_pymc_kalman_pt.price_target_high_ytd_ago,
                     mv_pymc_kalman_pt.price_target_high_1y_ago,
                     mv_pymc_kalman_pt.price_target_median_1w_ago,
                     mv_pymc_kalman_pt.price_target_median_mtd_ago,
                     mv_pymc_kalman_pt.price_target_median_1m_ago,
                     mv_pymc_kalman_pt.price_target_median_qtd_ago,
                     mv_pymc_kalman_pt.price_target_median_3m_ago,
                     mv_pymc_kalman_pt.price_target_median_6m_ago,
                     mv_pymc_kalman_pt.price_target_median_ytd_ago,
                     mv_pymc_kalman_pt.price_target_median_1y_ago,
                     mv_pymc_kalman_pt.price_target_num_1w_ago,
                     mv_pymc_kalman_pt.price_target_num_mtd_ago,
                     mv_pymc_kalman_pt.price_target_num_1m_ago,
                     mv_pymc_kalman_pt.price_target_num_qtd_ago,
                     mv_pymc_kalman_pt.price_target_num_3m_ago,
                     mv_pymc_kalman_pt.price_target_num_6m_ago,
                     mv_pymc_kalman_pt.price_target_num_ytd_ago,
                     mv_pymc_kalman_pt.price_target_num_1y_ago,
                     mv_pymc_kalman_pt.price_1d_ago,
                     mv_pymc_kalman_pt.price_5d_ago,
                     mv_pymc_kalman_pt.price_1w_ago,
                     mv_pymc_kalman_pt.price_mtd_ago,
                     mv_pymc_kalman_pt.price_1m_ago,
                     mv_pymc_kalman_pt.price_3m_ago,
                     mv_pymc_kalman_pt.price_6m_ago,
                     mv_pymc_kalman_pt.price_ytd_ago,
                     mv_pymc_kalman_pt.price_1y_ago,
                     mv_pymc_kalman_pt.price_3y_ago,
                     mv_pymc_kalman_pt.price_5y_ago,
                     mv_pymc_kalman_pt.price_qtd_ago,
                     mv_pymc_kalman_pt.feat_implied_upside,
                     mv_pymc_kalman_pt.feat_holds,
                     mv_pymc_kalman_pt.feat_buys,
                     mv_pymc_kalman_pt.feat_sells,
                     mv_pymc_kalman_pt.feat_no_opinion,
                     mv_pymc_kalman_pt.feat_analyst_bullish_pct,
                     mv_pymc_kalman_pt.feat_analyst_bearish_pct,
                     mv_pymc_kalman_pt.feat_analyst_neutral_pct,
                     mv_pymc_kalman_pt.feat_analyst_conviction,
                     mv_pymc_kalman_pt.feat_analyst_rating,
                     mv_pymc_kalman_pt.feat_pt_achievement_1y,
                     mv_pymc_kalman_pt.feat_pt_accuracy_1y,
                     mv_pymc_kalman_pt.feat_pt_range_hit_rate,
                     mv_pymc_kalman_pt.feat_pt_drift,
                     mv_pymc_kalman_pt.feat_pt_drift_n,
                     mv_pymc_kalman_pt.feat_price_drift,
                     mv_pymc_kalman_pt.feat_price_drift_n,
                     mv_pymc_kalman_pt.feat_pt_high_drift,
                     mv_pymc_kalman_pt.feat_pt_low_drift,
                     mv_pymc_kalman_pt.feat_pt_median_drift,
                     mv_pymc_kalman_pt.feat_coverage_drift,
                     mv_pymc_kalman_pt.feat_pt_noise_drift,
                     mv_pymc_kalman_pt.feat_pt_noise_sigma,
                     mv_pymc_kalman_pt.price_target_stddev_1w_ago,
                     mv_pymc_kalman_pt.price_target_stddev_mtd_ago,
                     mv_pymc_kalman_pt.price_target_stddev_1m_ago,
                     mv_pymc_kalman_pt.price_target_stddev_qtd_ago,
                     mv_pymc_kalman_pt.price_target_stddev_3m_ago,
                     mv_pymc_kalman_pt.price_target_stddev_6m_ago,
                     mv_pymc_kalman_pt.price_target_stddev_ytd_ago,
                     mv_pymc_kalman_pt.price_target_stddev_1y_ago,
                     mv_pymc_kalman_pt.feat_pt_range_norm,
                     mv_pymc_kalman_pt.feat_one_day_return,
                     mv_pymc_kalman_pt.feat_price_chg_pct_3m,
                     mv_pymc_kalman_pt.feat_vol_drift,
                     mv_pymc_kalman_pt.feat_vol_drift_n,
                     mv_pymc_kalman_pt.feat_vol_level,
                     mv_pymc_kalman_pt.feat_log_mcap,
                     mv_pymc_kalman_pt.beta_1y,
                     mv_pymc_kalman_pt.beta_2y,
                     mv_pymc_kalman_pt.beta_5y,
                     mv_pymc_kalman_pt.feat_avg_beta,
                     mv_pymc_kalman_pt.feat_total_return_ytd,
                     mv_pymc_kalman_pt.feat_total_return_5y,
                     mv_pymc_kalman_pt.feat_total_return_10y,
                     mv_pymc_kalman_pt.feat_tr_cagr_3y,
                     mv_pymc_kalman_pt.feat_tr_cagr_10y,
                     mv_pymc_kalman_pt.feat_tr_cagr_5y,
                     mv_pymc_kalman_pt.feat_tr_cagr_1y,
                     mv_pymc_kalman_pt.feat_total_return_1d,
                     mv_pymc_kalman_pt.feat_total_return_5d,
                     mv_pymc_kalman_pt.feat_total_return_1w,
                     mv_pymc_kalman_pt.feat_total_return_1m,
                     mv_pymc_kalman_pt.feat_total_return_3m,
                     mv_pymc_kalman_pt.feat_total_return_6m,
                     mv_pymc_kalman_pt.feat_total_return_1y,
                     mv_pymc_kalman_pt.feat_total_return_3y,
                     mv_pymc_kalman_pt.feat_total_return_mtd,
                     mv_pymc_kalman_pt.feat_total_return_qtd,
                     mv_pymc_kalman_pt.feat_total_return_2025,
                     mv_pymc_kalman_pt.feat_total_return_2024,
                     mv_pymc_kalman_pt.feat_total_return_2023,
                     mv_pymc_kalman_pt.feat_total_return_2022,
                     mv_pymc_kalman_pt.feat_total_return_2021,
                     mv_pymc_kalman_pt.feat_mcap_global_r,
                     mv_pymc_kalman_pt.feat_mcap_global_sec_r,
                     mv_pymc_kalman_pt.feat_mcap_region_r,
                     mv_pymc_kalman_pt.feat_mcap_region_sec_r,
                     mv_pymc_kalman_pt.feat_mcap_country_r,
                     mv_pymc_kalman_pt.feat_mcap_country_sec_r,
                     mv_pymc_kalman_pt.feat_net_eps_drift,
                     mv_pymc_kalman_pt.feat_net_eps_drift_n,
                     mv_pymc_kalman_pt.feat_last_q_surprise,
                     mv_pymc_kalman_pt.feat_last_y_surprise,
                     mv_pymc_kalman_pt.feat_eps_beat_rate,
                     mv_pymc_kalman_pt.feat_eps_beat_rate_annual,
                     mv_pymc_kalman_pt.feat_piotroski_f_score_fy,
                     mv_pymc_kalman_pt.feat_piotroski_f_score_neg1fy,
                     mv_pymc_kalman_pt.feat_piotroski_f_score_neg2fy,
                     mv_pymc_kalman_pt.feat_piotroski_f_score_neg3fy,
                     mv_pymc_kalman_pt.feat_median_piotroski_f_score
              FROM pml.mv_pymc_kalman_pt),
     uplift AS (SELECT b_1.isin,
                       CASE
	                       WHEN pml.safe_divide(b_1.observed_pt, b_1.last_price) > 0::double precision
		                       THEN ln(pml.safe_divide(b_1.observed_pt, b_1.last_price))
		                       ELSE NULL::double precision
	                       END AS lu_now,
                       CASE
	                       WHEN pml.safe_divide(b_1.price_target_1w_ago, b_1.price_1w_ago) > 0::double precision
		                       THEN ln(pml.safe_divide(b_1.price_target_1w_ago, b_1.price_1w_ago))
		                       ELSE NULL::double precision
	                       END AS lu_1w,
                       CASE
	                       WHEN pml.safe_divide(b_1.price_target_1m_ago, b_1.price_1m_ago) > 0::double precision
		                       THEN ln(pml.safe_divide(b_1.price_target_1m_ago, b_1.price_1m_ago))
		                       ELSE NULL::double precision
	                       END AS lu_1m,
                       CASE
	                       WHEN pml.safe_divide(b_1.price_target_3m_ago, b_1.price_3m_ago) > 0::double precision
		                       THEN ln(pml.safe_divide(b_1.price_target_3m_ago, b_1.price_3m_ago))
		                       ELSE NULL::double precision
	                       END AS lu_3m,
                       CASE
	                       WHEN pml.safe_divide(b_1.price_target_6m_ago, b_1.price_6m_ago) > 0::double precision
		                       THEN ln(pml.safe_divide(b_1.price_target_6m_ago, b_1.price_6m_ago))
		                       ELSE NULL::double precision
	                       END AS lu_6m,
                       CASE
	                       WHEN pml.safe_divide(b_1.price_target_1y_ago, b_1.price_1y_ago) > 0::double precision
		                       THEN ln(pml.safe_divide(b_1.price_target_1y_ago, b_1.price_1y_ago))
		                       ELSE NULL::double precision
	                       END AS lu_1y
                FROM base b_1),
     eps AS (SELECT b_1.isin,
                    (SELECT avg(v.v) AS avg
                     FROM unnest(ARRAY [b_1.feat_last_q_surprise / 100.0::double precision, b_1.feat_last_y_surprise / 100.0::double precision]) v(v)
                     WHERE v.v IS NOT NULL) AS eps_surprise,
                    (SELECT avg(v.v) AS avg
                     FROM unnest(ARRAY [b_1.feat_eps_beat_rate, b_1.feat_eps_beat_rate_annual]) v(v)
                     WHERE v.v IS NOT NULL) AS eps_beat,
                    (SELECT count(v.v)::double precision / 5.0::double precision
                     FROM unnest(ARRAY [b_1.feat_net_eps_drift, b_1.feat_last_q_surprise, b_1.feat_last_y_surprise, b_1.feat_eps_beat_rate, b_1.feat_eps_beat_rate_annual]) v(v)
                     WHERE v.v IS NOT NULL) AS eps_coverage
             FROM base b_1)
SELECT b.isin,
       b.ticker,
       b.name,
       b.trading_region,
       b.region,
       b.country,
       b.country_name,
       b.trading_country,
       b.trading_country_name,
       b.exchange,
       b.exchange_name,
       b.unit,
       b.unit_name,
       b.style_class,
       b.size_class,
       b.sector,
       b.industry,
       b.last_updated,
       b.income_statement_report_date,
       b.next_earnings,
       b.next_earnings_when,
       b.next_earnings_status,
       b.fy_end_date,
       b.next_fiscal_quarter,
       b.next_income_statement_report_date,
       b.next_fy_end_date,
       b.expected_report_date,
       b.days_to_next_earnings,
       b.days_since_last_report,
       b.days_to_next_fy_end,
       b.days_to_next_fiscal_quarter,
       b.days_to_next_report,
       b.days_to_expected_report,
       b.days_since_fy_end,
       b.market_cap,
       b.enterprise_value,
       b.market_cap_global_r,
       b.market_cap_global_sec_r,
       b.market_cap_region_r,
       b.market_cap_region_sec_r,
       b.market_cap_country_r,
       b.market_cap_country_sec_r,
       b.feat_rel_volume,
       b.observed_pt,
       b.last_price,
       b.price_target_median,
       b.price_target_high,
       b.price_target_low,
       b.n_analysts,
       b.price_target_1w_ago,
       b.price_target_mtd_ago,
       b.price_target_1m_ago,
       b.price_target_qtd_ago,
       b.price_target_3m_ago,
       b.price_target_6m_ago,
       b.price_target_ytd_ago,
       b.price_target_1y_ago,
       b.price_target_low_1w_ago,
       b.price_target_low_mtd_ago,
       b.price_target_low_1m_ago,
       b.price_target_low_qtd_ago,
       b.price_target_low_3m_ago,
       b.price_target_low_6m_ago,
       b.price_target_low_ytd_ago,
       b.price_target_low_1y_ago,
       b.price_target_high_1w_ago,
       b.price_target_high_mtd_ago,
       b.price_target_high_1m_ago,
       b.price_target_high_qtd_ago,
       b.price_target_high_3m_ago,
       b.price_target_high_6m_ago,
       b.price_target_high_ytd_ago,
       b.price_target_high_1y_ago,
       b.price_target_median_1w_ago,
       b.price_target_median_mtd_ago,
       b.price_target_median_1m_ago,
       b.price_target_median_qtd_ago,
       b.price_target_median_3m_ago,
       b.price_target_median_6m_ago,
       b.price_target_median_ytd_ago,
       b.price_target_median_1y_ago,
       b.price_target_num_1w_ago,
       b.price_target_num_mtd_ago,
       b.price_target_num_1m_ago,
       b.price_target_num_qtd_ago,
       b.price_target_num_3m_ago,
       b.price_target_num_6m_ago,
       b.price_target_num_ytd_ago,
       b.price_target_num_1y_ago,
       b.price_1d_ago,
       b.price_5d_ago,
       b.price_1w_ago,
       b.price_mtd_ago,
       b.price_1m_ago,
       b.price_3m_ago,
       b.price_6m_ago,
       b.price_ytd_ago,
       b.price_1y_ago,
       b.price_3y_ago,
       b.price_5y_ago,
       b.price_qtd_ago,
       b.feat_implied_upside,
       b.feat_holds,
       b.feat_buys,
       b.feat_sells,
       b.feat_no_opinion,
       b.feat_analyst_bullish_pct,
       b.feat_analyst_bearish_pct,
       b.feat_analyst_neutral_pct,
       b.feat_analyst_conviction,
       b.feat_analyst_rating,
       b.feat_pt_achievement_1y,
       b.feat_pt_accuracy_1y,
       b.feat_pt_range_hit_rate,
       b.feat_pt_drift,
       b.feat_pt_drift_n,
       b.feat_price_drift,
       b.feat_price_drift_n,
       b.feat_pt_high_drift,
       b.feat_pt_low_drift,
       b.feat_pt_median_drift,
       b.feat_coverage_drift,
       b.feat_pt_noise_drift,
       b.feat_pt_noise_sigma,
       b.price_target_stddev_1w_ago,
       b.price_target_stddev_mtd_ago,
       b.price_target_stddev_1m_ago,
       b.price_target_stddev_qtd_ago,
       b.price_target_stddev_3m_ago,
       b.price_target_stddev_6m_ago,
       b.price_target_stddev_ytd_ago,
       b.price_target_stddev_1y_ago,
       b.feat_pt_range_norm,
       b.feat_one_day_return,
       b.feat_price_chg_pct_3m,
       b.feat_vol_drift,
       b.feat_vol_drift_n,
       b.feat_vol_level,
       b.feat_log_mcap,
       b.beta_1y,
       b.beta_2y,
       b.beta_5y,
       b.feat_avg_beta,
       b.feat_total_return_ytd,
       b.feat_total_return_5y,
       b.feat_total_return_10y,
       b.feat_tr_cagr_3y,
       b.feat_tr_cagr_10y,
       b.feat_tr_cagr_5y,
       b.feat_tr_cagr_1y,
       b.feat_total_return_1d,
       b.feat_total_return_5d,
       b.feat_total_return_1w,
       b.feat_total_return_1m,
       b.feat_total_return_3m,
       b.feat_total_return_6m,
       b.feat_total_return_1y,
       b.feat_total_return_3y,
       b.feat_total_return_mtd,
       b.feat_total_return_qtd,
       b.feat_total_return_2025,
       b.feat_total_return_2024,
       b.feat_total_return_2023,
       b.feat_total_return_2022,
       b.feat_total_return_2021,
       b.feat_mcap_global_r,
       b.feat_mcap_global_sec_r,
       b.feat_mcap_region_r,
       b.feat_mcap_region_sec_r,
       b.feat_mcap_country_r,
       b.feat_mcap_country_sec_r,
       b.feat_net_eps_drift,
       b.feat_net_eps_drift_n,
       b.feat_last_q_surprise,
       b.feat_last_y_surprise,
       b.feat_eps_beat_rate,
       b.feat_eps_beat_rate_annual,
       b.feat_piotroski_f_score_fy,
       b.feat_piotroski_f_score_neg1fy,
       b.feat_piotroski_f_score_neg2fy,
       b.feat_piotroski_f_score_neg3fy,
       b.feat_median_piotroski_f_score,
       u.lu_now                                        AS feat_log_uplift_now,
       u.lu_1w                                         AS feat_log_uplift_1w,
       u.lu_1m                                         AS feat_log_uplift_1m,
       u.lu_3m                                         AS feat_log_uplift_3m,
       u.lu_6m                                         AS feat_log_uplift_6m,
       u.lu_1y                                         AS feat_log_uplift_1y,
       CASE
	       WHEN u.lu_now IS NOT NULL THEN 1
	                                 ELSE 0
	       END +
       CASE
	       WHEN u.lu_1w IS NOT NULL THEN 1
	                                ELSE 0
	       END +
       CASE
	       WHEN u.lu_1m IS NOT NULL THEN 1
	                                ELSE 0
	       END +
       CASE
	       WHEN u.lu_3m IS NOT NULL THEN 1
	                                ELSE 0
	       END +
       CASE
	       WHEN u.lu_6m IS NOT NULL THEN 1
	                                ELSE 0
	       END +
       CASE
	       WHEN u.lu_1y IS NOT NULL THEN 1
	                                ELSE 0
	       END                                         AS n_trail_obs,
       -- trail_days_{now,1w,1m,3m,6m,1y} were emitted here as SQL literals until
       -- 2026-08-19: 0/7/30/91/182/365, identical on every row, so zero
       -- information stored once per name -- and the model never read them, it
       -- built the same grid from DEFAULT_LOOKBACK_DAYS in Python. The OU
       -- kernel's x-axis now has one home, pml.vw_pymc_trail_days
       -- (pml_feature_catalogue.sql), which maps each offset to the
       -- feat_log_uplift_* column it describes and is tied to this MV by
       -- pml.assert_pymc_trail_days_map(). Their metadata rows are retired in
       -- pml_df_metadata_populate.sql section 7l -- dropping the columns without
       -- that de-registration raises PHANTOM_CATALOGUE_ALIAS.
       b.price_target_num_1w_ago                       AS n_analysts_1w,
       b.price_target_num_1m_ago                       AS n_analysts_1m,
       b.price_target_num_3m_ago                       AS n_analysts_3m,
       b.price_target_num_6m_ago                       AS n_analysts_6m,
       b.price_target_num_1y_ago                       AS n_analysts_1y,
       e.eps_surprise                                  AS feat_eps_signal_surprise,
       e.eps_beat                                      AS feat_eps_signal_beat,
       COALESCE(e.eps_coverage, 0.0::double precision) AS feat_eps_signal_coverage,
       now()                                           AS built_at
FROM base            b
	     JOIN uplift u ON u.isin = b.isin
	     JOIN eps    e ON e.isin = b.isin
WHERE u.lu_now IS NOT NULL
;

comment on materialized view pml.mv_pymc_kalman_pt_v2 is 'v2 Kalman price-target feature matrix. Derived from mv_pymc_kalman_pt; adds the log-uplift response trail, its calendar offsets, per-lookback analyst coverage and a consolidated EPS block. Refresh AFTER mv_pymc_kalman_pt.'
;

comment on column pml.mv_pymc_kalman_pt_v2.feat_log_uplift_now is 'ln(price_target / last_price). Raw decimal log ratio. The snapshot response and the anchor (offset 0) of the OU time grid.'
;

comment on column pml.mv_pymc_kalman_pt_v2.n_trail_obs is 'Count of non-NULL trail cells, 1..6. The per-name T actually contributing to the likelihood; cells outside it are masked, not imputed.'
;

comment on column pml.mv_pymc_kalman_pt_v2.n_analysts_1y is 'Analyst count behind the 1y consensus (price_target_num_1y_ago). Per-cell measurement precision; v1 had one weight for all T.'
;

comment on column pml.mv_pymc_kalman_pt_v2.feat_eps_signal_surprise is 'Mean of whichever of feat_last_{q,y}_surprise is non-NULL, divided by 100 so it is a signed RAW DECIMAL like every other feat_ column. NULL when neither leg exists.'
;

comment on column pml.mv_pymc_kalman_pt_v2.feat_eps_signal_beat is 'Mean of whichever of feat_eps_beat_rate{,_annual} is non-NULL. A frequency in [0, 1] -- a different quantity from feat_eps_signal_surprise, which is a magnitude, which is why they are separate columns.'
;

comment on column pml.mv_pymc_kalman_pt_v2.feat_eps_signal_coverage is 'Share of all five EPS legs (net_eps_drift, last_{q,y}_surprise, eps_beat_rate{,_annual}) that were non-NULL, in [0, 1]. Lets the model distinguish an informative zero from an absent measurement.'
;

comment on column pml.mv_pymc_kalman_pt_v2.built_at is 'Refresh timestamp. The parent MV computes its days_* horizons against CURRENT_DATE, so this is what tells two refreshes apart.'
;

alter materialized view pml.mv_pymc_kalman_pt_v2 owner to postgres
;

create unique index idx_mv_pymc_kalman_pt_v2_isin
	on pml.mv_pymc_kalman_pt_v2 (isin)
;

create index idx_mv_pymc_kalman_pt_v2_trail
	on pml.mv_pymc_kalman_pt_v2 (n_trail_obs)
;