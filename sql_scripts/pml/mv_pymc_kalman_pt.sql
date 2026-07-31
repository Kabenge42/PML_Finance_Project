-- ---- 3. KalmanFilterPriceTarget ----------------------------------------------
-- The single-security GaussianRandomWalk filter reconstructs its time axis from
-- the embedded *_ago cohort. The fiscal-calendar date anchors + day-count
-- horizons below mirror mv_pymc_price_target so the notebook can derive the
-- *real* (irregular) elapsed-time spacing that the marginalized GRW uses to
-- scale its process-variance covariance kernel (min(tau_s, tau_t)) — see
-- KalmanFilterModel._build_marginalized_likelihood / _resolve_time_deltas.
-- target_drift is now computed for every price_* / price_target_* trail (mean,
-- high, low, median, raw price, analyst-count and dispersion) so the per-ISIN
-- snapshot exposes the full drift/state-transition signal set.
--
-- FUSED MvGRW PANEL: the cross-sectional fused model
-- (build_fused_kalman_pt_model) consumes these same MV columns in fused
-- state-space roles — feat_avg_beta (the NULL-aware mean of beta_{1y,2y,5y}) as
-- the SYSTEMATIC RISK (CAPM beta) driver that conditions
-- risk_adj_return = expected_return * exp(-risk_penalty * z(feat_avg_beta)),
-- feat_pt_noise_sigma as the cv that widens
-- sigma_isin = sigma_base * (1 + cv) / sqrt(n_analysts), and n_analysts as the
-- precision count. feat_vol_drift (drift across the realized-vol term structure
-- 1m -> 1y, mirroring feat_pt_noise_drift) is the observation-noise-drift
-- widener; the absolute feat_vol_{1m,3m,6m,1y} levels are no longer emitted.
CREATE MATERIALIZED VIEW mv_pymc_kalman_pt AS
SELECT isin,
       ticker,
       name,
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
       last_updated,
       -- ---- Fiscal-calendar anchors (raw DATE coords for the GRW time axis) ----
       income_statement_report_date,
       next_earnings,
       -- Low-cardinality categorical fiscal-calendar coords (encode upstream in PyMC)
       next_earnings_when,
       next_earnings_status,
       fy_end_date,
       next_fiscal_quarter,
       next_income_statement_report_date,
       next_fy_end_date,
       expected_report_date,
       -- Pre-computed numeric horizons (days). Feed pm.Data as the time-step
       -- deltas the marginalized GaussianRandomWalk scales innovations by.
       (next_earnings - CURRENT_DATE)::INT                                                                                                                                                                     AS days_to_next_earnings,
       (CURRENT_DATE - income_statement_report_date)::INT                                                                                                                                                      AS days_since_last_report,
       (next_fy_end_date - CURRENT_DATE)::INT                                                                                                                                                                  AS days_to_next_fy_end,
       (next_fiscal_quarter - CURRENT_DATE)::INT                                                                                                                                                                  AS days_to_next_fiscal_quarter,
       (next_income_statement_report_date - CURRENT_DATE)::INT                                                                                                                                                 AS days_to_next_report,
       (expected_report_date - CURRENT_DATE)::INT                                                                                                                                                              AS days_to_expected_report,
       (fy_end_date - CURRENT_DATE)::INT                                                                                                                                                                       AS days_since_fy_end,
       market_cap,
       enterprise_value,
       market_cap_country_r,
       price_target                                                                                                                                                                                            AS observed_pt,
       last_price,
       price_target_median,
       price_target_high,
       price_target_low,
       price_target_num                                                                                                                                                                                        AS n_analysts,
       -- ---- Lagged analyst-target trail (kalman_pt observed state sequence) -----
       -- Full multi-horizon *_ago snapshots the Kalman panel observes as the latent
       -- price-target state sequence (level / low / high / median). pymc_role
       -- 'observed' for kalman_pt in pml.vw_pymc_feature_catalogue; emitted
       -- un-prefixed so feature_alias (== column_name) resolves against
       -- kalman_df.columns in the notebook present-check.
       price_target_1w_ago,
       price_target_mtd_ago,
       price_target_1m_ago,
       price_target_qtd_ago,
       price_target_3m_ago,
       price_target_6m_ago,
       price_target_ytd_ago,
       price_target_1y_ago,
       price_target_low_1w_ago,
       price_target_low_mtd_ago,
       price_target_low_1m_ago,
       price_target_low_qtd_ago,
       price_target_low_3m_ago,
       price_target_low_6m_ago,
       price_target_low_ytd_ago,
       price_target_low_1y_ago,
       price_target_high_1w_ago,
       price_target_high_mtd_ago,
       price_target_high_1m_ago,
       price_target_high_qtd_ago,
       price_target_high_3m_ago,
       price_target_high_6m_ago,
       price_target_high_ytd_ago,
       price_target_high_1y_ago,
       price_target_median_1w_ago,
       price_target_median_mtd_ago,
       price_target_median_1m_ago,
       price_target_median_qtd_ago,
       price_target_median_3m_ago,
       price_target_median_6m_ago,
       price_target_median_ytd_ago,
       price_target_median_1y_ago,
       -- ---- Lagged analyst-count trail (kalman_pt constant_data scale) ----------
       -- Coverage-count *_ago snapshots: fixed per-step analyst participation the
       -- panel conditions on (pymc_role 'constant_data'), incl. the 6m horizon
       -- (which also feeds feat_coverage_drift).
       price_target_num_1w_ago,
       price_target_num_mtd_ago,
       price_target_num_1m_ago,
       price_target_num_qtd_ago,
       price_target_num_3m_ago,
       price_target_num_6m_ago,
       price_target_num_ytd_ago,
       price_target_num_1y_ago,
       -- ---- Lagged spot-price trail (kalman_pt observed state sequence) ---------
       -- Raw historical price snapshots pooled with the analyst-target trail by
       -- KalmanFilterPriceTarget.build_price_target_history (the _AGO_HISTORY_RE
       -- unpivot already matches the bare price_* family), so the GRW filter's
       -- observed fair-value sequence gains the realised price path alongside the
       -- target path. pymc_role 'observed' for kalman_pt in
       -- pml.vw_pymc_feature_catalogue; emitted un-prefixed so feature_alias
       -- (== column_name) resolves against kalman_df.columns in the notebook
       -- present-check. Horizons mirror the feat_price_drift inputs plus the long
       -- 3y/5y anchors and the 1d / period-to-date (mtd/qtd/ytd) snapshots.
       price_1d_ago,
       price_5d_ago,
       price_1w_ago,
       price_mtd_ago,
       price_1m_ago,
       price_3m_ago,
       price_6m_ago,
       price_ytd_ago,
       price_1y_ago,
       price_3y_ago,
       price_5y_ago,
       price_qtd_ago,
       calc_change_ratio(price_target::numeric, last_price::numeric)                                                                                                                                           AS feat_implied_upside,
       -- ---- Analyst rating mix, conviction and 1y achievement / accuracy ---------
       -- Copied verbatim from mv_pymc_price_target so the kalman_pt cross-section
       -- carries the same analyst-sentiment predictors (all mutable_predictor).
       num_hold_ratings                                                                                                                                                                                        AS feat_holds,
       num_strong_buys_ratings + num_buys_ratings                                                                                                                                                              AS feat_buys,
       num_strong_sell_ratings + num_sell_ratings                                                                                                                                                              AS feat_sells,
       num_no_opinion_ratings                                                                                                                                                                                  AS feat_no_opinion,
       pml.safe_divide((num_strong_buys_ratings + num_buys_ratings)::numeric,
                       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
                        num_sell_ratings +
                        num_strong_sell_ratings)::numeric)                                                                                                                                                     AS feat_analyst_bullish_pct,
       pml.safe_divide((num_sell_ratings + num_strong_sell_ratings)::numeric,
                       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
                        num_sell_ratings +
                        num_strong_sell_ratings)::numeric)                                                                                                                                                     AS feat_analyst_bearish_pct,
       pml.safe_divide(num_hold_ratings::numeric,
                       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
                        num_sell_ratings +
                        num_strong_sell_ratings)::numeric)                                                                                                                                                     AS feat_analyst_neutral_pct,
       abs(pml.safe_divide(
		       (num_strong_buys_ratings + num_buys_ratings - (num_sell_ratings + num_strong_sell_ratings))::numeric,
		       (num_strong_buys_ratings + num_buys_ratings + num_hold_ratings + num_no_opinion_ratings +
		        num_sell_ratings +
		        num_strong_sell_ratings)::numeric))                                                                                                                                                            AS feat_analyst_conviction,
       analyst_rating                                                                                                                                                                                          AS feat_analyst_rating,
       CASE
	       WHEN price_target_1y_ago > 0::double precision AND last_price >= price_target_1y_ago
		       THEN 1.0::double precision
	       WHEN price_target_1y_ago > 0::double precision THEN pml.safe_divide(last_price, price_target_1y_ago)
	       ELSE NULL::double precision END                                                                                                                                                                     AS feat_pt_achievement_1y,
       pml.safe_divide(abs(last_price - price_target_1y_ago),
                       abs(price_target_1y_ago))                                                                                                                                                               AS feat_pt_accuracy_1y,
       CASE
	       WHEN last_price >= price_target_low_1y_ago AND last_price <= price_target_high_1y_ago THEN 1.0
	       ELSE 0.0 END                                                                                                                                                                                        AS feat_pt_range_hit_rate,
       -- ---- Per-step drift (mean log-uplift) across every price / target trail ----
       -- Each drift is min-points-guarded (>=2 valid consecutive pairs) so a single
       -- noisy pair can't masquerade as signal, and winsorised to [-1, 1] to bound the
       -- heavy ratio tails. A companion ``*_n`` valid-pair count is emitted so the
       -- fused Kalman panel (and its coverage guard) can gate on real data
       -- availability instead of the post-fill zero spike — see pml.target_drift_n /
       -- the KALMAN_RESPONSE_COVERAGE_MIN guard in prepare_kalman_panel_inputs.
       pml.winsorise(pml.target_drift(
		                     ARRAY [price_target::NUMERIC, price_target_1w_ago::NUMERIC, price_target_1m_ago::NUMERIC, price_target_3m_ago::NUMERIC, price_target_6m_ago::NUMERIC, price_target_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_pt_drift,
       pml.target_drift_n(ARRAY [price_target::NUMERIC, price_target_1w_ago::NUMERIC, price_target_1m_ago::NUMERIC, price_target_3m_ago::NUMERIC, price_target_6m_ago::NUMERIC, price_target_1y_ago::NUMERIC]) AS feat_pt_drift_n,
       pml.winsorise(pml.target_drift(
		                     ARRAY [last_price::NUMERIC, price_1w_ago::NUMERIC, price_1m_ago::NUMERIC, price_3m_ago::NUMERIC, price_6m_ago::NUMERIC, price_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_price_drift,
       pml.target_drift_n(ARRAY [last_price::NUMERIC, price_1w_ago::NUMERIC, price_1m_ago::NUMERIC, price_3m_ago::NUMERIC, price_6m_ago::NUMERIC, price_1y_ago::NUMERIC])                                      AS feat_price_drift_n,
       -- High / low / median analyst-target trails — capture skew in target drift
       pml.winsorise(pml.target_drift(
		                     ARRAY [price_target_high::NUMERIC, price_target_high_1w_ago::NUMERIC, price_target_high_1m_ago::NUMERIC, price_target_high_3m_ago::NUMERIC, price_target_high_6m_ago::NUMERIC, price_target_high_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_pt_high_drift,
       pml.winsorise(pml.target_drift(
		                     ARRAY [price_target_low::NUMERIC, price_target_low_1w_ago::NUMERIC, price_target_low_1m_ago::NUMERIC, price_target_low_3m_ago::NUMERIC, price_target_low_6m_ago::NUMERIC, price_target_low_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_pt_low_drift,
       pml.winsorise(pml.target_drift(
		                     ARRAY [price_target_median::NUMERIC, price_target_median_1w_ago::NUMERIC, price_target_median_1m_ago::NUMERIC, price_target_median_3m_ago::NUMERIC, price_target_median_6m_ago::NUMERIC, price_target_median_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_pt_median_drift,
       -- Analyst-coverage drift: rising / falling participation is a state signal
       pml.winsorise(pml.target_drift(
		                     ARRAY [price_target_num::NUMERIC, price_target_num_1w_ago::NUMERIC, price_target_num_1m_ago::NUMERIC, price_target_num_3m_ago::NUMERIC, price_target_num_6m_ago::NUMERIC, price_target_num_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_coverage_drift,
       -- Drift in analyst-stddev tells how noise itself is evolving (state-space Q)
       pml.winsorise(pml.target_drift(
		                     ARRAY [price_target_stddev::NUMERIC, price_target_stddev_1w_ago::NUMERIC, price_target_stddev_1m_ago::NUMERIC, price_target_stddev_3m_ago::NUMERIC, price_target_stddev_6m_ago::NUMERIC, price_target_stddev_1y_ago::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_pt_noise_drift,
       price_target_stddev                                                                                                                                                                                     AS feat_pt_noise_sigma,
       -- ---- Lagged analyst-stddev trail (kalman_pt observed noise sequence) -----
       -- Consensus-dispersion *_ago snapshots the panel observes as the evolving
       -- measurement-noise level (pymc_role 'observed'); emitted un-prefixed so
       -- feature_alias (== column_name) resolves in the notebook present-check.
       price_target_stddev_1w_ago,
       price_target_stddev_mtd_ago,
       price_target_stddev_1m_ago,
       price_target_stddev_qtd_ago,
       price_target_stddev_3m_ago,
       price_target_stddev_6m_ago,
       price_target_stddev_ytd_ago,
       price_target_stddev_1y_ago,
       -- Inter-analyst range (high - low) normalised by mean target
       pml.safe_divide(price_target_high - price_target_low,
                       NULLIF(price_target, 0))                                                                                                                                                                AS feat_pt_range_norm,
       -- Short-term momentum: last day's price change (mutable_predictor).
       one_day_pct                                                                                                                                                                                             AS feat_one_day_return,
       price_chg_pct_3m                                                                                                                                                                                             AS feat_price_chg_pct_3m,
       -- Drift across the realized-vol term structure (1m -> 1y) tells how price
       -- volatility itself is evolving (the sigma_obs widener analogue of
       -- feat_pt_noise_drift's state-space Q), with the matching valid-pair count.
       pml.winsorise(pml.target_drift(
		                     ARRAY [volatility_1m::NUMERIC, volatility_3m::NUMERIC, volatility_6m::NUMERIC, volatility_1y::NUMERIC],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_vol_drift,
       pml.target_drift_n(ARRAY [volatility_1m::NUMERIC, volatility_3m::NUMERIC, volatility_6m::NUMERIC, volatility_1y::NUMERIC])                                                                              AS feat_vol_drift_n,
       -- Raw beta windows (systematic-risk inputs to feat_avg_beta below).
       beta_1y,
       beta_2y,
       beta_5y,
       -- feat_avg_beta: NULL-aware mean of the available beta windows. The fused
       -- panel keys its risk adjustment on this systematic-risk (CAPM) driver,
       -- risk_adj_return = expected_return * exp(-risk_penalty * z(feat_avg_beta));
       -- realized vol enters only via the feat_vol_drift sigma_obs widener above.
       ((COALESCE(beta_1y, 0::double precision) + COALESCE(beta_2y, 0::double precision) +
         COALESCE(beta_5y, 0::double precision)) /
        NULLIF((beta_1y IS NOT NULL)::int + (beta_2y IS NOT NULL)::int + (beta_5y IS NOT NULL)::int,
               0))                                                                                                                                                                                             AS feat_avg_beta,
       total_return_ytd                                                                                                                                                                                        AS feat_total_return_ytd,
       total_return_5y                                                                                                                                                                                         AS feat_total_return_5y,
       total_return_10y                                                                                                                                                                                        AS feat_total_return_10y,
       tot_return_pct_cagr_3y                                                                                                                                                                                  AS feat_tr_cagr_3y,
       tot_return_pct_cagr_10y                                                                                                                                                                                 AS feat_tr_cagr_10y,
       tot_return_pct_cagr_5y                                                                                                                                                                                  AS feat_tr_cagr_5y,
       tot_return_pct_cagr_1y                                                                                                                                                                                  AS feat_tr_cagr_1y,
       total_return_1d                                                                                                                                                                                         AS feat_total_return_1d,
       total_return_5d                                                                                                                                                                                         AS feat_total_return_5d,
       total_return_1w                                                                                                                                                                                         AS feat_total_return_1w,
       total_return_1m                                                                                                                                                                                         AS feat_total_return_1m,
       total_return_3m                                                                                                                                                                                         AS feat_total_return_3m,
       total_return_6m                                                                                                                                                                                         AS feat_total_return_6m,
       total_return_1y                                                                                                                                                                                         AS feat_total_return_1y,
       total_return_3y                                                                                                                                                                                         AS feat_total_return_3y,
       total_return_mtd                                                                                                                                                                                        AS feat_total_return_mtd,
       total_return_qtd                                                                                                                                                                                        AS feat_total_return_qtd,
       total_return_2025                                                                                                                                                                                       AS feat_total_return_2025,
       total_return_2024                                                                                                                                                                                       AS feat_total_return_2024,
       total_return_2023                                                                                                                                                                                       AS feat_total_return_2023,
       total_return_2022                                                                                                                                                                                       AS feat_total_return_2022,
       total_return_2021                                                                                                                                                                                       AS feat_total_return_2021,
       -- ---- Cross-cutting market-cap / EV size & trend feats ----
       pml.calc_change_ratio(market_cap, market_cap_neg1fy)                                                                                                                                                    AS feat_mcap_trend_1y,
       pml.safe_divide(market_cap, market_cap_3yavg) AS feat_mcap_vs_3yavg,
       pml.safe_divide(100-market_cap_country_r, 100)                                                                                                                                                           AS feat_mcap_country_r,
       pml.safe_divide(enterprise_value, enterprise_value_3yavg)                                                                                                                                               AS feat_ev_vs_3yavg,
       -- ---- Market-cap / EV ratio trail (equity share of enterprise value) ----
       -- market_cap_ev = market_cap / enterprise_value: the equity fraction of EV
       -- (rises as a name de-levers / its equity re-rates above its net debt). The
       -- *_ago pairs reuse the matched market_cap / enterprise_value lag columns (FQ
       -- + FY trail and the 3y/5y averages). enterprise_value carries a neg5fy lag
       -- with no market_cap counterpart, so no neg5fy ratio is emitted.
       pml.safe_divide(market_cap, enterprise_value)                                                                                                                                                           AS market_cap_ev,
       pml.safe_divide(market_cap_neg1fq, enterprise_value_neg1fq)                                                                                                                                             AS market_cap_ev_neg1fq,
       pml.safe_divide(market_cap_neg2fq, enterprise_value_neg2fq)                                                                                                                                             AS market_cap_ev_neg2fq,
       pml.safe_divide(market_cap_neg3fq, enterprise_value_neg3fq)                                                                                                                                             AS market_cap_ev_neg3fq,
       pml.safe_divide(market_cap_neg4fq, enterprise_value_neg4fq)                                                                                                                                             AS market_cap_ev_neg4fq,
       pml.safe_divide(market_cap_neg1fy, enterprise_value_neg1fy)                                                                                                                                             AS market_cap_ev_neg1fy,
       pml.safe_divide(market_cap_neg2fy, enterprise_value_neg2fy)                                                                                                                                             AS market_cap_ev_neg2fy,
       pml.safe_divide(market_cap_neg3fy, enterprise_value_neg3fy)                                                                                                                                             AS market_cap_ev_neg3fy,
       pml.safe_divide(market_cap_neg4fy, enterprise_value_neg4fy)                                                                                                                                             AS market_cap_ev_neg4fy,
       pml.safe_divide(market_cap_3yavg, enterprise_value_3yavg)                                                                                                                                               AS market_cap_ev_3yavg,
       pml.safe_divide(market_cap_5yavg, enterprise_value_5yavg)                                                                                                                                               AS market_cap_ev_5yavg,
       -- feat_mv_ev_drift: per-step drift of market_cap_ev across the fiscal-year
       -- trail (current -> neg4fy), min-points-guarded (>=2 valid pairs) and
       -- winsorised to [-1, 1] like the analyst-target drifts. Positive drift flags
       -- equity re-rating / de-leveraging vs enterprise value; a kalman_pt
       -- mutable_predictor (state-transition mean / beta).
       pml.winsorise(pml.target_drift(
		                     ARRAY [pml.safe_divide(market_cap, enterprise_value), pml.safe_divide(market_cap_neg1fy, enterprise_value_neg1fy), pml.safe_divide(market_cap_neg2fy, enterprise_value_neg2fy), pml.safe_divide(market_cap_neg3fy, enterprise_value_neg3fy), pml.safe_divide(market_cap_neg4fy, enterprise_value_neg4fy)],
		                     2), -1,
                     1)                                                                                                                                                                                        AS feat_mv_ev_drift,
       -- ---- Piotroski F-score fundamental-quality trail ----
       -- Four per-fiscal-year 9-signal composites (pml.piotroski_f_score over the
       -- ROA / CFO / leverage / liquidity / share-count / margin / asset-turnover
       -- lag pairs, computed once in the pio LATERAL below) plus their median.
       -- Only feat_median_piotroski_f_score enters the fused drift design matrix
       -- (KALMAN_PIOTROSKI_COMPONENT_FEATURES bars the collinear per-year
       -- components in KalmanFilterModel.select_drift_features); the component
       -- scores are emitted for EDA / analytics.
       pio.feat_piotroski_f_score_fy,
       pio.feat_piotroski_f_score_neg1fy,
       pio.feat_piotroski_f_score_neg2fy,
       pio.feat_piotroski_f_score_neg3fy,
       -- Exact median of the four never-NULL scores: (sum - max - min) / 2.
       ((pio.feat_piotroski_f_score_fy + pio.feat_piotroski_f_score_neg1fy +
         pio.feat_piotroski_f_score_neg2fy + pio.feat_piotroski_f_score_neg3fy
	       - GREATEST(pio.feat_piotroski_f_score_fy, pio.feat_piotroski_f_score_neg1fy,
	                  pio.feat_piotroski_f_score_neg2fy, pio.feat_piotroski_f_score_neg3fy)
	       - LEAST(pio.feat_piotroski_f_score_fy, pio.feat_piotroski_f_score_neg1fy,
	               pio.feat_piotroski_f_score_neg2fy, pio.feat_piotroski_f_score_neg3fy)) /
        2.0)::double precision                                                                                                                                                                                 AS feat_median_piotroski_f_score
FROM pml.pml_df
	     CROSS JOIN LATERAL (SELECT pml.piotroski_f_score(return_on_assets_roa_pct_fy, return_on_assets_roa_pct_neg1fy,
	                                                      cfo_fy, net_income_fy,
	                                                      long_term_debt_equity_fy, long_term_debt_equity_neg1fy,
	                                                      current_ratio_fy, current_ratio_neg1fy,
	                                                      shrs_out, shrs_out_neg1fy,
	                                                      gross_profit_margin_pct_fy, gross_profit_margin_pct_neg1fy,
	                                                      asset_turnover_fy, asset_turnover_neg1fy)     AS feat_piotroski_f_score_fy,
	                                pml.piotroski_f_score(return_on_assets_roa_pct_neg1fy, return_on_assets_roa_pct_neg2fy,
	                                                      cfo_neg1fy, net_income_neg1fy,
	                                                      long_term_debt_equity_neg1fy, long_term_debt_equity_neg2fy,
	                                                      current_ratio_neg1fy, current_ratio_neg2fy,
	                                                      shrs_out_neg1fy, shrs_out_neg2fy,
	                                                      gross_profit_margin_pct_neg1fy, gross_profit_margin_pct_neg2fy,
	                                                      asset_turnover_neg1fy, asset_turnover_neg2fy) AS feat_piotroski_f_score_neg1fy,
	                                pml.piotroski_f_score(return_on_assets_roa_pct_neg2fy, return_on_assets_roa_pct_neg3fy,
	                                                      cfo_neg2fy, net_income_neg2fy,
	                                                      long_term_debt_equity_neg2fy, long_term_debt_equity_neg3fy,
	                                                      current_ratio_neg2fy, current_ratio_neg3fy,
	                                                      shrs_out_neg2fy, shrs_out_neg3fy,
	                                                      gross_profit_margin_pct_neg2fy, gross_profit_margin_pct_neg3fy,
	                                                      asset_turnover_neg2fy, asset_turnover_neg3fy) AS feat_piotroski_f_score_neg2fy,
	                                pml.piotroski_f_score(return_on_assets_roa_pct_neg3fy, return_on_assets_roa_pct_neg4fy,
	                                                      cfo_neg3fy, net_income_neg3fy,
	                                                      long_term_debt_equity_neg3fy, long_term_debt_equity_neg4fy,
	                                                      current_ratio_neg3fy, current_ratio_neg4fy,
	                                                      shrs_out_neg3fy, shrs_out_neg4fy,
	                                                      gross_profit_margin_pct_neg3fy, gross_profit_margin_pct_neg4fy,
	                                                      asset_turnover_neg3fy, asset_turnover_neg4fy) AS feat_piotroski_f_score_neg3fy) pio;

ALTER MATERIALIZED VIEW mv_pymc_kalman_pt OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_pymc_kalman_pt_isin ON mv_pymc_kalman_pt (isin);
