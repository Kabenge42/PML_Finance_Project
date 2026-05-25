CREATE MATERIALIZED VIEW pml.mv_pymc_kalman_pt AS
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
       price_target                                                                                                                                                                                                                                AS observed_pt,
       last_price,
       price_target_high,
       price_target_low,
       price_target_num                                                                                                                                                                                                                            AS n_analysts,
       target_drift(ARRAY [price_target::numeric, price_target_1w_ago::numeric, price_target_1m_ago::numeric, price_target_3m_ago::numeric, price_target_6m_ago::numeric, price_target_1y_ago::numeric])                                           AS feat_pt_drift,
       target_drift(ARRAY [last_price::numeric, price_1w_ago::numeric, price_1m_ago::numeric, price_3m_ago::numeric, price_6m_ago::numeric, price_1y_ago::numeric])                                                                                AS feat_price_drift,
       target_drift(ARRAY [price_target_stddev::numeric, price_target_stddev_1w_ago::numeric, price_target_stddev_1m_ago::numeric, price_target_stddev_3m_ago::numeric, price_target_stddev_6m_ago::numeric, price_target_stddev_1y_ago::numeric]) AS feat_pt_noise_drift,
       price_target_stddev                                                                                                                                                                                                                         AS feat_pt_noise_sigma,
       safe_divide(price_target_high - price_target_low,
                   NULLIF(price_target, 0::double precision))                                                                                                                                                                                      AS feat_pt_range_norm,
       volatility_1m                                                                                                                                                                                                                               AS feat_vol_1m,
       volatility_3m                                                                                                                                                                                                                               AS feat_vol_3m,
       volatility_6m                                                                                                                                                                                                                               AS feat_vol_6m,
       volatility_1y                                                                                                                                                                                                                               AS feat_vol_1y,
       total_return_ytd                                                                                                                                                                                                                            AS feat_total_return_ytd
FROM pml_df;

ALTER MATERIALIZED VIEW pml.mv_pymc_kalman_pt OWNER TO postgres;

CREATE UNIQUE INDEX idx_mv_pymc_kalman_pt_isin ON pml.mv_pymc_kalman_pt (isin);