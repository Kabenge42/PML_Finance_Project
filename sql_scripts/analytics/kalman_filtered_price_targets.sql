CREATE TABLE analytics.kalman_filtered_price_targets
(
	isin                   text,
	ticker                 text,
	name                   text,
	country                text,
	unit                   text,
	exchange               text,
	sector                 text,
	industry               text,
	expected_return_kalman double precision,
	price_target_kalman    double precision,
	kalman_variance        double precision,
	kalman_gain            double precision,
	signal_strength        double precision,
	original_price         double precision,
	original_target        double precision,
	market_cap             double precision,
	enterprise_value       double precision,
	expected_pt_hdi_lo     double precision,
	expected_pt_hdi_hi     double precision,
	risk_adj_return        double precision,
	er_mean                double precision,
	er_p05                 double precision,
	er_p50                 double precision,
	er_p95                 double precision,
	mc_prob_pos            double precision,
	cvar_book_weight       double precision,
	cvar_5pct_kalman       double precision,
	reward_to_cvar         double precision
);

ALTER TABLE analytics.kalman_filtered_price_targets
	OWNER TO postgres;