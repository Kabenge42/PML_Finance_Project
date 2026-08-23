create table analytics."10_screen_results_v2"
(
	isin               text,
	ticker             text,
	name               text,
	sector             text,
	industry           text,
	trading_region     text,
	country            text,
	style_class        text,
	size_class         text,
	n_analysts         double precision,
	market_cap         double precision,
	mcap_global_r      double precision,
	mcap_country_r     double precision,
	last_price         double precision,
	observed_pt        double precision,
	expected_upside    double precision,
	expected_upside_sd double precision,
	prob_pos           double precision,
	implied_upside     double precision,
	expected_pt        double precision,
	expected_pt_hdi_lo double precision,
	expected_pt_hdi_hi double precision,
	shrink_gain        double precision,
	risk_adj_return    double precision,
	kalman_gain        double precision,
	er_mean            double precision,
	er_sd              double precision,
	er_p05             double precision,
	er_p50             double precision,
	er_p95             double precision,
	mc_prob_pos        double precision,
	p_upside_pos_cond  double precision,
	run_id             text,
	exported_at        timestamp with time zone,
	source_sha         text,
	source_dirty       boolean
)
;

alter table analytics."10_screen_results_v2"
	owner to postgres
;