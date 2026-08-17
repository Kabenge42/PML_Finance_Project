create table analytics."10_screen_results"
(
	isin               text,
	ticker             text,
	name               text,
	trading_region     text,
	region             text,
	country            text,
	unit               text,
	exchange           text,
	sector             text,
	industry           text,
	size_class         text,
	style_class        text,
	market_cap         double precision,
	mcap_country_r     double precision,
	mcap_global_r      double precision,
	enterprise_value   double precision,
	last_price         double precision,
	observed_pt        double precision,
	expected_pt        double precision,
	expected_pt_hdi_lo double precision,
	expected_pt_hdi_hi double precision,
	expected_upside    double precision,
	risk_adj_return    double precision,
	prob_pos           double precision,
	implied_upside     double precision,
	total_return_ytd   double precision,
	total_return_5y    double precision,
	total_return_10y   double precision,
	tr_cagr_3y         double precision,
	n_analysts         double precision,
	er_mean            double precision,
	er_sd              double precision,
	er_p05             double precision,
	er_p50             double precision,
	er_p95             double precision,
	mc_prob_pos        double precision,
	run_id             text,
	exported_at        timestamp with time zone
)
;

alter table analytics."10_screen_results"
	owner to postgres
;