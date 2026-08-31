create table analytics.panel_vintage_v2
(
	asof_date              date                                   not null,
	isin                   text                                   not null,
	run_id                 text,
	captured_at            timestamp with time zone default now() not null,
	ticker                 text,
	name                   text,
	sector                 text,
	industry               text,
	trading_region         text,
	country                text,
	style_class            text,
	size_class             text,
	market_cap             double precision,
	last_price             double precision,
	observed_pt            double precision,
	n_analysts             double precision,
	price_1w_ago           double precision,
	price_1m_ago           double precision,
	price_3m_ago           double precision,
	price_6m_ago           double precision,
	price_1y_ago           double precision,
	price_target_1w_ago    double precision,
	price_target_1m_ago    double precision,
	price_target_3m_ago    double precision,
	price_target_6m_ago    double precision,
	price_target_1y_ago    double precision,
	n_analysts_1w          double precision,
	n_analysts_1m          double precision,
	n_analysts_3m          double precision,
	n_analysts_6m          double precision,
	n_analysts_1y          double precision,
	implied_upside         double precision,
	expected_return_kalman double precision,
	expected_upside_sd     double precision,
	shrink_gain            double precision,
	er_mean                double precision,
	er_sd                  double precision,
	er_p05                 double precision,
	er_p50                 double precision,
	er_p95                 double precision,
	mc_prob_pos            double precision,
	p_upside_pos_cond      double precision,
	cvar_5pct_kalman       double precision,
	out_of_support         boolean,
	constraint pk_panel_vintage_v2
		primary key (asof_date, isin)
)
;

comment on table analytics.panel_vintage_v2 is 'Append-only point-in-time capture of the Kalman v2 panel and its decision outputs. Two captures separated in time are what scripts/score_panel_vintages.py needs to score the model against realised returns rather than against the analyst trail it was fitted to.'
;

comment on column analytics.panel_vintage_v2.asof_date is 'The date this row describes. Supplied by the capture script, defaulting to CURRENT_DATE -- NOT derived from the data, because the trail columns carry no date of their own, which is the whole reason this table exists.'
;

comment on column analytics.panel_vintage_v2.expected_return_kalman is 'The model expected upside as of asof_date. Raw decimal. Scored later against (last_price at a subsequent vintage / last_price here - 1).'
;

comment on column analytics.panel_vintage_v2.shrink_gain is 'Weight the forecast-error update put on the name own smoothed observation. Captured because calibrating forecast_error_multiplier against realised outcomes is the point of the exercise -- a vintage without it cannot tell you whether the shrinkage helped.'
;

alter table analytics.panel_vintage_v2
	owner to postgres
;

create index idx_panel_vintage_v2_asof
	on analytics.panel_vintage_v2 (asof_date)
;

create index idx_panel_vintage_v2_isin
	on analytics.panel_vintage_v2 (isin)
;