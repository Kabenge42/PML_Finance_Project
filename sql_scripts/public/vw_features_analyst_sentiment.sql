CREATE VIEW public.vw_features_analyst_sentiment
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, analyst_bullish_pct,
			 analyst_bearish_pct, analyst_neutral_pct, analyst_conviction, upside_potential, price_target_spread_pct,
			 price_target_revision_1m, price_target_revision_3m, eps_revision_momentum, analyst_rating_normalized,
			 analyst_coverage_quality, pt_momentum_1w, pt_momentum_1m, pt_momentum_3m, pt_momentum_6m, pt_momentum_1y,
			 pt_median_momentum_1m, pt_median_momentum_3m, pt_acceleration_short, pt_acceleration_long,
			 pt_consensus_convergence, analyst_coverage_change_1m, analyst_coverage_change_3m,
			 analyst_coverage_change_1y, pt_vs_price_momentum, analyst_coverage_trend)
AS
-- missing source code
;

COMMENT ON VIEW public.vw_features_analyst_sentiment IS 'Analyst sentiment metrics including ratings distribution and price target dynamics.
    Source functions: calc_sentiment_features, calc_price_target_dynamics';

ALTER TABLE public.vw_features_analyst_sentiment
	OWNER TO postgres;