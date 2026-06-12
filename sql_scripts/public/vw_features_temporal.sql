CREATE VIEW public.vw_features_temporal
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, fiscal_quarter, fiscal_month,
			 fiscal_year, days_to_earnings, earnings_report_recency, reporting_lag, fiscal_year_progress,
			 days_since_last_report, days_to_fy_end, is_quarter_end_month, is_fy_end_month, earnings_season_flag,
			 pre_earnings_window, post_earnings_window, reporting_freshness_score, fiscal_quarter_progress)
AS
-- missing source code
;

COMMENT ON VIEW public.vw_features_temporal IS 'Temporal and fiscal calendar features for earnings timing and seasonality.
    Source functions: calc_temporal_features, calc_fiscal_calendar_features';

ALTER TABLE public.vw_features_temporal
	OWNER TO postgres;