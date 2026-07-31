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
SELECT id.isin,
       id.ticker,
       id.name,
       id.description,
       id.region,
       id.country,
       id.trading_country,
       id.exchange,
       id.sector,
       id.industry,
       id.dividend_record_frequency,
       id.earnings_report_frequency,
       id.fy_end,
       id.next_earnings_report,
       id.next_earnings_status,
       id.next_earnings_when,
       id.next_fiscal_quarter,
       id.reporting_interval,
       id.size_class,
       id.style_class,
       id.unit,
       id.dividend_record_announce_date,
       id.dividend_record_ex_date,
       id.dividend_record_payable_date,
       id.dividend_record_record_date,
       id.fy_end_date,
       id.income_statement_report_date,
       id.last_updated,
       id.next_earnings,
       id.next_fy_end_date,
       id.next_income_statement_report_date,
       id.reference_date,
       tf.fiscal_quarter,
       tf.fiscal_month,
       tf.fiscal_year,
       tf.days_to_earnings,
       tf.earnings_report_recency,
       tf.reporting_lag,
       tf.fiscal_year_progress,
       fcf.days_since_last_report,
       fcf.days_to_fy_end,
       fcf.is_quarter_end_month,
       fcf.is_fy_end_month,
       fcf.earnings_season_flag,
       fcf.pre_earnings_window,
       fcf.post_earnings_window,
       fcf.reporting_freshness_score,
       fcf.fiscal_quarter_progress
FROM vw_identifier_columns                         id
	     LEFT JOIN calc_temporal_features()        tf(isin, fiscal_quarter, fiscal_month, fiscal_year, days_to_earnings,
	                                                  earnings_report_recency, reporting_lag, fiscal_year_progress)
	               USING (isin)
	     LEFT JOIN calc_fiscal_calendar_features() fcf(isin, days_since_last_report, days_to_fy_end,
	                                                   is_quarter_end_month, is_fy_end_month, earnings_season_flag,
	                                                   pre_earnings_window, post_earnings_window,
	                                                   reporting_freshness_score, fiscal_quarter_progress) USING (isin);

COMMENT ON VIEW public.vw_features_temporal IS 'Temporal and fiscal calendar features for earnings timing and seasonality.
    Source functions: calc_temporal_features, calc_fiscal_calendar_features';

ALTER TABLE public.vw_features_temporal
	OWNER TO postgres;