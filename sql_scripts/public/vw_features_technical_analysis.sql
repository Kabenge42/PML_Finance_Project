CREATE VIEW public.vw_features_technical_analysis
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, ema_slope_20d, ema_trend_consistency,
			 price_vs_ema_100d, near_52w_high_flag, near_52w_low_flag, volume_momentum_score, breakout_signal,
			 high_volume_flag, low_volume_flag, volatility_compression, volatility_term_structure)
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
       ta.ema_slope_20d,
       ta.ema_trend_consistency,
       ta.price_vs_ema_100d,
       ta.near_52w_high_flag,
       ta.near_52w_low_flag,
       ta.volume_momentum_score,
       ta.breakout_signal,
       ta.high_volume_flag,
       ta.low_volume_flag,
       ta.volatility_compression,
       ta.volatility_term_structure
FROM vw_identifier_columns                            id
	     LEFT JOIN calc_technical_analysis_features() ta(isin, ema_slope_20d, ema_trend_consistency, price_vs_ema_100d,
	                                                     near_52w_high_flag, near_52w_low_flag, volume_momentum_score,
	                                                     breakout_signal, high_volume_flag, low_volume_flag,
	                                                     volatility_compression, volatility_term_structure)
	               USING (isin);

COMMENT ON VIEW public.vw_features_technical_analysis IS 'Technical analysis indicators including EMA trends, volume signals, and volatility patterns.
    Source function: calc_technical_analysis_features';

ALTER TABLE public.vw_features_technical_analysis
	OWNER TO postgres;