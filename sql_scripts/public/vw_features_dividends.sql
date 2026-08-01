CREATE VIEW public.vw_features_dividends
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, dividend_streak, dividend_yield_ltm,
			 dividend_yield_ntm, dividend_payout_ratio, fcf_dividend_coverage, buyback_yield, total_shareholder_yield,
			 dividend_growth_expectation, days_since_ex_date, days_to_payment, dividend_announced_flag,
			 ex_date_approaching_flag, dividend_frequency_score, dividend_consistency, recent_dividend_change,
			 dividend_yield_vs_5y_avg, div_yield_ltm, div_yield_ntm, div_yield_ind, div_yield_1fy_ind, div_yield_5y_avg,
			 div_yield_vs_5y_avg, div_yield_growth_expected, dividend_streak_comp, high_yield_flag,
			 sustainable_dividend_flag)
AS
-- missing source code
;

COMMENT ON VIEW public.vw_features_dividends IS 'Dividend metrics including yield, payout ratios, timing, and sustainability.
    Source functions: calc_dividend_features, calc_dividend_timing, calc_dividend_yield_comprehensive';

ALTER TABLE public.vw_features_dividends
	OWNER TO postgres;