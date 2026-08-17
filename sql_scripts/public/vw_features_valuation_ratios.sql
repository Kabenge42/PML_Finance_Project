create view public.vw_features_valuation_ratios
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, p_e_ratio, p_b_ratio, ev_ebitda_ratio,
			 ev_sales_ratio, dividend_yield, peg_ratio, ev_sales_trend_1y, ev_ebitda_momentum, p_e_momentum_yoy,
			 p_e_momentum_qoq, ev_sales_vs_3y_avg, ev_ebitda_vs_3y_avg, p_e_vs_3y_avg, ev_sales_forward_discount,
			 ev_ebitda_forward_discount, p_e_forward_discount, p_b_vs_5y_avg, ev_sales_qoq_1q, ev_sales_qoq_2q,
			 ev_sales_qoq_3q, ev_sales_qoq_4q, p_e_vs_5y_avg, p_e_percentile_proxy, valuation_mean_reversion,
			 ev_ebitda_qoq_trend, p_b_momentum_yoy, valuation_compression, forward_pe_premium, tangible_book_value_fy,
			 tangible_book_value_ltm, tangible_book_per_share, price_to_tangible_book, tangible_equity_ratio,
			 intangibles_to_equity, goodwill_to_equity, tangible_asset_quality, tbv_yoy_growth, tbv_vs_calculated)
as
-- missing source code
;

comment on view public.vw_features_valuation_ratios is 'Valuation metrics including P/E, P/B, EV/EBITDA, tangible book value, and timeseries analysis.
    Identifier columns inherited from vw_identifier_columns.
    Source functions: calc_valuation_features, calc_valuation_timeseries_features,
    calc_extended_valuation_timeseries, calc_tangible_book_features'
;

alter table public.vw_features_valuation_ratios
	owner to postgres
;