create view public.vw_features_unusual_items
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, other_unusual_items_ltm,
			 impairment_goodwill_ltm, asset_writedown_ltm, restructuring_charges_ltm, total_unusual_items,
			 unusual_items_to_revenue, unusual_items_to_ebitda, has_unusual_items_flag, earnings_quality_impact)
as
-- missing source code
;

comment on view public.vw_features_unusual_items is 'Non-recurring and unusual items analysis for earnings quality assessment.
    Source function: calc_unusual_items_features'
;

alter table public.vw_features_unusual_items
	owner to postgres
;