create view public.vw_features_quality_risk
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, has_goodwill_impairment,
			 has_asset_writedown, has_restructuring, goodwill_to_assets_pct, intangible_intensity,
			 exceptional_items_to_ebitda, altman_z_score, altman_z_trend, current_ratio, quick_ratio, beta_1y, beta_5y,
			 beta_spread, beta_trend, high_beta_flag, low_beta_flag, beta_stability_score, distress_risk_score,
			 liquidity_stress_score, working_capital_trend, cash_runway_months, combined_distress_score,
			 wc_deteriorating_flag, retained_earnings_growth, accumulated_deficit_flag, adequate_cash_buffer,
			 goodwill_change_rate, restructuring_intensity, exceptional_items_frequency, merger_impact_ratio,
			 non_operating_income_share, asset_sale_boost, accounting_quality_score, goodwill_impairment_ltm,
			 asset_writedown_ltm, restructuring_ltm, has_goodwill_impairment_ltm, goodwill_impairment_frequency,
			 asset_writedown_frequency, restructuring_frequency, exceptional_items_total_ltm,
			 exceptional_items_to_ebitda_comp, quality_issues_count_5y, accounting_quality_score_comp)
as
-- missing source code
;

comment on view public.vw_features_quality_risk is 'Quality and risk metrics including accounting quality, financial distress, and beta analysis.
    Source functions: calc_quality_features, calc_beta_risk_features, calc_financial_distress_features,
    calc_accounting_quality_features, calc_quality_features_comprehensive'
;

alter table public.vw_features_quality_risk
	owner to postgres
;