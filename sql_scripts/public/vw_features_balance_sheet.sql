CREATE VIEW public.vw_features_balance_sheet
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, assets_fq, assets_fy, assets_ltm,
			 assets_1fq, assets_2fq, assets_3fq, assets_4fq, assets_1fy, assets_2fy, assets_3fy, assets_4fy,
			 assets_qoq_growth, assets_yoy_growth, assets_3y_cagr, asset_growth_accel, asset_base_stable, inventory_ltm,
			 inventory_fq, inventory_fy, inventory_1fq, inventory_2fq, inventory_3fq, inventory_4fq, inventory_1fy,
			 inventory_2fy, inventory_3fy, inventory_4fy, inventory_qoq_change, inventory_yoy_change,
			 inventory_4q_trend, inventory_vs_5y_avg, inventory_days, inventory_turnover, inventory_to_revenue,
			 inventory_to_assets, inventory_buildup_flag, inventory_reduction_flag, inventory_volatility, goodwill_fq,
			 goodwill_ltm, goodwill_fy, goodwill_1fq, goodwill_2fq, goodwill_3fq, goodwill_4fq, goodwill_1fy,
			 goodwill_2fy, goodwill_3fy, goodwill_4fy, goodwill_qoq_change, goodwill_yoy_change, goodwill_3y_growth,
			 goodwill_vs_5y_avg, recent_acquisition_flag, goodwill_accumulation_rate, goodwill_to_assets_trend,
			 impairment_risk_score, goodwill_concentration)
AS
-- missing source code
;

COMMENT ON VIEW public.vw_features_balance_sheet IS 'Balance sheet temporal analysis including assets, inventory, and goodwill trends.
    Source functions: calc_total_assets_temporal, calc_inventory_temporal_features, calc_goodwill_temporal_features';

ALTER TABLE public.vw_features_balance_sheet
	OWNER TO postgres;