CREATE VIEW public.vw_features_profitability
			(isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
			 dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
			 next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
			 dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
			 dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
			 next_fy_end_date, next_income_statement_report_date, reference_date, roe, roa, gross_margin_pct,
			 operating_margin_pct, net_margin_pct, ebitda_margin_pct, roic, rnd_intensity, equity_multiplier,
			 gross_margin_trend_yoy, operating_margin_trend, net_margin_trend_yoy, ebitda_margin_trend,
			 margin_expansion_flag, margin_stability_score, ebit_fq, ebit_ltm, ebit_fy, ebit_1fy, ebit_2fy, ebit_3fy,
			 ebit_4fy, ebitda_fq, ebitda_ltm, ebitda_fy, ebitda_1fy, ebitda_2fy, ebitda_3fy, ebitda_4fy, ebit_5yavgfq,
			 ebit_5yavgltm, ebitda_5yavgfq, ebitda_5yavgltm, ebit_adj_fq, ebit_adj_ltm, ebit_adj_fy, ebitda_adj_fq,
			 ebitda_adj_ltm, ebitda_adj_fy, ebit_growth_yoy, ebitda_growth_yoy, ebit_margin_ltm, ebitda_margin_ltm,
			 ebit_positive_years, ebitda_positive_years, ebit_qoq_growth, ebitda_qoq_growth, ebit_cagr_3y,
			 ebitda_cagr_3y, ebit_vs_5y_avg, ebitda_vs_5y_avg, gp_fq, gp_fy, gp_ltm, gp_1fqfq, gp_2fqfq, gp_3fqfq,
			 gp_4fqfq, gp_1fy, gp_2fy, gp_3fy, gp_4fy, gp_qoq_growth, gp_yoy_growth, gp_margin_fq, gp_margin_trend,
			 gp_positive_quarters, gp_margin_expansion)
AS
-- missing source code
;

COMMENT ON VIEW public.vw_features_profitability IS 'Profitability metrics including ROE, ROA, margins, EBIT/EBITDA comprehensive analysis.
    Source functions: calc_profitability_features, calc_margin_trends,
    calc_ebit_ebitda_comprehensive, calc_gross_profit_temporal';

ALTER TABLE public.vw_features_profitability
	OWNER TO postgres;