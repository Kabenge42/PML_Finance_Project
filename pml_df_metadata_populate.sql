-- =============================================================================
-- PML SCHEMA METADATA POPULATION
-- =============================================================================
-- Populates pml.pml_df_metadata with suggested (category, feature_role) pairs
-- for every column of pml.pml_df, so feature-engineering pipelines can filter
-- columns directly in SQL (e.g. by feature_role for ML predictor selection,
-- by category for domain-based grouping/screening).
--
-- Design (mirrors equities_schema_metadata_setup.sql conventions):
--   * `feature_role`  -> coarse ML role used for SQL filtering during feature
--                        engineering. Stable, low-cardinality vocabulary:
--                          id           : row identifier (ticker, isin)
--                          categorical  : low-cardinality classification used
--                                         as fixed/hierarchical effect input
--                          date         : DATE / text fiscal calendar column
--                          target       : analyst-driven price target or
--                                         derived target % (model target /
--                                         label inputs)
--                          predictor    : numeric feature suitable as an ML
--                                         predictor (default for numerics)
--                          count        : integer count used as discrete
--                                         predictor (analyst counts, etc.)
--                          score        : pre-computed score / risk indicator
--                                         (Altman Z, beta, ratings)
--                          historical   : lagged level used to derive
--                                         momentum / drift / streak features
--                                         (history-only -- do NOT feed raw)
--                          surprise     : actual-vs-estimate surprise %
--                                         (signed, ready-to-use predictor)
--                          revision     : analyst estimate revision %
--                                         (signed, ready-to-use predictor)
--                          metadata     : free-text / row-level metadata
--                                         (name, description) -- excluded
--                                         from ML predictor sets by default
--
--   * `category`      -> data-domain bucket aligning with feature_catalogue /
--                        feature-view groupings. One column may belong to
--                        exactly one category; categories follow the
--                        feature_registry sections (valuation, profitability,
--                        growth, cash_flow, technical, etc.).
-- =============================================================================

BEGIN;

-- Backfill ordinal_position and data_type from information_schema so any
-- subsequent UPDATEs only have to touch (category, feature_role, description).
INSERT INTO pml.pml_df_metadata (column_name, category, feature_role, ordinal_position, data_type)
SELECT c.column_name, 'n/a' AS category, 'predictor' AS feature_role, c.ordinal_position, c.data_type
FROM information_schema.columns c
WHERE c.table_schema = 'pml'
  AND c.table_name = 'pml_df'
ON CONFLICT (column_name) DO UPDATE SET ordinal_position = excluded.ordinal_position,
                                        data_type        = excluded.data_type,
                                        updated_at       = CURRENT_TIMESTAMP;

-- ---------------------------------------------------------------------------
-- Helper: bulk assignment macro emulated via repeated UPDATE statements.
-- ---------------------------------------------------------------------------

-- =========================================================================
-- IDENTIFIERS  (feature_role = 'id' / 'metadata')
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'identifier',
    feature_role = 'id'
WHERE column_name IN ('ticker', 'isin');

UPDATE pml.pml_df_metadata
SET category     = 'identifier',
    feature_role = 'metadata'
WHERE column_name IN ('name', 'description');

-- =========================================================================
-- CLASSIFICATION  (hierarchical categorical effects -- mirrors _hierarchy.py
-- HIERARCHICAL_CATEGORY_COLS)
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'classification',
    feature_role = 'categorical'
WHERE column_name IN ('region', 'country', 'trading_country', 'exchange',
                      'unit', 'sector', 'industry', 'style_class', 'size_class');

-- =========================================================================
-- FISCAL CALENDAR  (DATE + textual fiscal-period descriptors)
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'fiscal_calendar',
    feature_role = 'date'
WHERE column_name IN ('last_updated',
                      'income_statement_report_date',
                      'next_earnings',
                      'fy_end_date',
                      'next_income_statement_report_date',
                      'next_fy_end_date',
                      'expected_report_date');

UPDATE pml.pml_df_metadata
SET category     = 'fiscal_calendar',
    feature_role = 'categorical'
WHERE column_name IN ('fy_end', 'next_earnings_when', 'next_earnings_status');

-- Derived day-count signals from fiscal-calendar dates (ready-to-use numeric
-- predictors). Computed at import time as:
--   days_to_earnings        = next_earnings - CURRENT_DATE
--   earnings_report_recency = CURRENT_DATE - income_statement_report_date
UPDATE pml.pml_df_metadata
SET category     = 'fiscal_calendar',
    feature_role = 'predictor'
WHERE column_name IN ('days_to_earnings', 'earnings_report_recency');

-- =========================================================================
-- DIVIDEND
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'dividend',
    feature_role = 'categorical'
WHERE column_name IN ('dividend_record_currency', 'dividend_record_frequency');

UPDATE pml.pml_df_metadata
SET category     = 'dividend',
    feature_role = 'date'
WHERE column_name IN ('dividend_record_announce_date', 'dividend_record_payable_date',
                      'dividend_record_record_date', 'dividend_record_ex_date');

UPDATE pml.pml_df_metadata
SET category     = 'dividend',
    feature_role = 'predictor'
WHERE column_name IN ('dividend_record_amount', 'dividend_streak',
                      'dividend_per_share_ltm', 'dividend_per_share_fq', 'dividend_per_share_fy',
                      'div_yield_ind', 'div_yield_ltm', 'div_yield_ttm', 'div_yield_ntm',
                      'div_yield_5yavgltm',
                      'common_dividends_paid_ltm', 'common_dividends_paid_fy',
                      'buyback_yield_ltm');

UPDATE pml.pml_df_metadata
SET category     = 'dividend',
    feature_role = 'historical'
WHERE column_name IN ('div_yield_neg1fyind', 'div_yield_neg2fyind', 'div_yield_neg3fyind',
                      'div_yield_neg4fyind', 'div_yield_neg5fyind',
                      'dividend_per_share_neg1fqfq', 'dividend_per_share_neg2fqfq',
                      'dividend_per_share_neg3fqfq', 'dividend_per_share_neg4fqfq',
                      'dividend_per_share_neg1fy', 'dividend_per_share_neg2fy',
                      'dividend_per_share_neg3fy', 'dividend_per_share_neg4fy',
                      'dividend_per_share_neg5fy');

-- =========================================================================
-- MARKET DATA  (size / EV / volume)
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'market_data',
    feature_role = 'predictor'
WHERE column_name IN ('market_cap', 'enterprise_value', 'volume_shrs', 'rel_volume',
                      'shrs_out', 'shrs_out_3yavg', 'shrs_out_5yavg');

UPDATE pml.pml_df_metadata
SET category     = 'market_data',
    feature_role = 'count'
WHERE column_name IN ('market_cap_country_r');

UPDATE pml.pml_df_metadata
SET category     = 'market_data',
    feature_role = 'historical'
WHERE column_name LIKE 'enterprise_value_neg%'
   OR column_name LIKE 'shrs_out_neg%'
   OR column_name = 'shrs_out_neg1fy';

-- =========================================================================
-- HISTORICAL PRICES  (raw price levels -- treat as 'historical': derive
-- momentum/returns downstream, do NOT feed levels directly to most models)
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'historical_prices',
    feature_role = 'predictor'
WHERE column_name = 'last_price';

UPDATE pml.pml_df_metadata
SET category     = 'historical_prices',
    feature_role = 'historical'
WHERE column_name IN ('price_1d_ago', 'price_5d_ago', 'price_1w_ago', 'price_1m_ago',
                      'price_3m_ago', 'price_6m_ago', 'price_1y_ago', 'price_3y_ago',
                      'price_5y_ago', 'price_qtd_ago', 'price_mtd_ago', 'price_ytd_ago');

-- =========================================================================
-- TECHNICAL / MOMENTUM  (52W bands, EMAs, % changes)
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'technical',
    feature_role = 'predictor'
WHERE column_name IN ('w_52high_adj', 'w_52low_adj',
                      'ema_20d', 'ema_50d', 'ema_100d', 'ema_250d',
                      'price_chg_pct_3m', 'one_day_pct');

UPDATE pml.pml_df_metadata
SET category     = 'volatility',
    feature_role = 'predictor'
WHERE column_name IN ('volatility_1m', 'volatility_3m', 'volatility_6m', 'volatility_1y',
                      'beta_1y', 'beta_2y', 'beta_5y');

-- =========================================================================
-- TOTAL RETURN  (realized returns -- typically used as label/target inputs)
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'total_return',
    feature_role = 'target'
WHERE column_name IN ('total_return_ytd', 'total_return_5y', 'total_return_10y',
                      'tot_return_pct_cagr_3y', 'tot_return_pct_cagr_10y');

-- =========================================================================
-- ANALYST PRICE TARGETS  (model labels for PriceTargetModel / DCF)
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'analyst_targets',
    feature_role = 'target'
WHERE column_name IN ('price_target', 'price_target_low', 'price_target_median',
                      'price_target_high', 'price_target_stddev',
                      'target_pct_avg', 'target_pct_med', 'target_pct_low', 'target_pct_high');

UPDATE pml.pml_df_metadata
SET category     = 'analyst_targets',
    feature_role = 'count'
WHERE column_name = 'price_target_num';

-- Lagged price-target snapshots -> historical (used for drift enrichment,
-- not as raw predictors)
UPDATE pml.pml_df_metadata
SET category     = 'analyst_targets',
    feature_role = 'historical'
WHERE column_name LIKE 'price_target_%_ago'
   OR column_name LIKE 'price_target_num_%_ago'
   OR column_name LIKE 'price_target_high_%_ago'
   OR column_name LIKE 'price_target_low_%_ago'
   OR column_name LIKE 'price_target_median_%_ago'
   OR column_name LIKE 'price_target_stddev_%_ago'
   OR column_name = 'price_target_ytd_ago';

-- =========================================================================
-- ANALYST RATINGS
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'analyst_ratings',
    feature_role = 'score'
WHERE column_name = 'analyst_rating';

UPDATE pml.pml_df_metadata
SET category     = 'analyst_ratings',
    feature_role = 'count'
WHERE column_name IN ('num_strong_sell_ratings', 'num_strong_buys_ratings',
                      'num_hold_ratings', 'num_buys_ratings', 'num_sell_ratings',
                      'num_no_opinion_ratings');

-- =========================================================================
-- VALUATION  (P/E, P/B, EV/Sales, EV/EBITDA, PEG)
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'valuation',
    feature_role = 'predictor'
WHERE column_name IN ('p_e_ntm', 'p_e_ltm', 'p_e_est_fy1', 'p_e_est_fy2', 'p_e_est_fy3',
                      'p_e_est_fy4', 'p_e_est_fy5',
                      'p_e_3yavgltm', 'p_e_5yavgltm', 'p_e_3yavgntm', 'p_e_5yavgntm',
                      'p_b_ltm', 'p_b_5yavg',
                      'ev_sales_ltm', 'ev_sales_ntm', 'ev_sales_3yavgltm',
                      'ev_ebitda_ltm', 'ev_ebitda_ntm', 'ev_ebitda_3yavgltm', 'ev_ebitda_est_fy1',
                      'peg_ntm', 'peg_3yavg', 'peg_5yavg');

UPDATE pml.pml_df_metadata
SET category     = 'valuation',
    feature_role = 'historical'
WHERE column_name LIKE 'p_e_neg%'
   OR column_name LIKE 'p_b_neg%'
   OR column_name LIKE 'ev_sales_neg%'
   OR column_name LIKE 'ev_ebitda_neg%'
   OR column_name LIKE 'peg_neg%';

-- =========================================================================
-- PROFITABILITY  (margins, ROA, gross profit)
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'profitability',
    feature_role = 'predictor'
WHERE column_name IN ('gross_profit_margin_pct_ltm', 'gross_profit_margin_pct_fy',
                      'gross_profit_margin_pct_fq',
                      'gross_profit_margin_pct_3yavgfq', 'gross_profit_margin_pct_5yavgfq',
                      'return_on_assets_roa_pct_ltm', 'return_on_assets_roa_pct_fy');

UPDATE pml.pml_df_metadata
SET category     = 'profitability',
    feature_role = 'historical'
WHERE column_name LIKE 'gross_profit_margin_pct_neg%'
   OR column_name LIKE 'gross_profit_neg%';

-- =========================================================================
-- EPS (NORM / ADJ / BASIC / GAAP)
-- =========================================================================
-- LTM/FQ/FY current-period EPS levels and forward estimates -> predictor
UPDATE pml.pml_df_metadata
SET category     = 'eps',
    feature_role = 'predictor'
WHERE column_name IN ('eps_adj_ltm', 'eps_adj_fy', 'eps_adj_fq',
                      'eps_norm_est_avg_ntm', 'eps_norm_est_avg_fy1e',
                      'eps_norm_est_avg_fy2e', 'eps_norm_est_avg_fy3e',
                      'eps_norm_est_avg_fy4e', 'eps_norm_est_avg_fy5e',
                      'eps_norm_est_avg_fq1e', 'eps_norm_est_avg_fq2e',
                      'eps_norm_est_avg_fq3e', 'eps_norm_est_avg_fq4e',
                      'eps_gaap_est_avg_ntm', 'eps_gaap_est_avg_fy1e',
                      'net_eps_basic_ltm', 'net_eps_basic_fy', 'net_eps_basic_fq',
                      'basic_eps_cont_ltm', 'basic_eps_cont_fy', 'basic_eps_cont_fq');

UPDATE pml.pml_df_metadata
SET category     = 'eps',
    feature_role = 'count'
WHERE column_name = 'eps_norm_est_num_fy1e';

UPDATE pml.pml_df_metadata
SET category     = 'eps',
    feature_role = 'historical'
WHERE column_name LIKE 'eps_adj_neg%'
   OR column_name LIKE 'net_eps_basic_neg%'
   OR column_name LIKE 'basic_eps_cont_neg%';

-- EPS revisions (signed % change in estimates) -> revision
UPDATE pml.pml_df_metadata
SET category     = 'eps',
    feature_role = 'revision'
WHERE column_name LIKE 'eps_est_avg_rev_pct_fy1e_%'
   OR column_name LIKE 'eps_gaap_est_avg_rev_pct_fy1e_%';

-- =========================================================================
-- EARNINGS SURPRISES  (EPS / EBIT / EBITDA / Sales: estimate, actual, surprise%)
-- These drive EarningsBeatModel; surprise % is signed and ready-to-use.
-- =========================================================================
-- EPS actuals/estimates (historical fundamentals) -> historical
UPDATE pml.pml_df_metadata
SET category     = 'eps',
    feature_role = 'historical'
WHERE column_name SIMILAR TO 'eps_neg[0-9]+f[qy](estimate|actual)';

UPDATE pml.pml_df_metadata
SET category     = 'eps',
    feature_role = 'surprise'
WHERE column_name SIMILAR TO 'eps_neg[0-9]+f[qy]surprise_pct';

-- EBIT
UPDATE pml.pml_df_metadata
SET category     = 'ebit',
    feature_role = 'historical'
WHERE column_name SIMILAR TO 'ebit_neg[0-9]+f[qy](estimate|actual)';

UPDATE pml.pml_df_metadata
SET category     = 'ebit',
    feature_role = 'surprise'
WHERE column_name SIMILAR TO 'ebit_neg[0-9]+f[qy]surprise_pct';

-- EBITDA  (estimates/actuals + dedicated forward estimate avgs)
UPDATE pml.pml_df_metadata
SET category     = 'ebitda',
    feature_role = 'historical'
WHERE column_name SIMILAR TO 'ebitda_neg[0-9]+f[qy](estimate|actual)';

UPDATE pml.pml_df_metadata
SET category     = 'ebitda',
    feature_role = 'surprise'
WHERE column_name SIMILAR TO 'ebitda_neg[0-9]+f[qy]surprise_pct';

UPDATE pml.pml_df_metadata
SET category     = 'ebitda',
    feature_role = 'predictor'
WHERE column_name IN ('ebitda_est_avg_ntm', 'ebitda_est_avg_fy1e');

-- Sales
UPDATE pml.pml_df_metadata
SET category     = 'sales',
    feature_role = 'historical'
WHERE column_name SIMILAR TO 'sales_neg[0-9]+f[qy](estimate|actual)';

UPDATE pml.pml_df_metadata
SET category     = 'sales',
    feature_role = 'surprise'
WHERE column_name SIMILAR TO 'sales_neg[0-9]+f[qy]surprise_pct';

-- =========================================================================
-- CASH FLOW  (CFO / CFI / CFF / CapEx / FCF / FCF per share)
-- Note: issuance / repurchase of common stock are classified under
-- market_data (capital-structure / share-supply signals), not cash_flow.
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'cash_flow',
    feature_role = 'predictor'
WHERE column_name IN ('cfo_ltm', 'cfo_fq', 'cfo_fy',
                      'cfi_ltm', 'cfi_fq', 'cfi_fy',
                      'cff_ltm', 'cff_fq', 'cff_fy',
                      'capital_expenditure_ltm', 'capital_expenditure_fq', 'capital_expenditure_fy',
                      'fcf_ltm', 'fcf_fq', 'fcf_fy',
                      'fcf_per_share_ltm', 'fcf_per_share_fq', 'fcf_per_share_fy',
                      'fcf_est_avg_fy1e', 'fcf_est_avg_fy2e', 'fcf_est_avg_fy3e',
                      'fcf_est_avg_fy4e', 'fcf_est_avg_fy5e');

-- Issuance / repurchase of common stock are share-supply (capital structure)
-- signals -- bucketed under market_data alongside shares-outstanding metrics.
UPDATE pml.pml_df_metadata
SET category     = 'market_data',
    feature_role = 'predictor'
WHERE column_name IN ('issuance_common_stock_ltm', 'issuance_common_stock_fq', 'issuance_common_stock_fy',
                      'repurchase_common_stock_ltm', 'repurchase_common_stock_fq', 'repurchase_common_stock_fy',
                      'repurchase_common_stock_3yavgfq', 'repurchase_common_stock_5yavgfq');

UPDATE pml.pml_df_metadata
SET category     = 'cash_flow',
    feature_role = 'historical'
WHERE column_name LIKE 'cfo_neg%'
   OR column_name LIKE 'cfi_neg%'
   OR column_name LIKE 'cff_neg%'
   OR column_name LIKE 'capital_expenditure_neg%'
   OR column_name LIKE 'fcf_neg%'
   OR column_name LIKE 'fcf_per_share_neg%';

UPDATE pml.pml_df_metadata
SET category     = 'market_data',
    feature_role = 'historical'
WHERE column_name LIKE 'issuance_common_stock_neg%'
   OR column_name LIKE 'repurchase_common_stock_neg%';

-- =========================================================================
-- CREDIT RISK / DISTRESS  (Altman Z)
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'credit_risk',
    feature_role = 'score'
WHERE column_name IN ('altman_z_score_fy', 'altman_z_score_fq', 'altman_z_score_ltm');

UPDATE pml.pml_df_metadata
SET category     = 'credit_risk',
    feature_role = 'historical'
WHERE column_name LIKE 'altman_z_score_neg%';

-- =========================================================================
-- HEADCOUNT  (employees -- low-frequency fundamentals)
-- =========================================================================
UPDATE pml.pml_df_metadata
SET category     = 'employees',
    feature_role = 'count'
WHERE column_name IN ('full_time_employees_fq', 'full_time_employees_fy',
                      'avg_employees_5yavgfy');

UPDATE pml.pml_df_metadata
SET category     = 'employees',
    feature_role = 'historical'
WHERE column_name IN ('full_time_employees_neg1fy', 'full_time_employees_neg2fy',
                      'full_time_employees_neg3fy');

-- =========================================================================
-- DESCRIPTIONS  (per-column human-readable description)
-- =========================================================================

-- Identifiers / metadata
UPDATE pml.pml_df_metadata
SET description = 'Ticker symbol identifier'
WHERE column_name = 'ticker';
UPDATE pml.pml_df_metadata
SET description = 'International Securities Identification Number'
WHERE column_name = 'isin';
UPDATE pml.pml_df_metadata
SET description = 'Company name'
WHERE column_name = 'name';
UPDATE pml.pml_df_metadata
SET description = 'Company description / business summary'
WHERE column_name = 'description';

-- Classification
UPDATE pml.pml_df_metadata
SET description = 'Geographic region'
WHERE column_name = 'region';
UPDATE pml.pml_df_metadata
SET description = 'Country of incorporation'
WHERE column_name = 'country';
UPDATE pml.pml_df_metadata
SET description = 'Country where stock is traded'
WHERE column_name = 'trading_country';
UPDATE pml.pml_df_metadata
SET description = 'Stock exchange'
WHERE column_name = 'exchange';
UPDATE pml.pml_df_metadata
SET description = 'Reporting currency unit'
WHERE column_name = 'unit';
UPDATE pml.pml_df_metadata
SET description = 'GICS business sector'
WHERE column_name = 'sector';
UPDATE pml.pml_df_metadata
SET description = 'GICS industry classification'
WHERE column_name = 'industry';
UPDATE pml.pml_df_metadata
SET description = 'Investment style classification (growth/value/blend)'
WHERE column_name = 'style_class';
UPDATE pml.pml_df_metadata
SET description = 'Market cap size classification (large/mid/small)'
WHERE column_name = 'size_class';

-- Fiscal calendar
UPDATE pml.pml_df_metadata
SET description = 'Fiscal year end month (text)'
WHERE column_name = 'fy_end';
UPDATE pml.pml_df_metadata
SET description = 'Next earnings timing indicator'
WHERE column_name = 'next_earnings_when';
UPDATE pml.pml_df_metadata
SET description = 'Next earnings status (confirmed/estimated)'
WHERE column_name = 'next_earnings_status';
UPDATE pml.pml_df_metadata
SET description = 'Date of last data update'
WHERE column_name = 'last_updated';
UPDATE pml.pml_df_metadata
SET description = 'Most recent income statement report date'
WHERE column_name = 'income_statement_report_date';
UPDATE pml.pml_df_metadata
SET description = 'Next earnings announcement date'
WHERE column_name = 'next_earnings';
UPDATE pml.pml_df_metadata
SET description = 'Fiscal year end date'
WHERE column_name = 'fy_end_date';
UPDATE pml.pml_df_metadata
SET description = 'Next fiscal year end date'
WHERE column_name = 'next_fy_end_date';
UPDATE pml.pml_df_metadata
SET description = 'Next income statement report date'
WHERE column_name = 'next_income_statement_report_date';
UPDATE pml.pml_df_metadata
SET description = 'Expected earnings report date'
WHERE column_name = 'expected_report_date';
UPDATE pml.pml_df_metadata
SET description = 'Days until next earnings announcement (next_earnings - CURRENT_DATE)'
WHERE column_name = 'days_to_earnings';
UPDATE pml.pml_df_metadata
SET description = 'Days since most recent income statement report date (CURRENT_DATE - income_statement_report_date)'
WHERE column_name = 'earnings_report_recency';

-- Dividend
UPDATE pml.pml_df_metadata
SET description = 'Dividend payment currency'
WHERE column_name = 'dividend_record_currency';
UPDATE pml.pml_df_metadata
SET description = 'Dividend payment frequency'
WHERE column_name = 'dividend_record_frequency';
UPDATE pml.pml_df_metadata
SET description = 'Dividend announcement date'
WHERE column_name = 'dividend_record_announce_date';
UPDATE pml.pml_df_metadata
SET description = 'Dividend payable date'
WHERE column_name = 'dividend_record_payable_date';
UPDATE pml.pml_df_metadata
SET description = 'Dividend record date'
WHERE column_name = 'dividend_record_record_date';
UPDATE pml.pml_df_metadata
SET description = 'Dividend ex-dividend date'
WHERE column_name = 'dividend_record_ex_date';
UPDATE pml.pml_df_metadata
SET description = 'Most recent dividend amount per share'
WHERE column_name = 'dividend_record_amount';
UPDATE pml.pml_df_metadata
SET description = 'Consecutive years of dividend payments'
WHERE column_name = 'dividend_streak';
UPDATE pml.pml_df_metadata
SET description = 'Dividend per share (Last Twelve Months)'
WHERE column_name = 'dividend_per_share_ltm';
UPDATE pml.pml_df_metadata
SET description = 'Dividend per share (Fiscal Quarter)'
WHERE column_name = 'dividend_per_share_fq';
UPDATE pml.pml_df_metadata
SET description = 'Dividend per share (Fiscal Year)'
WHERE column_name = 'dividend_per_share_fy';
UPDATE pml.pml_df_metadata
SET description = 'Indicated dividend yield (annualized)'
WHERE column_name = 'div_yield_ind';
UPDATE pml.pml_df_metadata
SET description = 'Dividend yield (Last Twelve Months)'
WHERE column_name = 'div_yield_ltm';
UPDATE pml.pml_df_metadata
SET description = 'Dividend yield trailing twelve months'
WHERE column_name = 'div_yield_ttm';
UPDATE pml.pml_df_metadata
SET description = 'Dividend yield next twelve months (forward)'
WHERE column_name = 'div_yield_ntm';
UPDATE pml.pml_df_metadata
SET description = 'Dividend yield 5-year average (LTM)'
WHERE column_name = 'div_yield_5yavgltm';
UPDATE pml.pml_df_metadata
SET description = 'Common dividends paid (Last Twelve Months)'
WHERE column_name = 'common_dividends_paid_ltm';
UPDATE pml.pml_df_metadata
SET description = 'Common dividends paid (Fiscal Year)'
WHERE column_name = 'common_dividends_paid_fy';
UPDATE pml.pml_df_metadata
SET description = 'Buyback yield (Last Twelve Months)'
WHERE column_name = 'buyback_yield_ltm';
UPDATE pml.pml_df_metadata
SET description = 'Lagged indicated dividend yield (prior fiscal year snapshot)'
WHERE column_name LIKE 'div_yield_neg%fyind';
UPDATE pml.pml_df_metadata
SET description = 'Lagged dividend per share (prior fiscal quarter)'
WHERE column_name LIKE 'dividend_per_share_neg%fqfq';
UPDATE pml.pml_df_metadata
SET description = 'Lagged dividend per share (prior fiscal year)'
WHERE column_name SIMILAR TO 'dividend_per_share_neg[0-9]+fy';

-- Market data
UPDATE pml.pml_df_metadata
SET description = 'Market capitalization'
WHERE column_name = 'market_cap';
UPDATE pml.pml_df_metadata
SET description = 'Enterprise value'
WHERE column_name = 'enterprise_value';
UPDATE pml.pml_df_metadata
SET description = 'Trading volume in shares'
WHERE column_name = 'volume_shrs';
UPDATE pml.pml_df_metadata
SET description = 'Relative trading volume ratio'
WHERE column_name = 'rel_volume';
UPDATE pml.pml_df_metadata
SET description = 'Shares outstanding'
WHERE column_name = 'shrs_out';
UPDATE pml.pml_df_metadata
SET description = 'Shares outstanding 3-year average'
WHERE column_name = 'shrs_out_3yavg';
UPDATE pml.pml_df_metadata
SET description = 'Shares outstanding 5-year average'
WHERE column_name = 'shrs_out_5yavg';
UPDATE pml.pml_df_metadata
SET description = 'Market cap rank within country'
WHERE column_name = 'market_cap_country_r';
UPDATE pml.pml_df_metadata
SET description = 'Lagged enterprise value (prior period)'
WHERE column_name LIKE 'enterprise_value_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged shares outstanding (prior period)'
WHERE column_name LIKE 'shrs_out_neg%';

-- Historical prices
UPDATE pml.pml_df_metadata
SET description = 'Last trading price'
WHERE column_name = 'last_price';
UPDATE pml.pml_df_metadata
SET description = 'Lagged trading price (prior period offset indicated by suffix)'
WHERE column_name LIKE 'price_%_ago';

-- Technical / momentum
UPDATE pml.pml_df_metadata
SET description = '52-week high (adjusted)'
WHERE column_name = 'w_52high_adj';
UPDATE pml.pml_df_metadata
SET description = '52-week low (adjusted)'
WHERE column_name = 'w_52low_adj';
UPDATE pml.pml_df_metadata
SET description = '20-day exponential moving average'
WHERE column_name = 'ema_20d';
UPDATE pml.pml_df_metadata
SET description = '50-day exponential moving average'
WHERE column_name = 'ema_50d';
UPDATE pml.pml_df_metadata
SET description = '100-day exponential moving average'
WHERE column_name = 'ema_100d';
UPDATE pml.pml_df_metadata
SET description = '250-day exponential moving average'
WHERE column_name = 'ema_250d';
UPDATE pml.pml_df_metadata
SET description = '3-month price change percentage'
WHERE column_name = 'price_chg_pct_3m';
UPDATE pml.pml_df_metadata
SET description = '1-day price change percentage'
WHERE column_name = 'one_day_pct';

-- Volatility / beta
UPDATE pml.pml_df_metadata
SET description = 'Realized volatility 1 month'
WHERE column_name = 'volatility_1m';
UPDATE pml.pml_df_metadata
SET description = 'Realized volatility 3 months'
WHERE column_name = 'volatility_3m';
UPDATE pml.pml_df_metadata
SET description = 'Realized volatility 6 months'
WHERE column_name = 'volatility_6m';
UPDATE pml.pml_df_metadata
SET description = 'Realized volatility 1 year'
WHERE column_name = 'volatility_1y';
UPDATE pml.pml_df_metadata
SET description = 'Beta coefficient (1Y window)'
WHERE column_name = 'beta_1y';
UPDATE pml.pml_df_metadata
SET description = 'Beta coefficient (2Y window)'
WHERE column_name = 'beta_2y';
UPDATE pml.pml_df_metadata
SET description = 'Beta coefficient (5Y window)'
WHERE column_name = 'beta_5y';

-- Total return
UPDATE pml.pml_df_metadata
SET description = 'Total return year-to-date'
WHERE column_name = 'total_return_ytd';
UPDATE pml.pml_df_metadata
SET description = 'Total return 5 years'
WHERE column_name = 'total_return_5y';
UPDATE pml.pml_df_metadata
SET description = 'Total return 10 years'
WHERE column_name = 'total_return_10y';
UPDATE pml.pml_df_metadata
SET description = 'Total return CAGR 3 years'
WHERE column_name = 'tot_return_pct_cagr_3y';
UPDATE pml.pml_df_metadata
SET description = 'Total return CAGR 10 years'
WHERE column_name = 'tot_return_pct_cagr_10y';

-- Analyst price targets
UPDATE pml.pml_df_metadata
SET description = 'Analyst consensus price target'
WHERE column_name = 'price_target';
UPDATE pml.pml_df_metadata
SET description = 'Low analyst price target'
WHERE column_name = 'price_target_low';
UPDATE pml.pml_df_metadata
SET description = 'Median analyst price target'
WHERE column_name = 'price_target_median';
UPDATE pml.pml_df_metadata
SET description = 'High analyst price target'
WHERE column_name = 'price_target_high';
UPDATE pml.pml_df_metadata
SET description = 'Standard deviation of analyst price targets'
WHERE column_name = 'price_target_stddev';
UPDATE pml.pml_df_metadata
SET description = 'Implied upside vs. average analyst target (%)'
WHERE column_name = 'target_pct_avg';
UPDATE pml.pml_df_metadata
SET description = 'Implied upside vs. median analyst target (%)'
WHERE column_name = 'target_pct_med';
UPDATE pml.pml_df_metadata
SET description = 'Implied upside vs. low analyst target (%)'
WHERE column_name = 'target_pct_low';
UPDATE pml.pml_df_metadata
SET description = 'Implied upside vs. high analyst target (%)'
WHERE column_name = 'target_pct_high';
UPDATE pml.pml_df_metadata
SET description = 'Number of analyst price targets'
WHERE column_name = 'price_target_num';
UPDATE pml.pml_df_metadata
SET description = 'Lagged analyst price target snapshot (offset indicated by suffix)'
WHERE column_name LIKE 'price_target_%_ago'
  AND column_name NOT LIKE 'price_target_num_%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged count of analyst price targets (offset indicated by suffix)'
WHERE column_name LIKE 'price_target_num_%_ago';

-- Analyst ratings
UPDATE pml.pml_df_metadata
SET description = 'Consensus analyst rating score'
WHERE column_name = 'analyst_rating';
UPDATE pml.pml_df_metadata
SET description = 'Number of strong sell ratings'
WHERE column_name = 'num_strong_sell_ratings';
UPDATE pml.pml_df_metadata
SET description = 'Number of strong buy ratings'
WHERE column_name = 'num_strong_buys_ratings';
UPDATE pml.pml_df_metadata
SET description = 'Number of hold ratings'
WHERE column_name = 'num_hold_ratings';
UPDATE pml.pml_df_metadata
SET description = 'Number of buy ratings'
WHERE column_name = 'num_buys_ratings';
UPDATE pml.pml_df_metadata
SET description = 'Number of sell ratings'
WHERE column_name = 'num_sell_ratings';
UPDATE pml.pml_df_metadata
SET description = 'Number of no-opinion ratings'
WHERE column_name = 'num_no_opinion_ratings';

-- Valuation (pattern-based)
UPDATE pml.pml_df_metadata
SET description = 'Price-to-earnings ratio (period indicated by suffix)'
WHERE column_name LIKE 'p_e_%'
  AND column_name NOT LIKE 'p_e_neg%'
  AND column_name NOT LIKE 'peg_%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged price-to-earnings ratio (prior period)'
WHERE column_name LIKE 'p_e_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Price-to-book ratio (period indicated by suffix)'
WHERE column_name LIKE 'p_b_%'
  AND column_name NOT LIKE 'p_b_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged price-to-book ratio (prior period)'
WHERE column_name LIKE 'p_b_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Enterprise value to sales (period indicated by suffix)'
WHERE column_name LIKE 'ev_sales_%'
  AND column_name NOT LIKE 'ev_sales_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged enterprise value to sales (prior period)'
WHERE column_name LIKE 'ev_sales_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Enterprise value to EBITDA (period indicated by suffix)'
WHERE column_name LIKE 'ev_ebitda_%'
  AND column_name NOT LIKE 'ev_ebitda_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged enterprise value to EBITDA (prior period)'
WHERE column_name LIKE 'ev_ebitda_neg%';
UPDATE pml.pml_df_metadata
SET description = 'PEG ratio (P/E to growth, period indicated by suffix)'
WHERE column_name LIKE 'peg_%'
  AND column_name NOT LIKE 'peg_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged PEG ratio (prior period)'
WHERE column_name LIKE 'peg_neg%';

-- Profitability
UPDATE pml.pml_df_metadata
SET description = 'Gross profit margin % (period indicated by suffix)'
WHERE column_name LIKE 'gross_profit_margin_pct_%'
  AND column_name NOT LIKE 'gross_profit_margin_pct_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged gross profit margin % (prior period)'
WHERE column_name LIKE 'gross_profit_margin_pct_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged gross profit level (prior period)'
WHERE column_name LIKE 'gross_profit_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Return on assets % (Last Twelve Months)'
WHERE column_name = 'return_on_assets_roa_pct_ltm';
UPDATE pml.pml_df_metadata
SET description = 'Return on assets % (Fiscal Year)'
WHERE column_name = 'return_on_assets_roa_pct_fy';

-- EPS
UPDATE pml.pml_df_metadata
SET description = 'Adjusted EPS (period indicated by suffix)'
WHERE column_name LIKE 'eps_adj_%'
  AND column_name NOT LIKE 'eps_adj_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged adjusted EPS (prior period)'
WHERE column_name LIKE 'eps_adj_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Normalized EPS estimate average (period indicated by suffix)'
WHERE column_name LIKE 'eps_norm_est_avg_%';
UPDATE pml.pml_df_metadata
SET description = 'GAAP EPS estimate average (period indicated by suffix)'
WHERE column_name LIKE 'eps_gaap_est_avg_%';
UPDATE pml.pml_df_metadata
SET description = 'Net basic EPS (period indicated by suffix)'
WHERE column_name LIKE 'net_eps_basic_%'
  AND column_name NOT LIKE 'net_eps_basic_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged net basic EPS (prior period)'
WHERE column_name LIKE 'net_eps_basic_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Basic EPS continuing operations (period indicated by suffix)'
WHERE column_name LIKE 'basic_eps_cont_%'
  AND column_name NOT LIKE 'basic_eps_cont_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged basic EPS continuing operations (prior period)'
WHERE column_name LIKE 'basic_eps_cont_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Number of normalized EPS estimates for FY1'
WHERE column_name = 'eps_norm_est_num_fy1e';

-- EPS revisions
UPDATE pml.pml_df_metadata
SET description = 'Normalized EPS estimate revision % (FY1E, horizon in suffix)'
WHERE column_name LIKE 'eps_est_avg_rev_pct_fy1e_%';
UPDATE pml.pml_df_metadata
SET description = 'GAAP EPS estimate revision % (FY1E, horizon in suffix)'
WHERE column_name LIKE 'eps_gaap_est_avg_rev_pct_fy1e_%';

-- Earnings surprises (EPS / EBIT / EBITDA / Sales)
UPDATE pml.pml_df_metadata
SET description = 'EPS estimate for prior period (period indicated by suffix)'
WHERE column_name SIMILAR TO 'eps_neg[0-9]+f[qy]estimate';
UPDATE pml.pml_df_metadata
SET description = 'EPS actual for prior period (period indicated by suffix)'
WHERE column_name SIMILAR TO 'eps_neg[0-9]+f[qy]actual';
UPDATE pml.pml_df_metadata
SET description = 'EPS surprise % for prior period (actual vs. estimate)'
WHERE column_name SIMILAR TO 'eps_neg[0-9]+f[qy]surprise_pct';
UPDATE pml.pml_df_metadata
SET description = 'EBIT estimate for prior period (period indicated by suffix)'
WHERE column_name SIMILAR TO 'ebit_neg[0-9]+f[qy]estimate';
UPDATE pml.pml_df_metadata
SET description = 'EBIT actual for prior period (period indicated by suffix)'
WHERE column_name SIMILAR TO 'ebit_neg[0-9]+f[qy]actual';
UPDATE pml.pml_df_metadata
SET description = 'EBIT surprise % for prior period (actual vs. estimate)'
WHERE column_name SIMILAR TO 'ebit_neg[0-9]+f[qy]surprise_pct';
UPDATE pml.pml_df_metadata
SET description = 'EBITDA estimate for prior period (period indicated by suffix)'
WHERE column_name SIMILAR TO 'ebitda_neg[0-9]+f[qy]estimate';
UPDATE pml.pml_df_metadata
SET description = 'EBITDA actual for prior period (period indicated by suffix)'
WHERE column_name SIMILAR TO 'ebitda_neg[0-9]+f[qy]actual';
UPDATE pml.pml_df_metadata
SET description = 'EBITDA surprise % for prior period (actual vs. estimate)'
WHERE column_name SIMILAR TO 'ebitda_neg[0-9]+f[qy]surprise_pct';
UPDATE pml.pml_df_metadata
SET description = 'EBITDA estimate average (Next Twelve Months)'
WHERE column_name = 'ebitda_est_avg_ntm';
UPDATE pml.pml_df_metadata
SET description = 'EBITDA estimate average (FY1)'
WHERE column_name = 'ebitda_est_avg_fy1e';
UPDATE pml.pml_df_metadata
SET description = 'Sales estimate for prior period (period indicated by suffix)'
WHERE column_name SIMILAR TO 'sales_neg[0-9]+f[qy]estimate';
UPDATE pml.pml_df_metadata
SET description = 'Sales actual for prior period (period indicated by suffix)'
WHERE column_name SIMILAR TO 'sales_neg[0-9]+f[qy]actual';
UPDATE pml.pml_df_metadata
SET description = 'Sales surprise % for prior period (actual vs. estimate)'
WHERE column_name SIMILAR TO 'sales_neg[0-9]+f[qy]surprise_pct';

-- Cash flow
UPDATE pml.pml_df_metadata
SET description = 'Cash from operations (period indicated by suffix)'
WHERE column_name LIKE 'cfo_%'
  AND column_name NOT LIKE 'cfo_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged cash from operations (prior period)'
WHERE column_name LIKE 'cfo_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Cash from investing (period indicated by suffix)'
WHERE column_name LIKE 'cfi_%'
  AND column_name NOT LIKE 'cfi_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged cash from investing (prior period)'
WHERE column_name LIKE 'cfi_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Cash from financing (period indicated by suffix)'
WHERE column_name LIKE 'cff_%'
  AND column_name NOT LIKE 'cff_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged cash from financing (prior period)'
WHERE column_name LIKE 'cff_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Capital expenditure (period indicated by suffix)'
WHERE column_name LIKE 'capital_expenditure_%'
  AND column_name NOT LIKE 'capital_expenditure_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged capital expenditure (prior period)'
WHERE column_name LIKE 'capital_expenditure_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Free cash flow (period indicated by suffix)'
WHERE column_name LIKE 'fcf_%'
  AND column_name NOT LIKE 'fcf_neg%'
  AND column_name NOT LIKE 'fcf_per_share_%'
  AND column_name NOT LIKE 'fcf_est_%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged free cash flow (prior period)'
WHERE column_name LIKE 'fcf_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Free cash flow per share (period indicated by suffix)'
WHERE column_name LIKE 'fcf_per_share_%'
  AND column_name NOT LIKE 'fcf_per_share_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged free cash flow per share (prior period)'
WHERE column_name LIKE 'fcf_per_share_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Free cash flow estimate average (forward FY indicated by suffix)'
WHERE column_name LIKE 'fcf_est_avg_%';
UPDATE pml.pml_df_metadata
SET description = 'Common stock issuance (period indicated by suffix)'
WHERE column_name LIKE 'issuance_common_stock_%'
  AND column_name NOT LIKE 'issuance_common_stock_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged common stock issuance (prior period)'
WHERE column_name LIKE 'issuance_common_stock_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Common stock repurchases (period indicated by suffix)'
WHERE column_name LIKE 'repurchase_common_stock_%'
  AND column_name NOT LIKE 'repurchase_common_stock_neg%';
UPDATE pml.pml_df_metadata
SET description = 'Lagged common stock repurchases (prior period)'
WHERE column_name LIKE 'repurchase_common_stock_neg%';

-- Credit risk / Altman Z
UPDATE pml.pml_df_metadata
SET description = 'Altman Z-Score (Fiscal Year) - distress risk indicator'
WHERE column_name = 'altman_z_score_fy';
UPDATE pml.pml_df_metadata
SET description = 'Altman Z-Score (Fiscal Quarter) - distress risk indicator'
WHERE column_name = 'altman_z_score_fq';
UPDATE pml.pml_df_metadata
SET description = 'Altman Z-Score (LTM) - distress risk indicator'
WHERE column_name = 'altman_z_score_ltm';
UPDATE pml.pml_df_metadata
SET description = 'Lagged Altman Z-Score (prior period)'
WHERE column_name LIKE 'altman_z_score_neg%';

-- Employees
UPDATE pml.pml_df_metadata
SET description = 'Full time employees (Fiscal Quarter)'
WHERE column_name = 'full_time_employees_fq';
UPDATE pml.pml_df_metadata
SET description = 'Full time employees (Fiscal Year)'
WHERE column_name = 'full_time_employees_fy';
UPDATE pml.pml_df_metadata
SET description = 'Average employees 5-year average (Fiscal Year)'
WHERE column_name = 'avg_employees_5yavgfy';
UPDATE pml.pml_df_metadata
SET description = 'Lagged full time employees (prior fiscal year)'
WHERE column_name LIKE 'full_time_employees_neg%fy';

-- Fallback: any column still without a description gets a generated one.
UPDATE pml.pml_df_metadata
SET description = INITCAP(REPLACE(column_name, '_', ' '))
WHERE description IS NULL
   OR description = '';

-- =============================================================================
-- PYMC ALIGNMENT  (pm.Data containers / InferenceData modeling)
-- =============================================================================
-- The legacy `feature_role` vocabulary is data-centric. To drive PyMC models
-- and ArviZ InferenceData directly from SQL, we add two columns:
--
--   * `pymc_role`     -> which kind of `pm.Data` container the column maps to
--                        when fed into a PyMC model. Vocabulary aligned with
--                        the pymc.Data primer and the project's
--                        `MODEL_FEATURE_CONTAINERS` registry in
--                        `pymc_expected_returns_model.ipynb`:
--
--                          coord             : value used to label an ArviZ
--                                              dim coord (e.g. `isin`,
--                                              `sector`, `region`). Lands in
--                                              `idata.constant_data` /
--                                              `idata.posterior.coords`.
--                          index             : integer index that selects a
--                                              hierarchical level inside a
--                                              PyMC model (sector_idx,
--                                              industry_idx). Stored as
--                                              `pm.Data` constant.
--                          observed          : observed likelihood input
--                                              (`pm.Normal(..., observed=y)`).
--                                              Lands in
--                                              `idata.observed_data`.
--                          mutable_predictor : numeric predictor passed as
--                                              `pm.Data("x", x)`; supports
--                                              `pm.set_data` for OOS
--                                              prediction. Lands in
--                                              `idata.constant_data` and is
--                                              consumed by feature_alias
--                                              registries.
--                          constant_data     : auxiliary numeric input that
--                                              should be carried into
--                                              `idata.constant_data` but is
--                                              NOT a feature_alias (counts,
--                                              denominators, weights).
--                          derived_input     : lagged / historical level that
--                                              must be transformed (drift,
--                                              momentum, streak) BEFORE
--                                              entering a `pm.Data`
--                                              container. Never fed raw.
--                          excluded          : free text / identifier /
--                                              calendar metadata that is
--                                              never wrapped in `pm.Data`.
--
--   * `model_targets` -> TEXT[] of PyMC model names from
--                        `probabilistic_ml_model.pymc_models` that consume
--                        this column. Keys mirror `MODEL_FEATURE_CONTAINERS`:
--                          earnings_beat, price_target, kalman_pt, dcf_pt,
--                          dividend_safety, credit_risk, accounting_anomaly.
--                        The notebook can rebuild its `(category,
--                        feature_alias)` registry directly from a SQL view
--                        over this column.
-- =============================================================================

-- 1. Schema additions (idempotent).
ALTER TABLE pml.pml_df_metadata
	ADD COLUMN IF NOT EXISTS pymc_role     TEXT,
	ADD COLUMN IF NOT EXISTS model_targets TEXT[] NOT NULL DEFAULT ARRAY []::TEXT[];

-- 2. Default pymc_role derived deterministically from feature_role so that
--    every row gets a non-NULL value. Subsequent UPDATEs refine specific
--    columns (e.g. promote `sector` to `coord`, demote `historical` lags to
--    `derived_input`).
UPDATE pml.pml_df_metadata
SET pymc_role = CASE feature_role
	                WHEN 'id' THEN 'coord'
	                WHEN 'categorical' THEN 'coord'
	                WHEN 'metadata' THEN 'excluded'
	                WHEN 'date' THEN 'excluded'
	                WHEN 'historical' THEN 'derived_input'
	                WHEN 'count' THEN 'constant_data'
	                WHEN 'target' THEN 'observed'
	                WHEN 'predictor' THEN 'mutable_predictor'
	                WHEN 'score' THEN 'mutable_predictor'
	                WHEN 'surprise' THEN 'mutable_predictor'
	                WHEN 'revision' THEN 'mutable_predictor'
	                ELSE 'excluded' END
WHERE pymc_role IS NULL
   OR pymc_role = '';

-- 3. Refine hierarchical coord/index columns.
--    `isin` is the primary observation coord; the hierarchy levels are
--    coords AND backing arrays for `*_idx` integer indices used inside
--    `build_nested_logit_normal_rates` (see _hierarchy.py).
UPDATE pml.pml_df_metadata
SET pymc_role = 'coord'
WHERE column_name IN ('isin', 'ticker',
                      'region', 'country', 'trading_country', 'exchange',
                      'unit', 'sector', 'industry', 'style_class', 'size_class');

-- 4. Observed likelihood inputs (model labels / targets).
--    PriceTargetModel / DCF use price_target* fields; total_return_* are
--    realised-return labels for backtesting / posterior predictive checks.
UPDATE pml.pml_df_metadata
SET pymc_role = 'observed'
WHERE column_name IN ('price_target', 'price_target_median', 'price_target_low',
                      'price_target_high', 'price_target_stddev',
                      'target_pct_avg', 'target_pct_med', 'target_pct_low', 'target_pct_high',
                      'total_return_ytd', 'total_return_5y', 'total_return_10y',
                      'tot_return_pct_cagr_3y', 'tot_return_pct_cagr_10y');

-- 5. `n_total` / `n_beats` style discrete inputs and analyst counts stay as
--    constant_data (not predictors but carried into idata.constant_data).
UPDATE pml.pml_df_metadata
SET pymc_role = 'constant_data'
WHERE column_name IN ('price_target_num',
                      'num_strong_sell_ratings', 'num_strong_buys_ratings',
                      'num_hold_ratings', 'num_buys_ratings', 'num_sell_ratings',
                      'num_no_opinion_ratings',
                      'eps_norm_est_num_fy1e',
                      'market_cap_country_r',
                      'full_time_employees_fq', 'full_time_employees_fy',
                      'avg_employees_5yavgfy');

-- 6. Lagged historical levels always feed downstream derivation, never raw.
UPDATE pml.pml_df_metadata
SET pymc_role = 'derived_input'
WHERE feature_role = 'historical';

-- =============================================================================
-- MODEL_TARGETS WIRING  (mirrors MODEL_FEATURE_CONTAINERS in the notebook)
-- =============================================================================
-- Helper: append a model tag to `model_targets` only if not already present
-- (set semantics) so re-runs stay idempotent.
--   model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY['<tag>'])))
-- =============================================================================

-- 7a. EarningsBeatBayesian: surprises + revisions + EPS estimates + sector.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['earnings_beat'])))
WHERE category IN ('eps', 'eps', 'sales', 'ebit', 'ebitda')
   OR column_name IN ('sector', 'industry', 'eps_norm_est_avg_fy1e',
                      'eps_norm_est_avg_ntm', 'eps_norm_est_num_fy1e',
                      'days_to_earnings', 'earnings_report_recency');

-- 7b. PriceTargetAchievement: analyst target levels, dispersion, counts +
--     analyst ratings + valuation predictors that drive achievement priors.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['price_target'])))
WHERE category IN ('analyst_targets', 'analyst_ratings', 'valuation')
   OR column_name IN ('last_price', 'sector', 'industry');

-- 7c. KalmanFilterPriceTarget: lagged target snapshots, lagged prices,
--     dispersion (stddev) -- explicitly consumes derived_input lags.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['kalman_pt'])))
WHERE column_name LIKE 'price_target_%_ago'
   OR column_name LIKE 'price_%_ago'
   OR column_name IN ('price_target', 'price_target_stddev', 'last_price',
                      'volatility_1m', 'volatility_3m', 'volatility_6m', 'volatility_1y');

-- 7d. DCFPriceTarget: cash-flow predictors, FCF estimates, valuation,
--     EV/EBITDA, profitability margins.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['dcf_pt'])))
WHERE category IN ('cash_flow', 'profitability')
   OR column_name IN ('enterprise_value', 'market_cap', 'shrs_out',
                      'ev_sales_ltm', 'ev_sales_ntm',
                      'ev_ebitda_ltm', 'ev_ebitda_ntm', 'ev_ebitda_est_fy1',
                      'price_target');

-- 7e. DividendSafetyBayesian: dividend block + FCF + leverage signals.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['dividend_safety'])))
WHERE category = 'dividend'
   OR column_name IN ('fcf_ltm', 'fcf_fy', 'fcf_per_share_ltm',
                      'common_dividends_paid_ltm', 'common_dividends_paid_fy',
                      'altman_z_score_ltm', 'altman_z_score_fy',
                      'sector', 'industry');

-- 7f. CreditRiskBayesian: Altman Z + balance-sheet proxies + sector hierarchy.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['credit_risk'])))
WHERE category = 'credit_risk'
   OR column_name IN ('fcf_ltm', 'fcf_fy', 'cfo_ltm', 'cfo_fy',
                      'capital_expenditure_ltm', 'capital_expenditure_fy',
                      'enterprise_value', 'market_cap',
                      'sector', 'industry');

-- 7g. AccountingAnomalyBayesian: EPS adj/basic levels + cash-flow + accruals
--     proxies + profitability + sales -- the multi-layered anomaly stack.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['accounting_anomaly'])))
WHERE category IN ('eps', 'profitability', 'cash_flow', 'sales', 'ebit', 'ebitda')
   OR column_name IN ('sector', 'industry');

-- =============================================================================
-- 7h. ENHANCED FEATURE-COVERAGE GAP FILLERS
-- =============================================================================
-- The materialized views in pml_feature_catalogue.sql were extended with new
-- mutable_predictor columns sourced from raw pml.pml_df fields that previously
-- fell outside the broad category-based wiring above. The updates below ensure
-- every such raw input is tagged with the consuming model so it flows into
-- vw_pymc_feature_catalogue / vw_pymc_feature_aliases and ultimately into the
-- notebook's MODEL_FEATURE_CONTAINERS registry.
-- =============================================================================

-- 7h.1 EarningsBeatBayesian: pull in calendar-derived signals and the FQ1E EPS
--      estimate that drive feat_days_to_earnings / feat_report_recency /
--      feat_next_earnings_status / feat_eps_fq1e in mv_pymc_earnings_beat.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['earnings_beat'])))
WHERE column_name IN ('days_to_earnings',
                      'earnings_report_recency',
                      'next_earnings_status',
                      'eps_norm_est_avg_fq1e');

-- 7h.2 PriceTargetAchievement: range-position (52w high/low), 3m PT-momentum
--      and 3m coverage-change anchors, plus volatility_3m proxy. Also wires
--      the fiscal-calendar date columns + their pre-computed day-count
--      horizons into the price_target model so the notebook can drive the
--      MvGaussianRandomWalk time axis directly from mv_pymc_price_target.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['price_target'])))
WHERE column_name IN ('w_52high_adj',
                      'w_52low_adj',
                      'price_target_3m_ago',
                      'price_target_num_3m_ago',
                      'volatility_3m',
	-- MvGRW time-axis anchors
                      'income_statement_report_date',
                      'next_earnings',
                      'fy_end_date',
                      'next_income_statement_report_date',
                      'next_fy_end_date',
                      'expected_report_date',
	-- Derived day-count horizons (numeric, ready for pm.Data)
                      'days_to_next_earnings',
                      'days_since_last_report',
                      'days_to_next_fy_end',
                      'days_to_next_report',
                      'days_to_expected_report',
                      'days_to_fy_end');

-- Day-count horizons are ready-to-use numeric predictors -> mutable_predictor.
UPDATE pml.pml_df_metadata
SET pymc_role    = 'mutable_predictor',
    category     = 'fiscal_calendar',
    feature_role = 'predictor'
WHERE column_name IN ('days_to_next_earnings',
                      'days_since_last_report',
                      'days_to_next_fy_end',
                      'days_to_next_report',
                      'days_to_expected_report',
                      'days_to_fy_end');

-- Raw DATE columns: promote pymc_role from 'excluded' to 'coord' so the
-- notebook can register them as the `time` dimension for the MvGRW panel
-- (one slice per fiscal-period anchor).
UPDATE pml.pml_df_metadata
SET pymc_role = 'coord'
WHERE column_name IN ('income_statement_report_date',
                      'next_earnings',
                      'fy_end_date',
                      'next_income_statement_report_date',
                      'next_fy_end_date',
                      'expected_report_date');

-- 7h.2a PriceTargetAchievement: realised-vs-target accuracy block. The 1Y-ago
--       target snapshots (level / low / high / median), the current median, and
--       the trailing analyst-count snapshots drive feat_pt_achievement_1y /
--       feat_pt_accuracy_1y / feat_pt_optimism_bias / feat_pt_range_hit_rate /
--       feat_pt_median_vs_mean_spread / feat_pt_high_low_convergence_1y /
--       feat_analyst_count_stability in mv_pymc_price_target.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['price_target'])))
WHERE column_name IN ('price_target_1y_ago',
                      'price_target_low_1y_ago',
                      'price_target_high_1y_ago',
                      'price_target_median_1y_ago',
                      'price_target_median',
                      'price_target'
                      );

-- 7h.3 KalmanFilterPriceTarget: full stddev-trail snapshots and analyst-range
--      bounds used in feat_pt_noise_drift / feat_pt_range_norm. The high / low /
--      median / num *_ago target trails that now feed feat_pt_high_drift /
--      feat_pt_low_drift / feat_pt_median_drift / feat_coverage_drift are
--      already wired to kalman_pt via the 'price_target_%_ago' / 'price_%_ago'
--      LIKE patterns in section 7c above.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['kalman_pt'])))
WHERE column_name IN ('price_target_high',
                      'price_target_low',
                      'price_target_median',
                      'price_target_num',
                      'price_target_stddev_1w_ago',
                      'price_target_stddev_1m_ago',
                      'price_target_stddev_3m_ago',
                      'price_target_stddev_6m_ago',
                      'price_target_stddev_1y_ago',
                      'total_return_ytd');

-- 7h.3a KalmanFilterPriceTarget: fiscal-calendar anchors + derived day-count
--       horizons (mirrors the price_target wiring in 7h.2). These give the
--       marginalized GaussianRandomWalk a real, irregular time axis so the
--       process variance can be scaled by actual elapsed time between the
--       *_ago observations (KalmanFilterModel._resolve_time_deltas). The raw
--       DATE columns are already pymc_role='coord' and the day-count columns
--       pymc_role='mutable_predictor' (set in 7h.2); here we only extend their
--       model_targets so they surface in vw_pymc_feature_catalogue for kalman_pt.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['kalman_pt'])))
WHERE column_name IN ('income_statement_report_date',
                      'next_earnings',
                      'fy_end_date',
                      'next_income_statement_report_date',
                      'next_fy_end_date',
                      'expected_report_date',
                      'days_to_next_earnings',
                      'days_since_last_report',
                      'days_to_next_fy_end',
                      'days_to_next_report',
                      'days_to_expected_report',
                      'days_to_fy_end');

-- 7h.4 DCFPriceTarget: terminal FCF estimates, historical CAGRs, PEG / EV-sales
--      / ROA / beta anchors used in feat_fcf_terminal_growth / feat_tr_cagr_* /
--      feat_peg_ntm / feat_roa_ltm / feat_beta_5y.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['dcf_pt'])))
WHERE column_name IN ('fcf_est_avg_fy4e',
                      'fcf_est_avg_fy5e',
                      'tot_return_pct_cagr_3y',
                      'tot_return_pct_cagr_10y',
                      'peg_ntm',
                      'ev_sales_ltm',
                      'return_on_assets_roa_pct_ltm',
                      'gross_profit_margin_pct_ltm',
                      'beta_5y');

-- 7h.5 DividendSafetyBayesian: longer-run DPS history, frequency, ROA, EPS
--      level for the payout ratio, buyback level + buybacks needed for
--      feat_total_yield / feat_eps_payout_ratio / feat_dps_growth_{3,5}y.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['dividend_safety'])))
WHERE column_name IN ('dividend_record_frequency',
                      'dividend_per_share_neg3fy',
                      'dividend_per_share_neg5fy',
                      'net_eps_basic_ltm',
                      'buyback_yield_ltm',
                      'repurchase_common_stock_ltm',
                      'return_on_assets_roa_pct_ltm',
                      'cfo_ltm');

-- 7h.6 CreditRiskBayesian: multi-year Altman trajectory, financing-flow burn,
--      net equity issuance, headcount trend, valuation/leverage proxies.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['credit_risk'])))
WHERE column_name IN ('altman_z_score_neg3fy',
                      'cff_ltm',
                      'issuance_common_stock_ltm',
                      'repurchase_common_stock_ltm',
                      'market_cap',
                      'full_time_employees_fy',
                      'full_time_employees_neg1fy',
                      'p_b_ltm',
                      'volatility_6m',
                      'volatility_1y',
                      'beta_2y');

-- 7h.7 AccountingAnomalyBayesian: cash-flow composition, share-supply &
--      Beneish-style dilution, employee productivity, FCF-vs-EPS earnings
--      quality, PEG anchor.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(model_targets || ARRAY ['accounting_anomaly'])))
WHERE column_name IN ('cfi_ltm',
                      'cff_ltm',
                      'shrs_out',
                      'shrs_out_neg1fy',
                      'issuance_common_stock_ltm',
                      'market_cap',
                      'full_time_employees_fy',
                      'full_time_employees_neg1fy',
                      'fcf_per_share_ltm',
                      'peg_ntm');

-- 7h.8 Promote pymc_role for the new calendar-derived numeric predictors so
--      they land in idata.constant_data as mutable predictors (not excluded).
UPDATE pml.pml_df_metadata
SET pymc_role = 'mutable_predictor'
WHERE column_name IN ('days_to_earnings', 'earnings_report_recency')
  AND pymc_role <> 'mutable_predictor';

-- 7h.9 next_earnings_status is a low-cardinality categorical signal feeding
--      feat_next_earnings_status; encode it as a coord (one-hot upstream in
--      PyMC) rather than the default 'excluded' for categoricals tied to
--      fiscal calendar text.
UPDATE pml.pml_df_metadata
SET pymc_role = 'coord'
WHERE column_name = 'next_earnings_status';

-- 7h.10 dividend_record_frequency feeds feat_div_frequency. Keep it as a
--       coord (low-cardinality categorical) so the dividend_safety model can
--       attach it via constant_data after integer encoding.
UPDATE pml.pml_df_metadata
SET pymc_role = 'coord'
WHERE column_name = 'dividend_record_frequency';

-- ---------------------------------------------------------------------------
-- Validation: highlight any column that did NOT get a category assigned
-- (still has the seeded default 'n/a'). After the script runs cleanly this
-- result set should be empty.
-- ---------------------------------------------------------------------------
-- SELECT column_name, data_type, ordinal_position
-- FROM pml.pml_df_metadata
-- WHERE category = 'n/a'
-- ORDER BY ordinal_position;
--
-- Validation: PyMC role coverage
-- SELECT pymc_role, COUNT(*) FROM pml.pml_df_metadata GROUP BY 1 ORDER BY 2 DESC;
--
-- Validation: model_targets coverage by model
-- SELECT m AS model, COUNT(*) AS n_columns
-- FROM pml.pml_df_metadata, UNNEST(model_targets) AS m
-- GROUP BY 1 ORDER BY 2 DESC;

-- =============================================================================
-- CLASSIFICATION COORD WIRING (region/country/.../sector/industry feed every
-- per-model materialized view as coord columns).
-- =============================================================================
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT
                                         unnest(model_targets ||
                                                ARRAY ['earnings_beat', 'price_target', 'kalman_pt', 'dcf_pt', 'dividend_safety', 'credit_risk', 'accounting_anomaly']))
                    )
WHERE column_name IN ('region', 'country', 'trading_country', 'exchange',
                      'unit', 'style_class', 'size_class', 'sector', 'industry');

-- =============================================================================
-- DEFAULT (MODEL-AGNOSTIC) feature_alias FOR COORD COLUMNS
-- =============================================================================
UPDATE pml.pml_df_metadata
SET feature_alias = column_name
WHERE column_name IN ('isin', 'ticker',
                      'region', 'country', 'trading_country', 'exchange',
                      'unit', 'style_class', 'size_class', 'sector', 'industry');

-- =============================================================================
-- PER-MODEL feature_alias OVERRIDES (mirror mv_pymc_* column aliases)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- FK-safety seed: ensure every column_name referenced by the per-model alias
-- INSERT below exists as a parent row in pml.pml_df_metadata. Some columns
-- (notably the derived day-count horizons feat_days_to_*) are computed at
-- import time and are NOT present in pml.pml_df, so the initial
-- information_schema-driven seed would not create rows for them. Without
-- this step the alias INSERT fails with:
--   ERROR:  insert or update on table "pml_df_feature_alias" violates
--           foreign key constraint "pml_df_feature_alias_column_name_fkey"
-- Inserting them here (ON CONFLICT DO NOTHING) is idempotent and harmless
-- for columns that already exist.
-- ---------------------------------------------------------------------------
INSERT INTO pml.pml_df_metadata (column_name, category, feature_role, pymc_role)
SELECT col, 'fiscal_calendar', 'predictor', 'mutable_predictor'
FROM unnest(ARRAY [ 'days_to_next_earnings', 'days_since_last_report', 'days_to_next_fy_end', 'days_to_next_report', 'days_to_expected_report', 'days_to_fy_end' ]) AS col
ON CONFLICT (column_name) DO NOTHING;

-- The derived day-count horizons are created by the FK-safety seed above, which
-- cannot set model_targets. The earlier 7h.2 (price_target) / 7h.3a (kalman_pt)
-- UPDATEs run *before* these rows exist, so they never matched. (Re)assert the
-- model_targets here, after the seed, so the columns surface in
-- vw_pymc_feature_catalogue for BOTH consuming models.
UPDATE pml.pml_df_metadata
SET model_targets = (SELECT ARRAY(SELECT DISTINCT unnest(
		model_targets || ARRAY ['price_target', 'kalman_pt'])))
WHERE column_name IN ('days_to_next_earnings',
                      'days_since_last_report',
                      'days_to_next_fy_end',
                      'days_to_next_report',
                      'days_to_expected_report',
                      'days_to_fy_end');

-- ---------------------------------------------------------------------------
-- TASK 5 / FINDING 4: trim the mutable_predictor surface to what each MV emits.
-- The category-based model_targets wiring (sections 7a / 7d / 7g) is far broader
-- than each MV's curated output, so unaliased raw columns fall back to
-- feature_alias = column_name and are reindexed to all-zero by the catalogue-
-- driven models. We narrow the surface here.
-- ---------------------------------------------------------------------------

-- 5a. EarningsBeat: the neg1..neg5 surprise trails are consumed into the
--     pml.beat_counts() arrays inside mv_pymc_earnings_beat (only the neg0
--     carriers are emitted as feat_*). Demote them from mutable_predictor to
--     derived_input so they leave the predictor surface (~30+ phantom columns).
UPDATE pml.pml_df_metadata
SET pymc_role = 'derived_input'
WHERE feature_role = 'surprise'
  AND column_name SIMILAR TO '%neg[1-9]f[qy]surprise_pct';

-- 5b. EarningsBeat: drop unused eps/ebit/ebitda/sales PREDICTOR levels that the
--     MV never emits (it only consumes the FY1E/FQ1E normalized estimates and
--     the revision %s, which keep their earnings_beat wiring). Revision and
--     neg0-surprise columns are untouched.
UPDATE pml.pml_df_metadata
SET model_targets = array_remove(model_targets, 'earnings_beat')
WHERE 'earnings_beat' = ANY (model_targets)
  AND category IN ('eps', 'ebit', 'ebitda', 'sales')
  AND feature_role = 'predictor'
  AND column_name NOT IN ('eps_norm_est_avg_fy1e', 'eps_norm_est_avg_fq1e');

-- 5c. DCFPriceTarget: keep only the cash_flow / profitability raw columns the
--     MV actually reads; remove dcf_pt from the rest of those two categories.
UPDATE pml.pml_df_metadata
SET model_targets = array_remove(model_targets, 'dcf_pt')
WHERE 'dcf_pt' = ANY (model_targets)
  AND category IN ('cash_flow', 'profitability')
  AND column_name NOT IN ('fcf_ltm', 'fcf_est_avg_fy1e', 'fcf_est_avg_fy2e', 'fcf_est_avg_fy3e',
                          'fcf_est_avg_fy4e', 'fcf_est_avg_fy5e', 'cfo_ltm', 'capital_expenditure_ltm',
                          'return_on_assets_roa_pct_ltm', 'gross_profit_margin_pct_ltm');

-- ---------------------------------------------------------------------------
-- TASK 4: Register engineered (multi-source) MV feature columns directly.
-- These feat_* columns are derived from several raw pml_df columns inside the
-- mv_pymc_* views, so they cannot be expressed via a single per-source alias
-- row (PK (column_name, model_target)). Without their own metadata row they
-- never appear in vw_pymc_feature_catalogue and the catalogue-driven models
-- silently reindex them to 0.0 (Finding 1). We register each as its own
-- column_name = feature_alias = '<feat>' with pymc_role='mutable_predictor'
-- and model_targets pointing at the consuming MV(s).
-- ---------------------------------------------------------------------------
INSERT INTO pml.pml_df_metadata (column_name,                          category,          feature_role, pymc_role,
	                                                                                                                         feature_alias,
	                                                                                                                                                               model_targets                      )
VALUES
	-- kalman_pt drift / range engineered feats (mv_pymc_kalman_pt)
	                            ('feat_implied_upside',                'analyst_targets', 'predictor',  'mutable_predictor', 'feat_implied_upside',                ARRAY ['kalman_pt', 'price_target']),
	                            ('feat_pt_drift',                      'analyst_targets', 'predictor',  'mutable_predictor', 'feat_pt_drift',                      ARRAY ['kalman_pt']                ),
	                            ('feat_price_drift',                   'analyst_targets', 'predictor',  'mutable_predictor', 'feat_price_drift',                   ARRAY ['kalman_pt']                ),
	                            ('feat_pt_high_drift',                 'analyst_targets', 'predictor',  'mutable_predictor', 'feat_pt_high_drift',                 ARRAY ['kalman_pt']                ),
	                            ('feat_pt_low_drift',                  'analyst_targets', 'predictor',  'mutable_predictor', 'feat_pt_low_drift',                  ARRAY ['kalman_pt']                ),
	                            ('feat_pt_median_drift',               'analyst_targets', 'predictor',  'mutable_predictor', 'feat_pt_median_drift',               ARRAY ['kalman_pt']                ),
	                            ('feat_coverage_drift',                'analyst_targets', 'predictor',  'mutable_predictor', 'feat_coverage_drift',                ARRAY ['kalman_pt']                ),
	                            ('feat_pt_noise_drift',                'analyst_targets', 'predictor',  'mutable_predictor', 'feat_pt_noise_drift',                ARRAY ['kalman_pt']                ),
	                            ('feat_pt_range_norm',                 'analyst_targets', 'predictor',  'mutable_predictor', 'feat_pt_range_norm',                 ARRAY ['kalman_pt']                ),
	-- earnings_beat logit-beat-rate + revision-acceleration feats (mv_pymc_earnings_beat)
	                            ('feat_logit_beat_rate',               'eps',             'predictor',  'mutable_predictor', 'feat_logit_beat_rate',               ARRAY ['earnings_beat']            ),
	                            ('feat_logit_beat_rate_annual',        'eps',             'predictor',  'mutable_predictor', 'feat_logit_beat_rate_annual',        ARRAY ['earnings_beat']            ),
	                            ('feat_ebit_logit_beat_rate',          'ebit',            'predictor',  'mutable_predictor', 'feat_ebit_logit_beat_rate',          ARRAY ['earnings_beat']            ),
	                            ('feat_ebit_logit_beat_rate_annual',   'ebit',            'predictor',  'mutable_predictor', 'feat_ebit_logit_beat_rate_annual',   ARRAY ['earnings_beat']            ),
	                            ('feat_ebitda_logit_beat_rate',        'ebitda',          'predictor',  'mutable_predictor', 'feat_ebitda_logit_beat_rate',        ARRAY ['earnings_beat']            ),
	                            ('feat_ebitda_logit_beat_rate_annual', 'ebitda',          'predictor',  'mutable_predictor', 'feat_ebitda_logit_beat_rate_annual', ARRAY ['earnings_beat']            ),
	                            ('feat_sales_logit_beat_rate',         'sales',           'predictor',  'mutable_predictor', 'feat_sales_logit_beat_rate',         ARRAY ['earnings_beat']            ),
	                            ('feat_sales_logit_beat_rate_annual',  'sales',           'predictor',  'mutable_predictor', 'feat_sales_logit_beat_rate_annual',  ARRAY ['earnings_beat']            ),
	                            ('feat_rev_accel_1m_6m',               'eps',             'predictor',  'mutable_predictor', 'feat_rev_accel_1m_6m',               ARRAY ['earnings_beat']            ),
	-- dcf_pt FCF-growth engineered feats (mv_pymc_dcf_pt)
	                            ('feat_fcf_growth_1y',                 'cash_flow',       'predictor',  'mutable_predictor', 'feat_fcf_growth_1y',                 ARRAY ['dcf_pt']                   ),
	                            ('feat_fcf_growth_2y',                 'cash_flow',       'predictor',  'mutable_predictor', 'feat_fcf_growth_2y',                 ARRAY ['dcf_pt']                   ),
	                            ('feat_fcf_terminal_growth',           'cash_flow',       'predictor',  'mutable_predictor', 'feat_fcf_terminal_growth',           ARRAY ['dcf_pt']                   ),
	                            ('feat_reinvest_rate',                 'cash_flow',       'predictor',  'mutable_predictor', 'feat_reinvest_rate',                 ARRAY ['dcf_pt']                   ),
	-- price_target multi-source engineered feats (mv_pymc_price_target). Task 6:
	-- these previously had MISNAMED aliases overloaded on raw carriers
	-- (feat_*_high/_low/_lag/_1y/_6m). Register them as self-rows so the alias
	-- equals the MV column name exactly, and drop the bad carrier aliases below.
	                            ('feat_target_range_width',            'analyst_targets', 'predictor',  'mutable_predictor', 'feat_target_range_width',            ARRAY ['price_target']             ),
	                            ('feat_52w_range_position',            'technical',       'predictor',  'mutable_predictor', 'feat_52w_range_position',            ARRAY ['price_target']             ),
	                            ('feat_pt_range_hit_rate',             'analyst_targets', 'predictor',  'mutable_predictor', 'feat_pt_range_hit_rate',             ARRAY ['price_target']             ),
	                            ('feat_pt_high_low_convergence_1y',    'analyst_targets', 'predictor',  'mutable_predictor', 'feat_pt_high_low_convergence_1y',    ARRAY ['price_target']             ),
	                            ('feat_analyst_count_stability',       'analyst_targets', 'predictor',  'mutable_predictor', 'feat_analyst_count_stability',       ARRAY ['price_target']             ),
	                            ('feat_pt_accuracy_1y',                'analyst_targets', 'predictor',  'mutable_predictor', 'feat_pt_accuracy_1y',                ARRAY ['price_target']             ),
	                            ('feat_pt_optimism_bias',              'analyst_targets', 'predictor',  'mutable_predictor', 'feat_pt_optimism_bias',              ARRAY ['price_target']             ),
	                            ('feat_net_buy_sentiment',             'analyst_ratings', 'predictor',  'mutable_predictor', 'feat_net_buy_sentiment',             ARRAY ['price_target']             ),
	                            ('feat_conviction_ratio',              'analyst_ratings', 'predictor',  'mutable_predictor', 'feat_conviction_ratio',              ARRAY ['price_target']             )
ON CONFLICT (column_name) DO UPDATE SET pymc_role     = excluded.pymc_role,
                                        feature_alias = excluded.feature_alias,
                                        model_targets = (SELECT ARRAY(SELECT DISTINCT
                                                                             unnest(pml.pml_df_metadata.model_targets || excluded.model_targets))
                                        ),
                                        updated_at    = CURRENT_TIMESTAMP;

TRUNCATE pml.pml_df_feature_alias;

INSERT INTO pml.pml_df_feature_alias (column_name,                         model_target,         feature_alias                      )
VALUES
	-- ---- earnings_beat aliases (from mv_pymc_earnings_beat) ----
	                                 ('eps_norm_est_avg_fy1e',             'earnings_beat',      'feat_eps_fy1e'                    ),
	                                 ('eps_norm_est_avg_fq1e',             'earnings_beat',      'feat_eps_fq1e'                    ),
	                                 ('eps_norm_est_num_fy1e',             'earnings_beat',      'n_eps_estimates'                  ),
	                                 ('eps_est_avg_rev_pct_fy1e_1w',       'earnings_beat',      'feat_rev_1w'                      ),
	                                 ('eps_est_avg_rev_pct_fy1e_1m',       'earnings_beat',      'feat_rev_1m'                      ),
	                                 ('eps_est_avg_rev_pct_fy1e_3m',       'earnings_beat',      'feat_rev_3m'                      ),
	                                 ('eps_est_avg_rev_pct_fy1e_6m',       'earnings_beat',      'feat_rev_6m'                      ),
	                                 ('eps_est_avg_rev_pct_fy1e_1y',       'earnings_beat',      'feat_rev_1y'                      ),
	                                 ('eps_gaap_est_avg_rev_pct_fy1e_3m',  'earnings_beat',      'feat_rev_gaap_gap_3m'             ),
	                                 ('eps_neg0fqsurprise_pct',            'earnings_beat',      'feat_last_q_surprise'             ),
	                                 ('eps_neg0fysurprise_pct',            'earnings_beat',      'feat_last_y_surprise'             ),
		-- Most-recent single-period EBIT / EBITDA / Sales surprises. The full
		-- quarterly / annual surprise trails feed pml.beat_counts -> n_*_total /
		-- n_*_beats / feat_*_logit_beat_rate in mv_pymc_earnings_beat; those are
		-- array-derived (multi-column) so, like the EPS n_total / n_beats /
		-- feat_logit_beat_rate columns, they are intentionally NOT aliased here
		-- (the alias table is keyed PRIMARY KEY (column_name, model_target)).
		-- Only the neg0-period carriers below get an alias.
		                                 ('ebit_neg0fqsurprise_pct',           'earnings_beat',      'feat_ebit_last_q_surprise'        ),
		                                 ('ebit_neg0fysurprise_pct',           'earnings_beat',      'feat_ebit_last_y_surprise'        ),
		                                 ('ebitda_neg0fqsurprise_pct',         'earnings_beat',      'feat_ebitda_last_q_surprise'      ),
		                                 ('ebitda_neg0fysurprise_pct',         'earnings_beat',      'feat_ebitda_last_y_surprise'      ),
		                                 ('sales_neg0fqsurprise_pct',          'earnings_beat',      'feat_sales_last_q_surprise'       ),
		                                 ('sales_neg0fysurprise_pct',          'earnings_beat',      'feat_sales_last_y_surprise'       ),
	                                 ('days_to_earnings',                  'earnings_beat',      'feat_days_to_earnings'            ),
	                                 ('earnings_report_recency',           'earnings_beat',      'feat_report_recency'              ),
	                                 ('next_earnings_status',              'earnings_beat',      'feat_next_earnings_status'        ),

	-- ---- price_target aliases (from mv_pymc_price_target) ----
	                                 ('target_pct_avg',                    'price_target',       'observed_target_pct'              ),
	                                 ('target_pct_med',                    'price_target',       'observed_target_pct_med'          ),
	                                 ('last_price',                        'price_target',       'last_price'                       ),
	                                 ('price_target_num',                  'price_target',       'n_analysts'                       ),
	                                 ('num_hold_ratings',                  'price_target',       'feat_holds'                       ),
	                                 ('num_no_opinion_ratings',            'price_target',       'feat_no_opinion'                  ),
	-- ---- price_target normalized analyst-sentiment %s (multi-source carriers) ----
	-- feat_analyst_bullish_pct / bearish_pct / neutral_pct / conviction in
	-- mv_pymc_price_target are each computed from ALL six num_*_ratings columns.
	-- The alias table is keyed PRIMARY KEY (column_name, model_target), so
	-- provenance is recorded on one representative (otherwise-unaliased) rating
	-- column per feature. num_hold_ratings / num_no_opinion_ratings are already
	-- mapped above (feat_holds / feat_no_opinion), leaving these four unaliased
	-- columns as carriers -- no PK collision. (feat_net_buy_sentiment /
	-- feat_conviction_ratio are multi-source and are now registered as their
	-- own self-rows in the TASK 4 metadata INSERT above, so they reach the
	-- catalogue directly rather than via the notebook KNOWN_FEATURES fallback.)
	                                 ('num_strong_buys_ratings',           'price_target',       'feat_analyst_bullish_pct'         ),
	                                 ('num_strong_sell_ratings',           'price_target',       'feat_analyst_bearish_pct'         ),
	                                 ('num_buys_ratings',                  'price_target',       'feat_analyst_neutral_pct'         ),
	                                 ('num_sell_ratings',                  'price_target',       'feat_analyst_conviction'          ),
	-- Task 6: removed ('price_target','price_target','feat_price_target') -- the
	-- MV emits raw `price_target`, not a `feat_price_target` column.
	                                 ('price_target_stddev',               'price_target',       'feat_target_dispersion_cv'        ),
	                                 ('p_e_ntm',                           'price_target',       'feat_pe_ntm'                      ),
	                                 ('ev_ebitda_ntm',                     'price_target',       'feat_ev_ebitda_ntm'               ),
	                                 ('volatility_3m',                     'price_target',       'feat_vol_3m'                      ),
	                                 ('analyst_rating',                    'price_target',       'feat_analyst_rating'              ),
	-- Task 6: feat_52w_range_position (from w_52high_adj + w_52low_adj) and
	-- feat_target_range_width (from target_pct_high + target_pct_low) are
	-- multi-source; registered as self-rows above. The misnamed *_high/_low
	-- carrier aliases the MV never emits were removed here.
	                                 ('price_target_3m_ago',               'price_target',       'feat_pt_momentum_3m'              ),
	                                 ('price_target_num_3m_ago',           'price_target',       'feat_coverage_change_3m'          ),
	-- ---- price_target MvGRW time-axis anchors (raw DATE coords) ----
	                                 ('income_statement_report_date',      'price_target',       'income_statement_report_date'     ),
	                                 ('next_earnings',                     'price_target',       'next_earnings'                    ),
	                                 ('fy_end_date',                       'price_target',       'fy_end_date'                      ),
	                                 ('next_income_statement_report_date', 'price_target',       'next_income_statement_report_date'),
	                                 ('next_fy_end_date',                  'price_target',       'next_fy_end_date'                 ),
	                                 ('expected_report_date',              'price_target',       'expected_report_date'             ),
	-- ---- price_target derived day-count horizons (mutable_predictor) ----
	                                 ('days_to_next_earnings',             'price_target',       'feat_days_to_next_earnings'       ),
	                                 ('days_since_last_report',            'price_target',       'feat_days_since_last_report'      ),
	                                 ('days_to_next_fy_end',               'price_target',       'feat_days_to_next_fy_end'         ),
	                                 ('days_to_next_report',               'price_target',       'feat_days_to_next_report'         ),
	                                 ('days_to_expected_report',           'price_target',       'feat_days_to_expected_report'     ),
	                                 ('days_to_fy_end',                    'price_target',       'feat_days_to_fy_end'              ),
	-- ---- price_target achievement / accuracy (realised vs 1Y-ago targets) ----
	                                 ('price_target_1y_ago',               'price_target',       'feat_pt_achievement_1y'           ),
	-- Task 6: feat_pt_range_hit_rate (low_1y_ago + high_1y_ago),
	-- feat_pt_high_low_convergence_1y (high/low/median + 1y_ago) and
	-- feat_analyst_count_stability (num + 1y/6m/3m_ago) are multi-source;
	-- registered as self-rows above. The misnamed *_low/_high/_lag/_1y/_6m
	-- carrier aliases (which the MV never emits) were removed here.
	                                 ('price_target_median',               'price_target',       'feat_pt_median_vs_mean_spread'    ),

	-- ---- kalman_pt aliases (from mv_pymc_kalman_pt) ----
	                                 ('price_target',                      'kalman_pt',          'observed_pt'                      ),
	                                 ('last_price',                        'kalman_pt',          'last_price'                       ),
	                                 ('price_target_high',                 'kalman_pt',          'price_target_high'                ),
	                                 ('price_target_low',                  'kalman_pt',          'price_target_low'                 ),
	                                 ('price_target_median',               'kalman_pt',          'price_target_median'              ),
	                                 ('price_target_num',                  'kalman_pt',          'n_analysts'                       ),
	                                 ('price_target_stddev',               'kalman_pt',          'feat_pt_noise_sigma'              ),
	                                 ('volatility_1m',                     'kalman_pt',          'feat_vol_1m'                      ),
	                                 ('volatility_3m',                     'kalman_pt',          'feat_vol_3m'                      ),
	                                 ('volatility_6m',                     'kalman_pt',          'feat_vol_6m'                      ),
	                                 ('volatility_1y',                     'kalman_pt',          'feat_vol_1y'                      ),
	                                 ('total_return_ytd',                  'kalman_pt',          'feat_total_return_ytd'            ),
	-- NOTE: mv_pymc_kalman_pt also emits `feat_implied_upside`
	--       (= calc_change_ratio(price_target, last_price)). It derives from BOTH
	--       `price_target` and `last_price`, so it cannot be an alias row here
	--       (PK (column_name, model_target); both sources already map to
	--       observed_pt / last_price). It is instead registered as its own
	--       self-row in the TASK 4 metadata INSERT above (model_targets =
	--       {kalman_pt, price_target}), so it now reaches vw_pymc_feature_catalogue
	--       directly instead of relying only on the notebook KNOWN_FEATURES fallback.
	-- ---- kalman_pt fiscal-calendar time-axis anchors (raw DATE coords) ----
	-- Alias == raw MV column name (the MV emits these un-prefixed), so the
	-- notebook's `feature_alias IN kalman_df.columns` present-check resolves.
	                                 ('income_statement_report_date',      'kalman_pt',          'income_statement_report_date'     ),
	                                 ('next_earnings',                     'kalman_pt',          'next_earnings'                    ),
	                                 ('fy_end_date',                       'kalman_pt',          'fy_end_date'                      ),
	                                 ('next_income_statement_report_date', 'kalman_pt',          'next_income_statement_report_date'),
	                                 ('next_fy_end_date',                  'kalman_pt',          'next_fy_end_date'                 ),
	                                 ('expected_report_date',              'kalman_pt',          'expected_report_date'             ),
	-- ---- kalman_pt derived day-count horizons (mutable_predictor) ----
	                                 ('days_to_next_earnings',             'kalman_pt',          'days_to_next_earnings'            ),
	                                 ('days_since_last_report',            'kalman_pt',          'days_since_last_report'           ),
	                                 ('days_to_next_fy_end',               'kalman_pt',          'days_to_next_fy_end'              ),
	                                 ('days_to_next_report',               'kalman_pt',          'days_to_next_report'              ),
	                                 ('days_to_expected_report',           'kalman_pt',          'days_to_expected_report'          ),
	                                 ('days_to_fy_end',                    'kalman_pt',          'days_to_fy_end'                   ),

	-- ---- dcf_pt aliases (from mv_pymc_dcf_pt) ----
	                                 ('price_target',                      'dcf_pt',             'observed_pt'                      ),
	                                 ('market_cap',                        'dcf_pt',             'market_cap'                       ),
	                                 ('enterprise_value',                  'dcf_pt',             'enterprise_value'                 ),
	                                 ('shrs_out',                          'dcf_pt',             'shrs_out'                         ),
	                                 ('fcf_ltm',                           'dcf_pt',             'feat_fcf_ltm'                     ),
	                                 ('fcf_est_avg_fy1e',                  'dcf_pt',             'feat_fcf_fy1e'                    ),
	                                 ('fcf_est_avg_fy2e',                  'dcf_pt',             'feat_fcf_fy2e'                    ),
	                                 ('fcf_est_avg_fy3e',                  'dcf_pt',             'feat_fcf_fy3e'                    ),
	                                 ('fcf_est_avg_fy4e',                  'dcf_pt',             'feat_fcf_fy4e'                    ),
	                                 ('fcf_est_avg_fy5e',                  'dcf_pt',             'feat_fcf_fy5e'                    ),
	                                 ('cfo_ltm',                           'dcf_pt',             'feat_cfo_ltm'                     ),
	                                 ('capital_expenditure_ltm',           'dcf_pt',             'feat_capex_to_fcf'                ),
	                                 ('tot_return_pct_cagr_3y',            'dcf_pt',             'feat_tr_cagr_3y'                  ),
	                                 ('tot_return_pct_cagr_10y',           'dcf_pt',             'feat_tr_cagr_10y'                 ),
	                                 ('peg_ntm',                           'dcf_pt',             'feat_peg_ntm'                     ),
	                                 ('ev_sales_ltm',                      'dcf_pt',             'feat_ev_sales_ltm'                ),
	                                 ('ev_ebitda_ntm',                     'dcf_pt',             'feat_ev_ebitda_ntm'               ),
	                                 ('return_on_assets_roa_pct_ltm',      'dcf_pt',             'feat_roa_ltm'                     ),
	                                 ('gross_profit_margin_pct_ltm',       'dcf_pt',             'feat_gpm_ltm'                     ),
	                                 ('beta_5y',                           'dcf_pt',             'feat_beta_5y'                     ),

	-- ---- dividend_safety aliases (from mv_pymc_dividend_safety) ----
	                                 ('div_yield_ltm',                     'dividend_safety',    'observed_div_yield'               ),
	                                 ('dividend_streak',                   'dividend_safety',    'n_streak'                         ),
	                                 ('dividend_record_frequency',         'dividend_safety',    'feat_div_frequency'               ),
	                                 ('fcf_ltm',                           'dividend_safety',    'feat_fcf_coverage'                ),
	                                 ('cfo_ltm',                           'dividend_safety',    'feat_cfo_coverage'                ),
	                                 ('common_dividends_paid_ltm',         'dividend_safety',    'feat_fcf_coverage_denom'          ),
	                                 ('dividend_per_share_ltm',            'dividend_safety',    'feat_eps_payout_ratio'            ),
	                                 ('net_eps_basic_ltm',                 'dividend_safety',    'feat_eps_payout_ratio_denom'      ),
	                                 ('dividend_per_share_neg1fy',         'dividend_safety',    'feat_dps_growth_1y'               ),
	                                 ('dividend_per_share_neg3fy',         'dividend_safety',    'feat_dps_growth_3y'               ),
	                                 ('dividend_per_share_neg5fy',         'dividend_safety',    'feat_dps_growth_5y'               ),
	                                 ('buyback_yield_ltm',                 'dividend_safety',    'feat_buyback_yield'               ),
	                                 ('repurchase_common_stock_ltm',       'dividend_safety',    'feat_repurchases_ltm'             ),
	                                 ('altman_z_score_ltm',                'dividend_safety',    'feat_altman_z'                    ),
	                                 ('return_on_assets_roa_pct_ltm',      'dividend_safety',    'feat_roa_ltm'                     ),
	                                 ('div_yield_5yavgltm',                'dividend_safety',    'feat_yield_spread_vs_5y'          ),

	-- ---- credit_risk aliases (from mv_pymc_credit_risk) ----
	                                 ('altman_z_score_ltm',                'credit_risk',        'observed_altman_z'                ),
	                                 ('altman_z_score_neg1fy',             'credit_risk',        'feat_z_trend_1y'                  ),
	                                 ('altman_z_score_neg3fy',             'credit_risk',        'feat_z_trend_3y'                  ),
	                                 ('cfo_ltm',                           'credit_risk',        'feat_cfo_capex_cov'               ),
	                                 ('capital_expenditure_ltm',           'credit_risk',        'feat_cfo_capex_cov_denom'         ),
	                                 ('fcf_ltm',                           'credit_risk',        'feat_fcf_yield'                   ),
	                                 ('enterprise_value',                  'credit_risk',        'feat_fcf_yield_denom'             ),
	                                 ('cff_ltm',                           'credit_risk',        'feat_cff_to_ev'                   ),
	                                 ('issuance_common_stock_ltm',         'credit_risk',        'feat_net_equity_issuance'         ),
	                                 ('repurchase_common_stock_ltm',       'credit_risk',        'feat_net_equity_issuance_offset'  ),
	                                 ('market_cap',                        'credit_risk',        'feat_net_equity_issuance_denom'   ),
	                                 ('full_time_employees_fy',            'credit_risk',        'feat_employee_growth_1y'          ),
	                                 ('full_time_employees_neg1fy',        'credit_risk',        'feat_employee_growth_1y_lag'      ),
	                                 ('p_b_ltm',                           'credit_risk',        'feat_pb_ltm'                      ),
	                                 ('beta_2y',                           'credit_risk',        'feat_beta_2y'                     ),
	                                 ('volatility_6m',                     'credit_risk',        'feat_vol_6m'                      ),
	                                 ('volatility_1y',                     'credit_risk',        'feat_vol_1y'                      ),

	-- ---- accounting_anomaly aliases (from mv_pymc_accounting_anomaly) ----
	                                 ('eps_adj_ltm',                       'accounting_anomaly', 'observed_eps_adj'                 ),
	                                 ('net_eps_basic_ltm',                 'accounting_anomaly', 'feat_accruals_ratio_ni'           ),
	                                 ('cfo_ltm',                           'accounting_anomaly', 'feat_accruals_ratio_cfo'          ),
	                                 ('enterprise_value',                  'accounting_anomaly', 'feat_accruals_ratio_scale'        ),
	                                 ('gross_profit_margin_pct_ltm',       'accounting_anomaly', 'feat_gpm_change_1y'               ),
	                                 ('gross_profit_margin_pct_neg1fy',    'accounting_anomaly', 'feat_gpm_change_1y_lag'           ),
	                                 ('sales_neg0fyactual',                'accounting_anomaly', 'feat_sales_growth_1y'             ),
	                                 ('sales_neg1fyactual',                'accounting_anomaly', 'feat_sales_growth_1y_lag'         ),
	                                 ('ebit_neg0fyactual',                 'accounting_anomaly', 'feat_ebit_growth_1y'              ),
	                                 ('ebit_neg1fyactual',                 'accounting_anomaly', 'feat_ebit_growth_1y_lag'          ),
	                                 ('ebitda_neg0fyactual',               'accounting_anomaly', 'feat_ebitda_growth_1y'            ),
	                                 ('ebitda_neg1fyactual',               'accounting_anomaly', 'feat_ebitda_growth_1y_lag'        ),
	                                 ('capital_expenditure_ltm',           'accounting_anomaly', 'feat_capex_intensity'             ),
	                                 ('cfi_ltm',                           'accounting_anomaly', 'feat_cfi_to_cfo'                  ),
	                                 ('cff_ltm',                           'accounting_anomaly', 'feat_cff_to_cfo'                  ),
	                                 ('shrs_out',                          'accounting_anomaly', 'feat_share_inflation_1y'          ),
	                                 ('shrs_out_neg1fy',                   'accounting_anomaly', 'feat_share_inflation_1y_lag'      ),
	                                 ('issuance_common_stock_ltm',         'accounting_anomaly', 'feat_issuance_intensity'          ),
	                                 ('market_cap',                        'accounting_anomaly', 'feat_issuance_intensity_denom'    ),
	                                 ('full_time_employees_fy',            'accounting_anomaly', 'feat_employee_growth_1y'          ),
	                                 ('full_time_employees_neg1fy',        'accounting_anomaly', 'feat_employee_growth_1y_lag'      ),
	                                 ('fcf_per_share_ltm',                 'accounting_anomaly', 'feat_fcfps_vs_eps_gap'            ),
	                                 ('peg_ntm',                           'accounting_anomaly', 'feat_peg_ntm'                     )
ON CONFLICT (column_name, model_target) DO UPDATE SET feature_alias = excluded.feature_alias;

-- Classification coords share the source column name as alias across every
-- pymc model_target.
INSERT INTO pml.pml_df_feature_alias (column_name, model_target, feature_alias)
SELECT col, m, col
FROM unnest(ARRAY ['isin', 'ticker', 'region', 'country', 'trading_country', 'exchange', 'unit', 'style_class', 'size_class', 'sector', 'industry']) col
	     CROSS JOIN unnest(ARRAY ['earnings_beat', 'price_target', 'kalman_pt', 'dcf_pt', 'dividend_safety', 'credit_risk', 'accounting_anomaly'])   m
ON CONFLICT (column_name, model_target) DO NOTHING;

-- =============================================================================
-- KalmanFilterPriceTarget: lagged analyst-target TRAIL as a multi-horizon
-- state-space observation set (mv_pymc_kalman_pt).
-- =============================================================================
-- Section 7c wired the price_target_%_ago / price_%_ago lags into
-- kalman_pt.model_targets but left their GLOBAL pml_df_metadata.pymc_role at the
-- 'derived_input' default (feature_role='historical'). The Kalman panel, however,
-- *observes* the full analyst-target trail (level / low / high / median) as the
-- latent price-target state sequence, and conditions on the analyst-count trail
-- as fixed per-step scale (constant_data).
--
-- We register per-model alias rows and override pymc_role ON THE ALIAS ROW ONLY
-- (vw_pymc_feature_catalogue resolves COALESCE(fa.pymc_role, md.pymc_role)). The
-- columns' GLOBAL role stays 'derived_input', so models that read these lags as
-- engineered predictors are unaffected (e.g. price_target consumes
-- price_target_1y_ago as the feat_pt_achievement_1y carrier — see TASK 2 below).
-- feature_alias == raw column name because mv_pymc_kalman_pt emits the trail
-- un-prefixed, so the notebook's `feature_alias IN kalman_df.columns` present-
-- check resolves. Mirrors the TASK 2 / FINDING 2 per-model override pattern.
INSERT INTO pml.pml_df_feature_alias (column_name, model_target, feature_alias, pymc_role)
SELECT col, 'kalman_pt', col, 'observed'
FROM unnest(ARRAY [ 'price_target_1w_ago', 'price_target_mtd_ago', 'price_target_1m_ago', 'price_target_qtd_ago', 'price_target_3m_ago', 'price_target_6m_ago', 'price_target_ytd_ago', 'price_target_1y_ago', 'price_target_low_1w_ago', 'price_target_low_mtd_ago', 'price_target_low_1m_ago', 'price_target_low_qtd_ago', 'price_target_low_3m_ago', 'price_target_low_6m_ago', 'price_target_low_ytd_ago', 'price_target_low_1y_ago', 'price_target_high_1w_ago', 'price_target_high_mtd_ago', 'price_target_high_1m_ago', 'price_target_high_qtd_ago', 'price_target_high_3m_ago', 'price_target_high_6m_ago', 'price_target_high_ytd_ago', 'price_target_high_1y_ago', 'price_target_median_1w_ago', 'price_target_median_mtd_ago', 'price_target_median_1m_ago', 'price_target_median_qtd_ago', 'price_target_median_3m_ago', 'price_target_median_6m_ago', 'price_target_median_ytd_ago', 'price_target_median_1y_ago' ]) AS col
ON CONFLICT (column_name, model_target) DO UPDATE SET feature_alias = excluded.feature_alias,
                                                      pymc_role     = excluded.pymc_role;

-- Analyst-count trail: fixed per-step participation the panel conditions on ->
-- constant_data. No 6m horizon: price_target_num_6m_ago feeds feat_coverage_drift
-- only and intentionally stays 'derived_input'.
INSERT INTO pml.pml_df_feature_alias (column_name, model_target, feature_alias, pymc_role)
SELECT col, 'kalman_pt', col, 'constant_data'
FROM unnest(ARRAY [ 'price_target_num_1w_ago', 'price_target_num_mtd_ago', 'price_target_num_1m_ago', 'price_target_num_qtd_ago', 'price_target_num_3m_ago', 'price_target_num_ytd_ago', 'price_target_num_1y_ago' ]) AS col
ON CONFLICT (column_name, model_target) DO UPDATE SET feature_alias = excluded.feature_alias,
                                                      pymc_role     = excluded.pymc_role;

-- Realized-volatility term-structure: the stochastic-volatility anchor.
-- mv_pymc_kalman_pt emits volatility_{1m,3m,6m,1y} AS feat_vol_{1m,3m,6m,1y}, so
-- without a per-model alias the catalogue's feature_alias falls back to the raw
-- column name ('volatility_1m') and the notebook's `feature_alias IN
-- kalman_df.columns` present-check misses the MV column ('feat_vol_1m'). We map
-- the alias to the MV name and keep the global 'mutable_predictor' role: these
-- columns inform the per-time prior mean (shape) of the log-volatility random walk
-- in KalmanFilterPriceTarget.fit(stochastic_volatility=True) — see
-- _build_stochastic_volatility(). Mirrors the alias-row pattern above.
INSERT INTO pml.pml_df_feature_alias (column_name, model_target, feature_alias, pymc_role)
SELECT col, 'kalman_pt', 'feat_vol_' || split_part(col, '_', 2), 'mutable_predictor'
FROM unnest(ARRAY [ 'volatility_1m', 'volatility_3m', 'volatility_6m', 'volatility_1y' ]) AS col
ON CONFLICT (column_name, model_target) DO UPDATE SET feature_alias = excluded.feature_alias,
                                                      pymc_role     = excluded.pymc_role;

-- =============================================================================
-- TASK 2 / FINDING 2: PER-MODEL pymc_role OVERRIDES
-- =============================================================================
-- A carrier column's GLOBAL pml_df_metadata.pymc_role is model-agnostic, but
-- several columns aliased as feat_* (i.e. used as a mutable_predictor in a
-- specific MV) carry a global role of observed / derived_input / constant_data.
-- vw_pymc_feature_catalogue filters mutable_predictor via
-- COALESCE(fa.pymc_role, md.pymc_role), so without an override these aliases
-- never reach the catalogue-driven models. We set the per-model role on the
-- alias row WITHOUT changing the column's global role (e.g. price_target_stddev
-- stays globally 'observed' but is a 'mutable_predictor' for kalman/price_target).
UPDATE pml.pml_df_feature_alias fa
SET pymc_role = 'mutable_predictor'
FROM (VALUES
	      -- price_target predictor carriers (global observed / derived_input / constant_data)
	      ('price_target_stddev', 'price_target'),
	      ('price_target_3m_ago', 'price_target'),
	      ('price_target_num_3m_ago', 'price_target'),
	      ('price_target_1y_ago', 'price_target'),
	      ('price_target_median', 'price_target'),
	      ('num_strong_buys_ratings', 'price_target'),
	      ('num_strong_sell_ratings', 'price_target'),
	      ('num_buys_ratings', 'price_target'),
	      ('num_sell_ratings', 'price_target'),
	      ('num_hold_ratings', 'price_target'),
	      ('num_no_opinion_ratings', 'price_target'),
	      -- kalman_pt predictor carriers (global observed)
	      ('price_target_stddev', 'kalman_pt'),
	      ('total_return_ytd', 'kalman_pt'),
	      -- dcf_pt CAGR carriers (global observed)
	      ('tot_return_pct_cagr_3y', 'dcf_pt'),
	      ('tot_return_pct_cagr_10y', 'dcf_pt'),
	      -- dividend_safety derived/coord carriers
	      ('dividend_per_share_neg1fy', 'dividend_safety'),
	      ('dividend_per_share_neg3fy', 'dividend_safety'),
	      ('dividend_per_share_neg5fy', 'dividend_safety'),
	      ('dividend_record_frequency', 'dividend_safety'),
	      -- credit_risk derived/count carriers
	      ('altman_z_score_neg1fy', 'credit_risk'),
	      ('altman_z_score_neg3fy', 'credit_risk'),
	      ('full_time_employees_fy', 'credit_risk'),
	      -- accounting_anomaly derived carriers
	      ('sales_neg0fyactual', 'accounting_anomaly'),
	      ('ebit_neg0fyactual', 'accounting_anomaly'),
	      ('ebitda_neg0fyactual', 'accounting_anomaly'),
	      ('full_time_employees_fy', 'accounting_anomaly')
     ) AS ov(column_name, model_target)
WHERE fa.column_name = ov.column_name
  AND fa.model_target = ov.model_target;

COMMIT;

-- =============================================================================
-- USEFUL FILTERS (for feature engineering)
-- =============================================================================
-- All ML predictors (numeric inputs ready to feed models):
--   SELECT column_name FROM pml.pml_df_metadata
--   WHERE feature_role IN ('predictor','score','count','surprise','revision');
--
-- All hierarchical categorical effects:
--   SELECT column_name FROM pml.pml_df_metadata
--   WHERE feature_role = 'categorical';
--
-- All model targets / labels:
--   SELECT column_name FROM pml.pml_df_metadata WHERE feature_role = 'target';
--
-- Historical lagged levels (use to derive momentum / drift features):
--   SELECT column_name FROM pml.pml_df_metadata WHERE feature_role = 'historical';
--
-- By data domain (e.g. valuation block only):
--   SELECT column_name FROM pml.pml_df_metadata WHERE category = 'valuation';
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_feature_role ON pml.pml_df_metadata (feature_role);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_category ON pml.pml_df_metadata (category);

CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_pymc_role ON pml.pml_df_metadata (pymc_role);

-- GIN index over model_targets so `WHERE 'earnings_beat' = ANY(model_targets)`
-- (and `model_targets @> ARRAY['dcf_pt']`) become index-backed.
CREATE INDEX IF NOT EXISTS idx_pml_df_metadata_model_targets ON pml.pml_df_metadata USING gin (model_targets);

-- =============================================================================
-- PYMC-ALIGNED VIEWS  (drive the notebook MODEL_FEATURE_CONTAINERS registry)
-- =============================================================================
-- vw_pml_df_pymc_features: one row per (model, column), exposing the pm.Data
-- container kind so the notebook can build, per model:
--     coords        : columns where pymc_role = 'coord'
--     observed      : columns where pymc_role = 'observed'
--     features      : columns where pymc_role = 'mutable_predictor'
--     constant_data : columns where pymc_role = 'constant_data'
-- This is the single SQL source of truth for MODEL_FEATURE_CONTAINERS.
CREATE OR REPLACE VIEW pml.vw_pml_df_pymc_features AS
SELECT m.model_name,
       md.column_name,
       md.category,
       md.feature_role,
       md.pymc_role,
       md.data_type,
       md.ordinal_position,
       md.description
FROM pml.pml_df_metadata         md,
     UNNEST(md.model_targets) AS m(model_name)
WHERE md.pymc_role IN ('coord', 'observed', 'mutable_predictor', 'constant_data');

-- Convenience predicate views (mirror equities_schema_metadata role views).
CREATE OR REPLACE VIEW pml.vw_pml_df_predictors AS
SELECT column_name, category, data_type, ordinal_position, description
FROM pml.pml_df_metadata
WHERE pymc_role = 'mutable_predictor';

CREATE OR REPLACE VIEW pml.vw_pml_df_observed AS
SELECT column_name, category, data_type, ordinal_position, description
FROM pml.pml_df_metadata
WHERE pymc_role = 'observed';

CREATE OR REPLACE VIEW pml.vw_pml_df_coords AS
SELECT column_name, category, data_type, ordinal_position, description
FROM pml.pml_df_metadata
WHERE pymc_role = 'coord';

CREATE OR REPLACE VIEW pml.vw_pml_df_derived_inputs AS
SELECT column_name, category, data_type, ordinal_position, description
FROM pml.pml_df_metadata
WHERE pymc_role = 'derived_input';

COMMENT ON TABLE pml.pml_df_metadata IS 'Metadata for pml.pml_df. (category, feature_role) drive domain/data-centric SQL filters; (pymc_role, model_targets) drive PyMC pm.Data container assignment and per-model feature selection. pymc_role vocabulary: coord | index | observed | mutable_predictor | constant_data | derived_input | excluded. model_targets is a TEXT[] keyed by MODEL_FEATURE_CONTAINERS (earnings_beat, price_target, kalman_pt, dcf_pt, dividend_safety, credit_risk, accounting_anomaly).';

COMMENT ON COLUMN pml.pml_df_metadata.pymc_role IS 'PyMC pm.Data container kind for this column. Aligns with arviz.InferenceData groups: coord/index -> idata.constant_data + posterior.coords; observed -> idata.observed_data; mutable_predictor/constant_data -> idata.constant_data (mutable_predictor supports pm.set_data for OOS); derived_input -> must be transformed before pm.Data; excluded -> never wrapped.';

COMMENT ON COLUMN pml.pml_df_metadata.model_targets IS 'Array of PyMC model names from probabilistic_ml_model.pymc_models that consume this column. Mirrors MODEL_FEATURE_CONTAINERS keys in pymc_expected_returns_model.ipynb.';