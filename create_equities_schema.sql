-- Drop existing table if it exists
DROP TABLE IF EXISTS equities CASCADE;

CREATE TABLE equities
(
    -- ===========================================
    -- ID role
    -- ===========================================
    "Ticker"                                           TEXT,                         -- id: Ticker | alias: ticker
    "ISIN"                                             TEXT,                         -- id: ISIN | alias: isin
    "Name"                                             TEXT,                         -- id: Name | alias: name
    "Description"                                      TEXT,                         -- id: Description | alias: description

    -- ===========================================
    -- CATEGORICAL role
    -- ===========================================
    "Region"                                           TEXT    DEFAULT 'n/a',        -- categorical: Region | alias: region
    "Country"                                          TEXT    DEFAULT 'n/a',        -- categorical: Country | alias: country
    "Trading Country"                                  TEXT    DEFAULT 'n/a',        -- categorical: Trading Country | alias: trading_country
    "Exchange"                                         TEXT    DEFAULT 'n/a',        -- categorical: Exchange | alias: exchange
    "Unit"                                             TEXT    DEFAULT 'n/a',        -- categorical: Unit | alias: unit
    "Sector"                                           TEXT    DEFAULT 'n/a',        -- categorical: Sector | alias: sector
    "Industry"                                         TEXT    DEFAULT 'n/a',        -- categorical: Industry | alias: industry
    "Style Class"                                      TEXT    DEFAULT 'n/a',        -- categorical: Style Class | alias: style_class
    "Size Class"                                       TEXT    DEFAULT 'n/a',        -- categorical: Size Class | alias: size_class
    "FY End"                                           TEXT    DEFAULT 'n/a',        -- categorical: FY End | alias: fy_end
    "Next Earnings (When)"                             TEXT    DEFAULT 'n/a',        -- categorical: Next Earnings (When) | alias: next_earnings_when
    "Next Earnings (Status)"                           TEXT    DEFAULT 'n/a',        -- categorical: Next Earnings (Status) | alias: next_earnings_status
    "Dividend Record (Currency)"                       TEXT    DEFAULT 'n/a',        -- categorical: Dividend Record (Currency) | alias: dividend_record_currency
    "Dividend Record (Frequency)"                      TEXT    DEFAULT 'n/a',        -- categorical: Dividend Record (Frequency) | alias: dividend_record_frequency
    "Current Fiscal Quarter"                           TEXT    DEFAULT 'n/a',        -- categorical: Current fiscal quarter (formatted as Q4 2025) | alias: current_fiscal_quarter
    "Next Fiscal Quarter"                              TEXT    DEFAULT 'n/a',        -- categorical: Next fiscal quarter (formatted as Q4 2025) | alias: next_fiscal_quarter
    "Next Earnings (Report)"                           TEXT    DEFAULT 'n/a',        -- categorical: Next earnings report type (Full Year/Interim) | alias: next_earnings_report
    "Reporting Interval"                               INTEGER,                      -- categorical: Reporting Interval | alias: reporting_interval
    "Earnings Report (Frequency)"                      TEXT    DEFAULT 'n/a',        -- categorical: Earnings Report (Frequency) | alias: earnings_report_frequency

    -- ===========================================
    -- DATE role
    -- ===========================================
    "Last Updated"                                     DATE,                         -- date: Last Updated | alias: last_updated
    "Income Statement Report Date"                     DATE,                         -- date: Income Statement Report Date | alias: income_statement_report_date
    "Next Earnings"                                    DATE,                         -- date: Next Earnings | alias: next_earnings
    "Dividend Record (Announce Date)"                  DATE,                         -- date: Dividend Record (Announce Date) | alias: dividend_record_announce_date
    "Dividend Record (Payable Date)"                   DATE,                         -- date: Dividend Record (Payable Date) | alias: dividend_record_payable_date
    "Dividend Record (Record Date)"                    DATE,                         -- date: Dividend Record (Record Date) | alias: dividend_record_record_date
    "Dividend Record (Ex Date)"                        DATE,                         -- date: Dividend Record (Ex Date) | alias: dividend_record_ex_date
    "Reference Date"                                   DATE    DEFAULT CURRENT_DATE, -- date: Reference Date | alias: reference_date
    "FY End Date"                                      DATE,                         -- date: Fiscal year end date (parsed from FY End text) | alias: fy_end_date
    "Next FY End Date"                                 DATE,                         -- date: Next fiscal year end date | alias: next_fy_end_date
    "Next Income Statement Report Date"                DATE,                         -- date: Next Income Statement Report Date | alias: next_income_statement_report_date

    -- ===========================================
    -- PRIMARY KEY & CONSTRAINTS
    -- ===========================================
    CONSTRAINT pk_equities PRIMARY KEY ("ISIN"),
    CONSTRAINT uq_equities_ticker UNIQUE ("Ticker"),

    -- ===========================================
    -- MARKET role
    -- ===========================================
    "Price Target"                                     NUMERIC,                      -- market_data: Price Target | alias: price_target
    "Price Target - Median"                            NUMERIC,                      -- market_data: Price Target - Median | alias: price_target_median
    "Dividend Record (Amount)"                         NUMERIC DEFAULT 0,            -- market_data: Dividend amount per share | alias: dividend_record_amount
    "Market Cap"                                       NUMERIC,                      -- market_data: Market capitalization | alias: market_cap
    "Enterprise Value"                                 NUMERIC,                      -- market_data: Enterprise value | alias: enterprise_value
    "Last Price"                                       NUMERIC,                      -- market_data: Last Price | alias: last_price
    "Price Target (YTD Ago)"                           NUMERIC,                      -- market_data: Price Target (YTD Ago) | alias: price_target_ytd_ago
    "Price Target - Low"                               NUMERIC,                      -- market_data: Price Target - Low | alias: price_target_low
    "Price Target - High"                              NUMERIC,                      -- market_data: Price Target - High | alias: price_target_high
    "Market Cap (Country R)"                           NUMERIC,                      -- market_data: Market Cap (Country R) | alias: market_cap_country_r
    "Volume (Shrs)"                                    NUMERIC DEFAULT 0,            -- market_data: Trading volume in shares | alias: volume_shrs
    "Dividend Per Share (LTM)"                         NUMERIC DEFAULT 0,            -- market_data: Dividend Per Share (LTM) | alias: dividend_per_share_ltm
    "Price (5D Ago)"                                   NUMERIC,                      -- market_data: Price (5D Ago) | alias: price_5d_ago
    "Price (1W Ago)"                                   NUMERIC,                      -- market_data: Price (1W Ago) | alias: price_1w_ago
    "Price (1M Ago)"                                   NUMERIC,                      -- market_data: Price (1M Ago) | alias: price_1m_ago
    "Price (3M Ago)"                                   NUMERIC,                      -- market_data: Price (3M Ago) | alias: price_3m_ago
    "Price (6M Ago)"                                   NUMERIC,                      -- market_data: Price (6M Ago) | alias: price_6m_ago
    "Price (1Y Ago)"                                   NUMERIC,                      -- market_data: Price (1Y Ago) | alias: price_1y_ago
    "Price (3Y Ago)"                                   NUMERIC,                      -- market_data: Price (3Y Ago) | alias: price_3y_ago
    "Price (5Y Ago)"                                   NUMERIC,                      -- market_data: Price (5Y Ago) | alias: price_5y_ago
    "Price (QTD Ago)"                                  NUMERIC,                      -- market_data: Price (QTD Ago) | alias: price_qtd_ago
    "Rel. Volume"                                      NUMERIC,                      -- market_data: Relative trading volume ratio | alias: rel_volume
    "52W High/Adj"                                     NUMERIC,                      -- market_data: 52W High/Adj | alias: 52w_high_adj
    "52W Low/Adj"                                      NUMERIC,                      -- market_data: 52W Low/Adj | alias: 52w_low_adj
    "EMA (20D)"                                        NUMERIC,                      -- market_data: EMA (20D) | alias: ema_20d
    "EMA (50D)"                                        NUMERIC,                      -- market_data: EMA (50D) | alias: ema_50d
    "EMA (100D)"                                       NUMERIC,                      -- market_data: EMA (100D) | alias: ema_100d
    "EMA (250D)"                                       NUMERIC,                      -- market_data: EMA (250D) | alias: ema_250d
    "Price Target (1W Ago)"                            NUMERIC,                      -- market_data: Price Target (1W Ago) | alias: price_target_1w_ago
    "Price Target (1M Ago)"                            NUMERIC,                      -- market_data: Price Target (1M Ago) | alias: price_target_1m_ago
    "Price Target (3M Ago)"                            NUMERIC,                      -- market_data: Price Target (3M Ago) | alias: price_target_3m_ago
    "Price Target (6M Ago)"                            NUMERIC,                      -- market_data: Price Target (6M Ago) | alias: price_target_6m_ago
    "Price Target (MTD Ago)"                           NUMERIC,                      -- market_data: Price Target (MTD Ago) | alias: price_target_mtd_ago
    "Price Target (QTD Ago)"                           NUMERIC,                      -- market_data: Price Target (QTD Ago) | alias: price_target_qtd_ago
    "Price Target (1Y Ago)"                            NUMERIC,                      -- market_data: Price Target (1Y Ago) | alias: price_target_1y_ago
    "Price Target - High (1W Ago)"                     NUMERIC,                      -- market_data: Price Target - High (1W Ago) | alias: price_target_high_1w_ago
    "Price Target - High (1M Ago)"                     NUMERIC,                      -- market_data: Price Target - High (1M Ago) | alias: price_target_high_1m_ago
    "Price Target - High (6M Ago)"                     NUMERIC,                      -- market_data: Price Target - High (6M Ago) | alias: price_target_high_6m_ago
    "Price Target - High (MTD Ago)"                    NUMERIC,                      -- market_data: Price Target - High (MTD Ago) | alias: price_target_high_mtd_ago
    "Price Target - High (3M Ago)"                     NUMERIC,                      -- market_data: Price Target - High (3M Ago) | alias: price_target_high_3m_ago
    "Price Target - High (QTD Ago)"                    NUMERIC,                      -- market_data: Price Target - High (QTD Ago) | alias: price_target_high_qtd_ago
    "Price Target - High (1Y Ago)"                     NUMERIC,                      -- market_data: Price Target - High (1Y Ago) | alias: price_target_high_1y_ago
    "Price Target - High (YTD Ago)"                    NUMERIC,                      -- market_data: Price Target - High (YTD Ago) | alias: price_target_high_ytd_ago
    "Price Target - Low (1W Ago)"                      NUMERIC,                      -- market_data: Price Target - Low (1W Ago) | alias: price_target_low_1w_ago
    "Price Target - Low (1M Ago)"                      NUMERIC,                      -- market_data: Price Target - Low (1M Ago) | alias: price_target_low_1m_ago
    "Price Target - Low (3M Ago)"                      NUMERIC,                      -- market_data: Price Target - Low (3M Ago) | alias: price_target_low_3m_ago
    "Price Target - Low (6M Ago)"                      NUMERIC,                      -- market_data: Price Target - Low (6M Ago) | alias: price_target_low_6m_ago
    "Price Target - Low (MTD Ago)"                     NUMERIC,                      -- market_data: Price Target - Low (MTD Ago) | alias: price_target_low_mtd_ago
    "Price Target - Low (QTD Ago)"                     NUMERIC,                      -- market_data: Price Target - Low (QTD Ago) | alias: price_target_low_qtd_ago
    "Price Target - Low (YTD Ago)"                     NUMERIC,                      -- market_data: Price Target - Low (YTD Ago) | alias: price_target_low_ytd_ago
    "Price Target - Low (1Y Ago)"                      NUMERIC,                      -- market_data: Price Target - Low (1Y Ago) | alias: price_target_low_1y_ago
    "Price Target - Median (1W Ago)"                   NUMERIC,                      -- market_data: Price Target - Median (1W Ago) | alias: price_target_median_1w_ago
    "Price Target - Median (1M Ago)"                   NUMERIC,                      -- market_data: Price Target - Median (1M Ago) | alias: price_target_median_1m_ago
    "Price Target - Median (3M Ago)"                   NUMERIC,                      -- market_data: Price Target - Median (3M Ago) | alias: price_target_median_3m_ago
    "Price Target - Median (6M Ago)"                   NUMERIC,                      -- market_data: Price Target - Median (6M Ago) | alias: price_target_median_6m_ago
    "Price Target - Median (MTD Ago)"                  NUMERIC,                      -- market_data: Price Target - Median (MTD Ago) | alias: price_target_median_mtd_ago
    "Price Target - Median (QTD Ago)"                  NUMERIC,                      -- market_data: Price Target - Median (QTD Ago) | alias: price_target_median_qtd_ago
    "Price Target - Median (YTD Ago)"                  NUMERIC,                      -- market_data: Price Target - Median (YTD Ago) | alias: price_target_median_ytd_ago
    "Price Target - Median (1Y Ago)"                   NUMERIC,                      -- market_data: Price Target - Median (1Y Ago) | alias: price_target_median_1y_ago

    -- ===========================================
    -- FINANCIAL STATEMENT role
    -- ===========================================
    "Total Revenues (FQ)"                              NUMERIC DEFAULT 0,            -- income_statement: Total Revenues (FQ) | alias: total_revenues_fq
    "Total Revenues (-1FY)"                            NUMERIC DEFAULT 0,            -- income_statement: Total Revenues (-1FY) | alias: total_revenues_1fy
    "Total Revenues (FY)"                              NUMERIC DEFAULT 0,            -- income_statement: Total revenues (Fiscal Year) | alias: total_revenues_fy
    "Total Revenues (LTM)"                             NUMERIC DEFAULT 0,            -- income_statement: Total revenues (Last Twelve Months) | alias: total_revenues_ltm
    "Net Income/Adj. (-1FY)"                           NUMERIC DEFAULT 0,            -- income_statement: Net Income/Adj. (-1FY) | alias: net_income_adj_1fy
    "EBITDA (FQ)"                                      NUMERIC DEFAULT 0,            -- income_statement: EBITDA (FQ) | alias: ebitda_fq
    "EBITDA (LTM)"                                     NUMERIC DEFAULT 0,            -- income_statement: EBITDA (Last Twelve Months) | alias: ebitda_ltm
    "EBITDA (FY)"                                      NUMERIC DEFAULT 0,            -- income_statement: EBITDA (Fiscal Year) | alias: ebitda_fy
    "EBITDA (-1FY)"                                    NUMERIC DEFAULT 0,            -- income_statement: EBITDA (-1FY) | alias: ebitda_1fy
    "EBITDA/Adj. (LTM)"                                NUMERIC DEFAULT 0,            -- income_statement: EBITDA/Adj. (LTM) | alias: ebitda_adj_ltm
    "EBITDA/Adj. (FY)"                                 NUMERIC DEFAULT 0,            -- income_statement: EBITDA/Adj. (FY) | alias: ebitda_adj_fy
    "EBITDA/Adj. (-1FY)"                               NUMERIC DEFAULT 0,            -- income_statement: EBITDA/Adj. (-1FY) | alias: ebitda_adj_1fy
    "EBIT (FQ)"                                        NUMERIC DEFAULT 0,            -- income_statement: EBIT (FQ) | alias: ebit_fq
    "EBIT (LTM)"                                       NUMERIC DEFAULT 0,            -- income_statement: EBIT (LTM) | alias: ebit_ltm
    "EBIT (FY)"                                        NUMERIC DEFAULT 0,            -- income_statement: EBIT (FY) | alias: ebit_fy
    "EBIT (-1FY)"                                      NUMERIC DEFAULT 0,            -- income_statement: EBIT (-1FY) | alias: ebit_1fy
    "EBIT/Adj. (-1FY)"                                 NUMERIC DEFAULT 0,            -- income_statement: EBIT/Adj. (-1FY) | alias: ebit_adj_1fy
    "EBIT/Adj. (FY)"                                   NUMERIC DEFAULT 0,            -- income_statement: EBIT/Adj. (FY) | alias: ebit_adj_fy
    "EBIT/Adj. (LTM)"                                  NUMERIC DEFAULT 0,            -- income_statement: EBIT/Adj. (LTM) | alias: ebit_adj_ltm
    "EBIT - Est Med (FY1E)"                            NUMERIC DEFAULT 0,            -- income_statement: EBIT - Est Med (FY1E) | alias: ebit_est_med_fy1e
    "EBIT - Est Med (NTM)"                             NUMERIC DEFAULT 0,            -- income_statement: EBIT - Est Med (NTM) | alias: ebit_est_med_ntm
    "Net Income - (IS) (FY)"                           NUMERIC DEFAULT 0,            -- income_statement: Net income from income statement (Fiscal Year) | alias: net_income_is_fy
    "Net Income - (IS) (LTM)"                          NUMERIC DEFAULT 0,            -- income_statement: Net income from income statement (Last Twelve Months) | alias: net_income_is_ltm
    "Normalized Net Income (FY)"                       NUMERIC DEFAULT 0,            -- income_statement: Normalized Net Income (FY) | alias: normalized_net_income_fy
    "Normalized Net Income (LTM)"                      NUMERIC DEFAULT 0,            -- income_statement: Normalized Net Income (LTM) | alias: normalized_net_income_ltm
    "Net Income/Adj. (FY)"                             NUMERIC DEFAULT 0,            -- income_statement: Net Income/Adj. (FY) | alias: net_income_adj_fy
    "Net Income/Adj. (LTM)"                            NUMERIC DEFAULT 0,            -- income_statement: Net Income/Adj. (LTM) | alias: net_income_adj_ltm
    "Gross Profit (LTM)"                               NUMERIC DEFAULT 0,            -- income_statement: Gross Profit (LTM) | alias: gross_profit_ltm
    "Gross Profit (FY)"                                NUMERIC DEFAULT 0,            -- income_statement: Gross Profit (FY) | alias: gross_profit_fy
    "Cost Of Revenues (LTM)"                           NUMERIC DEFAULT 0,            -- income_statement: Cost Of Revenues (LTM) | alias: cost_of_revenues_ltm
    "Operating Income (LTM)"                           NUMERIC DEFAULT 0,            -- income_statement: Operating Income (LTM) | alias: operating_income_ltm
    "Operating Income (FY)"                            NUMERIC DEFAULT 0,            -- income_statement: Operating Income (FY) | alias: operating_income_fy
    "R&D Expenses (LTM)"                               NUMERIC DEFAULT 0,            -- income_statement: R&D Expenses (LTM) | alias: randd_expenses_ltm
    "Interest Expense/Total (LTM)"                     NUMERIC DEFAULT 0,            -- income_statement: Interest Expense/Total (LTM) | alias: interest_expense_total_ltm
    "Interest Income On Investments (LTM)"             NUMERIC DEFAULT 0,            -- income_statement: Interest Income On Investments (LTM) | alias: interest_income_on_investments_ltm
    "Net Income - (IS) (-1FY)"                         NUMERIC DEFAULT 0,            -- income_statement: Net Income - (IS) (-1FY) | alias: net_income_is_1fy
    "Normalized Net Income (-1FY)"                     NUMERIC DEFAULT 0,            -- income_statement: Normalized Net Income (-1FY) | alias: normalized_net_income_1fy
    "Total Revenues (5YAVGFQ)"                         NUMERIC DEFAULT 0,            -- income_statement: Total Revenues (5YAVGFQ) | alias: total_revenues_5yavgfq
    "EBITDA (5YAVGFQ)"                                 NUMERIC DEFAULT 0,            -- income_statement: EBITDA (5YAVGFQ) | alias: ebitda_5yavgfq
    "EBIT (5YAVGFQ)"                                   NUMERIC DEFAULT 0,            -- income_statement: EBIT (5YAVGFQ) | alias: ebit_5yavgfq
    "Operating Income (FQ)"                            NUMERIC DEFAULT 0,            -- income_statement: Operating Income (FQ) | alias: operating_income_fq
    "Operating Income (5YAVGFQ)"                       NUMERIC DEFAULT 0,            -- income_statement: Operating Income (5YAVGFQ) | alias: operating_income_5yavgfq
    "Normalized Net Income (FQ)"                       NUMERIC DEFAULT 0,            -- income_statement: Normalized Net Income (FQ) | alias: normalized_net_income_fq
    "Normalized Net Income (5YAVGFQ)"                  NUMERIC DEFAULT 0,            -- income_statement: Normalized Net Income (5YAVGFQ) | alias: normalized_net_income_5yavgfq
    "Net Income/Adj. (FQ)"                             NUMERIC DEFAULT 0,            -- income_statement: Net Income/Adj. (FQ) | alias: net_income_adj_fq
    "Net Income/Adj. (5YAVGFQ)"                        NUMERIC DEFAULT 0,            -- income_statement: Net Income/Adj. (5YAVGFQ) | alias: net_income_adj_5yavgfq
    "Net Income - (IS) (FQ)"                           NUMERIC DEFAULT 0,            -- income_statement: Net Income - (IS) (FQ) | alias: net_income_is_fq
    "Net Income - (IS) (5YAVGFQ)"                      NUMERIC DEFAULT 0,            -- income_statement: Net Income - (IS) (5YAVGFQ) | alias: net_income_is_5yavgfq
    "Net Income - (IS) (5YAVGLTM)"                     NUMERIC DEFAULT 0,            -- income_statement: Net Income - (IS) (5YAVGLTM) | alias: net_income_is_5yavgltm
    "Normalized Net Income (5YAVGLTM)"                 NUMERIC DEFAULT 0,            -- income_statement: Normalized Net Income (5YAVGLTM) | alias: normalized_net_income_5yavgltm
    "EBITDA (5YAVGLTM)"                                NUMERIC DEFAULT 0,            -- income_statement: EBITDA (5YAVGLTM) | alias: ebitda_5yavgltm
    "EBIT (5YAVGLTM)"                                  NUMERIC DEFAULT 0,            -- income_statement: EBIT (5YAVGLTM) | alias: ebit_5yavgltm
    "Total Revenues (5YAVGLTM)"                        NUMERIC DEFAULT 0,            -- income_statement: Total Revenues (5YAVGLTM) | alias: total_revenues_5yavgltm
    "Selling General & Admin Expenses/Total (FQ)"      NUMERIC DEFAULT 0,            -- income_statement: Selling General & Admin Expenses/Total (FQ) | alias: selling_general_and_admin_expenses_total_fq
    "Selling General & Admin Expenses/Total (FY)"      NUMERIC DEFAULT 0,            -- income_statement: Selling General & Admin Expenses/Total (FY) | alias: selling_general_and_admin_expenses_total_fy
    "Selling General & Admin Expenses/Total (-1FY)"    NUMERIC DEFAULT 0,            -- income_statement: Selling General & Admin Expenses/Total (-1FY) | alias: selling_general_and_admin_expenses_total_1fy
    "Selling General & Admin Expenses/Total (5YAVGFQ)" NUMERIC DEFAULT 0,            -- income_statement: Selling General & Admin Expenses/Total (5YAVGFQ) | alias: selling_general_and_admin_expenses_total_5yavgfq
    "Marketing Expenses (FQ)"                          NUMERIC DEFAULT 0,            -- income_statement: Marketing Expenses (FQ) | alias: marketing_expenses_fq
    "Marketing Expenses (FY)"                          NUMERIC DEFAULT 0,            -- income_statement: Marketing Expenses (FY) | alias: marketing_expenses_fy
    "Marketing Expenses (-1FY)"                        NUMERIC DEFAULT 0,            -- income_statement: Marketing Expenses (-1FY) | alias: marketing_expenses_1fy
    "Marketing Expenses (5YAVGLTM)"                    NUMERIC DEFAULT 0,            -- income_statement: Marketing Expenses (5YAVGLTM) | alias: marketing_expenses_5yavgltm
    "Revenues - Est Avg (NTM)"                         NUMERIC DEFAULT 0,            -- income_statement: Revenues - Est Avg (NTM) | alias: revenues_est_avg_ntm
    "Revenues - Est Avg (FY1E)"                        NUMERIC DEFAULT 0,            -- income_statement: Revenues - Est Avg (FY1E) | alias: revenues_est_avg_fy1e
    "Revenues - Est Med (NTM)"                         NUMERIC DEFAULT 0,            -- income_statement: Revenues - Est Med (NTM) | alias: revenues_est_med_ntm
    "Revenues - Est Med (FY1E)"                        NUMERIC DEFAULT 0,            -- income_statement: Revenues - Est Med (FY1E) | alias: revenues_est_med_fy1e
    "EBITDA - Est Avg (NTM)"                           NUMERIC DEFAULT 0,            -- income_statement: EBITDA - Est Avg (NTM) | alias: ebitda_est_avg_ntm
    "EBITDA - Est Avg (FY1E)"                          NUMERIC DEFAULT 0,            -- income_statement: EBITDA - Est Avg (FY1E) | alias: ebitda_est_avg_fy1e
    "Total Revenues (-1FQFQ)"                          NUMERIC,                      -- income_statement: Total Revenues (-1FQFQ) | alias: total_revenues_1fqfq
    "Total Revenues (-2FQFQ)"                          NUMERIC,                      -- income_statement: Total Revenues (-2FQFQ) | alias: total_revenues_2fqfq
    "Total Revenues (-3FQFQ)"                          NUMERIC,                      -- income_statement: Total Revenues (-3FQFQ) | alias: total_revenues_3fqfq
    "Total Revenues (-4FQFQ)"                          NUMERIC,                      -- income_statement: Total Revenues (-4FQFQ) | alias: total_revenues_4fqfq
    "Total Revenues (-2FY)"                            NUMERIC,                      -- income_statement: Total Revenues (-2FY) | alias: total_revenues_2fy
    "Total Revenues (-3FY)"                            NUMERIC,                      -- income_statement: Total Revenues (-3FY) | alias: total_revenues_3fy
    "Total Revenues (-4FY)"                            NUMERIC,                      -- income_statement: Total Revenues (-4FY) | alias: total_revenues_4fy
    "Gross Profit (FQ)"                                NUMERIC DEFAULT 0,            -- income_statement: Gross Profit (FQ) | alias: gross_profit_fq
    "Gross Profit (-1FQFQ)"                            NUMERIC DEFAULT 0,            -- income_statement: Gross Profit (-1FQFQ) | alias: gross_profit_1fqfq
    "Gross Profit (-2FQFQ)"                            NUMERIC DEFAULT 0,            -- income_statement: Gross Profit (-2FQFQ) | alias: gross_profit_2fqfq
    "Gross Profit (-3FQFQ)"                            NUMERIC DEFAULT 0,            -- income_statement: Gross Profit (-3FQFQ) | alias: gross_profit_3fqfq
    "Gross Profit (-4FQFQ)"                            NUMERIC DEFAULT 0,            -- income_statement: Gross Profit (-4FQFQ) | alias: gross_profit_4fqfq
    "Gross Profit (-1FY)"                              NUMERIC DEFAULT 0,            -- income_statement: Gross Profit (-1FY) | alias: gross_profit_1fy
    "Gross Profit (-2FY)"                              NUMERIC DEFAULT 0,            -- income_statement: Gross Profit (-2FY) | alias: gross_profit_2fy
    "Gross Profit (-3FY)"                              NUMERIC DEFAULT 0,            -- income_statement: Gross Profit (-3FY) | alias: gross_profit_3fy
    "Gross Profit (-4FY)"                              NUMERIC DEFAULT 0,            -- income_statement: Gross Profit (-4FY) | alias: gross_profit_4fy
    "Operating Income (-1FQFQ)"                        NUMERIC DEFAULT 0,            -- income_statement: Operating Income (-1FQFQ) | alias: operating_income_1fqfq
    "Operating Income (-2FQFQ)"                        NUMERIC DEFAULT 0,            -- income_statement: Operating Income (-2FQFQ) | alias: operating_income_2fqfq
    "Operating Income (-3FQFQ)"                        NUMERIC DEFAULT 0,            -- income_statement: Operating Income (-3FQFQ) | alias: operating_income_3fqfq
    "Operating Income (-4FQFQ)"                        NUMERIC DEFAULT 0,            -- income_statement: Operating Income (-4FQFQ) | alias: operating_income_4fqfq
    "Operating Income (-1FY)"                          NUMERIC DEFAULT 0,            -- income_statement: Operating Income (-1FY) | alias: operating_income_1fy
    "Operating Income (-2FY)"                          NUMERIC DEFAULT 0,            -- income_statement: Operating Income (-2FY) | alias: operating_income_2fy
    "Operating Income (-3FY)"                          NUMERIC DEFAULT 0,            -- income_statement: Operating Income (-3FY) | alias: operating_income_3fy
    "Operating Income (-4FY)"                          NUMERIC DEFAULT 0,            -- income_statement: Operating Income (-4FY) | alias: operating_income_4fy
    "R&D Expenses (FQ)"                                NUMERIC DEFAULT 0,            -- income_statement: R&D Expenses (FQ) | alias: randd_expenses_fq
    "R&D Expenses (FY)"                                NUMERIC DEFAULT 0,            -- income_statement: R&D Expenses (FY) | alias: randd_expenses_fy
    "R&D Expenses (-1FQFQ)"                            NUMERIC DEFAULT 0,            -- income_statement: R&D Expenses (-1FQFQ) | alias: randd_expenses_1fqfq
    "R&D Expenses (-2FQFQ)"                            NUMERIC DEFAULT 0,            -- income_statement: R&D Expenses (-2FQFQ) | alias: randd_expenses_2fqfq
    "R&D Expenses (-3FQFQ)"                            NUMERIC DEFAULT 0,            -- income_statement: R&D Expenses (-3FQFQ) | alias: randd_expenses_3fqfq
    "R&D Expenses (-4FQFQ)"                            NUMERIC DEFAULT 0,            -- income_statement: R&D Expenses (-4FQFQ) | alias: randd_expenses_4fqfq
    "R&D Expenses (-1FY)"                              NUMERIC DEFAULT 0,            -- income_statement: R&D Expenses (-1FY) | alias: randd_expenses_1fy
    "R&D Expenses (-2FY)"                              NUMERIC DEFAULT 0,            -- income_statement: R&D Expenses (-2FY) | alias: randd_expenses_2fy
    "R&D Expenses (-3FY)"                              NUMERIC DEFAULT 0,            -- income_statement: R&D Expenses (-3FY) | alias: randd_expenses_3fy
    "R&D Expenses (-4FY)"                              NUMERIC DEFAULT 0,            -- income_statement: R&D Expenses (-4FY) | alias: randd_expenses_4fy
    "Net Income - (IS) (-1FQFQ)"                       NUMERIC DEFAULT 0,            -- income_statement: Net Income - (IS) (-1FQFQ) | alias: net_income_is_1fqfq
    "Net Income - (IS) (-2FQFQ)"                       NUMERIC DEFAULT 0,            -- income_statement: Net Income - (IS) (-2FQFQ) | alias: net_income_is_2fqfq
    "Net Income - (IS) (-3FQFQ)"                       NUMERIC DEFAULT 0,            -- income_statement: Net Income - (IS) (-3FQFQ) | alias: net_income_is_3fqfq
    "Net Income - (IS) (-4FQFQ)"                       NUMERIC DEFAULT 0,            -- income_statement: Net Income - (IS) (-4FQFQ) | alias: net_income_is_4fqfq
    "Net Income - (IS) (-2FY)"                         NUMERIC DEFAULT 0,            -- income_statement: Net Income - (IS) (-2FY) | alias: net_income_is_2fy
    "Net Income - (IS) (-3FY)"                         NUMERIC DEFAULT 0,            -- income_statement: Net Income - (IS) (-3FY) | alias: net_income_is_3fy
    "Net Income - (IS) (-4FY)"                         NUMERIC DEFAULT 0,            -- income_statement: Net Income - (IS) (-4FY) | alias: net_income_is_4fy
    "Normalized Net Income (-1FQFQ)"                   NUMERIC DEFAULT 0,            -- income_statement: Normalized Net Income (-1FQFQ) | alias: normalized_net_income_1fqfq
    "Normalized Net Income (-2FQFQ)"                   NUMERIC DEFAULT 0,            -- income_statement: Normalized Net Income (-2FQFQ) | alias: normalized_net_income_2fqfq
    "Normalized Net Income (-3FQFQ)"                   NUMERIC DEFAULT 0,            -- income_statement: Normalized Net Income (-3FQFQ) | alias: normalized_net_income_3fqfq
    "Normalized Net Income (-4FQFQ)"                   NUMERIC DEFAULT 0,            -- income_statement: Normalized Net Income (-4FQFQ) | alias: normalized_net_income_4fqfq
    "Normalized Net Income (-2FY)"                     NUMERIC DEFAULT 0,            -- income_statement: Normalized Net Income (-2FY) | alias: normalized_net_income_2fy
    "Normalized Net Income (-3FY)"                     NUMERIC DEFAULT 0,            -- income_statement: Normalized Net Income (-3FY) | alias: normalized_net_income_3fy
    "Normalized Net Income (-4FY)"                     NUMERIC DEFAULT 0,            -- income_statement: Normalized Net Income (-4FY) | alias: normalized_net_income_4fy
    "Net Income/Adj. (-1FQFQ)"                         NUMERIC DEFAULT 0,            -- income_statement: Net Income/Adj. (-1FQFQ) | alias: net_income_adj_1fqfq
    "Net Income/Adj. (-2FQFQ)"                         NUMERIC DEFAULT 0,            -- income_statement: Net Income/Adj. (-2FQFQ) | alias: net_income_adj_2fqfq
    "Net Income/Adj. (-3FQFQ)"                         NUMERIC DEFAULT 0,            -- income_statement: Net Income/Adj. (-3FQFQ) | alias: net_income_adj_3fqfq
    "Net Income/Adj. (-4FQFQ)"                         NUMERIC DEFAULT 0,            -- income_statement: Net Income/Adj. (-4FQFQ) | alias: net_income_adj_4fqfq
    "Net Income/Adj. (-2FY)"                           NUMERIC DEFAULT 0,            -- income_statement: Net Income/Adj. (-2FY) | alias: net_income_adj_2fy
    "Net Income/Adj. (-3FY)"                           NUMERIC DEFAULT 0,            -- income_statement: Net Income/Adj. (-3FY) | alias: net_income_adj_3fy
    "Net Income/Adj. (-4FY)"                           NUMERIC DEFAULT 0,            -- income_statement: Net Income/Adj. (-4FY) | alias: net_income_adj_4fy
    "EBIT (-1FQFQ)"                                    NUMERIC DEFAULT 0,            -- income_statement: EBIT (-1FQFQ) | alias: ebit_1fqfq
    "EBIT (-2FQFQ)"                                    NUMERIC DEFAULT 0,            -- income_statement: EBIT (-2FQFQ) | alias: ebit_2fqfq
    "EBIT (-3FQFQ)"                                    NUMERIC DEFAULT 0,            -- income_statement: EBIT (-3FQFQ) | alias: ebit_3fqfq
    "EBIT (-4FQFQ)"                                    NUMERIC DEFAULT 0,            -- income_statement: EBIT (-4FQFQ) | alias: ebit_4fqfq
    "EBIT (-2FY)"                                      NUMERIC DEFAULT 0,            -- income_statement: EBIT (-2FY) | alias: ebit_2fy
    "EBIT (-3FY)"                                      NUMERIC DEFAULT 0,            -- income_statement: EBIT (-3FY) | alias: ebit_3fy
    "EBIT (-4FY)"                                      NUMERIC DEFAULT 0,            -- income_statement: EBIT (-4FY) | alias: ebit_4fy
    "EBIT/Adj. (FQ)"                                   NUMERIC DEFAULT 0,            -- income_statement: EBIT/Adj. (FQ) | alias: ebit_adj_fq
    "EBIT/Adj. (-1FQFQ)"                               NUMERIC DEFAULT 0,            -- income_statement: EBIT/Adj. (-1FQFQ) | alias: ebit_adj_1fqfq
    "EBIT/Adj. (-2FQFQ)"                               NUMERIC DEFAULT 0,            -- income_statement: EBIT/Adj. (-2FQFQ) | alias: ebit_adj_2fqfq
    "EBIT/Adj. (-3FQFQ)"                               NUMERIC DEFAULT 0,            -- income_statement: EBIT/Adj. (-3FQFQ) | alias: ebit_adj_3fqfq
    "EBIT/Adj. (-4FQFQ)"                               NUMERIC DEFAULT 0,            -- income_statement: EBIT/Adj. (-4FQFQ) | alias: ebit_adj_4fqfq
    "EBIT/Adj. (-2FY)"                                 NUMERIC DEFAULT 0,            -- income_statement: EBIT/Adj. (-2FY) | alias: ebit_adj_2fy
    "EBIT/Adj. (-3FY)"                                 NUMERIC DEFAULT 0,            -- income_statement: EBIT/Adj. (-3FY) | alias: ebit_adj_3fy
    "EBIT/Adj. (-4FY)"                                 NUMERIC DEFAULT 0,            -- income_statement: EBIT/Adj. (-4FY) | alias: ebit_adj_4fy
    "EBITDA (-1FQFQ)"                                  NUMERIC DEFAULT 0,            -- income_statement: EBITDA (-1FQFQ) | alias: ebitda_1fqfq
    "EBITDA (-2FQFQ)"                                  NUMERIC DEFAULT 0,            -- income_statement: EBITDA (-2FQFQ) | alias: ebitda_2fqfq
    "EBITDA (-3FQFQ)"                                  NUMERIC DEFAULT 0,            -- income_statement: EBITDA (-3FQFQ) | alias: ebitda_3fqfq
    "EBITDA (-4FQFQ)"                                  NUMERIC DEFAULT 0,            -- income_statement: EBITDA (-4FQFQ) | alias: ebitda_4fqfq
    "EBITDA (-2FY)"                                    NUMERIC DEFAULT 0,            -- income_statement: EBITDA (-2FY) | alias: ebitda_2fy
    "EBITDA (-3FY)"                                    NUMERIC DEFAULT 0,            -- income_statement: EBITDA (-3FY) | alias: ebitda_3fy
    "EBITDA (-4FY)"                                    NUMERIC DEFAULT 0,            -- income_statement: EBITDA (-4FY) | alias: ebitda_4fy
    "EBITDA/Adj. (FQ)"                                 NUMERIC DEFAULT 0,            -- income_statement: EBITDA/Adj. (FQ) | alias: ebitda_adj_fq
    "EBITDA/Adj. (-1FQFQ)"                             NUMERIC DEFAULT 0,            -- income_statement: EBITDA/Adj. (-1FQFQ) | alias: ebitda_adj_1fqfq
    "EBITDA/Adj. (-2FQFQ)"                             NUMERIC DEFAULT 0,            -- income_statement: EBITDA/Adj. (-2FQFQ) | alias: ebitda_adj_2fqfq
    "EBITDA/Adj. (-3FQFQ)"                             NUMERIC DEFAULT 0,            -- income_statement: EBITDA/Adj. (-3FQFQ) | alias: ebitda_adj_3fqfq
    "EBITDA/Adj. (-4FQFQ)"                             NUMERIC DEFAULT 0,            -- income_statement: EBITDA/Adj. (-4FQFQ) | alias: ebitda_adj_4fqfq
    "EBITDA/Adj. (-2FY)"                               NUMERIC DEFAULT 0,            -- income_statement: EBITDA/Adj. (-2FY) | alias: ebitda_adj_2fy
    "EBITDA/Adj. (-3FY)"                               NUMERIC DEFAULT 0,            -- income_statement: EBITDA/Adj. (-3FY) | alias: ebitda_adj_3fy
    "EBITDA/Adj. (-4FY)"                               NUMERIC DEFAULT 0,            -- income_statement: EBITDA/Adj. (-4FY) | alias: ebitda_adj_4fy

    -- ===========================================
    -- BALANCE SHEET role
    -- ===========================================
    "TBV (FY)"                                         NUMERIC DEFAULT 0,            -- balance_sheet: TBV (FY) | alias: tbv_fy
    "TBV (LTM)"                                        NUMERIC DEFAULT 0,            -- balance_sheet: TBV (LTM) | alias: tbv_ltm
    "Total Debt (FY)"                                  NUMERIC DEFAULT 0,            -- balance_sheet: Total debt (Fiscal Year) | alias: total_debt_fy
    "Total Equity (FY)"                                NUMERIC DEFAULT 0,            -- balance_sheet: Total equity (Fiscal Year) | alias: total_equity_fy
    "Total Equity (LTM)"                               NUMERIC DEFAULT 0,            -- balance_sheet: Total Equity (LTM) | alias: total_equity_ltm
    "Total Debt (LTM)"                                 NUMERIC DEFAULT 0,            -- balance_sheet: Total Debt (LTM) | alias: total_debt_ltm
    "Total Assets (LTM)"                               NUMERIC DEFAULT 0,            -- balance_sheet: Total assets (Last Twelve Months) | alias: total_assets_ltm
    "Total Assets (FY)"                                NUMERIC DEFAULT 0,            -- balance_sheet: Total assets (Fiscal Year) | alias: total_assets_fy
    "Inventory (LTM)"                                  NUMERIC DEFAULT 0,            -- balance_sheet: Inventory (LTM) | alias: inventory_ltm
    "Goodwill (FQ)"                                    NUMERIC DEFAULT 0,            -- balance_sheet: Goodwill (FQ) | alias: goodwill_fq
    "Goodwill (LTM)"                                   NUMERIC DEFAULT 0,            -- balance_sheet: Goodwill (LTM) | alias: goodwill_ltm
    "Goodwill (FY)"                                    NUMERIC DEFAULT 0,            -- balance_sheet: Goodwill (FY) | alias: goodwill_fy
    "Goodwill (-1FY)"                                  NUMERIC DEFAULT 0,            -- balance_sheet: Goodwill (-1FY) | alias: goodwill_1fy
    "Retained Earnings (LTM)"                          NUMERIC DEFAULT 0,            -- balance_sheet: Retained Earnings (LTM) | alias: retained_earnings_ltm
    "Total Current Assets (LTM)"                       NUMERIC DEFAULT 0,            -- balance_sheet: Total Current Assets (LTM) | alias: total_current_assets_ltm
    "Total Current Liabilities (LTM)"                  NUMERIC DEFAULT 0,            -- balance_sheet: Total Current Liabilities (LTM) | alias: total_current_liabilities_ltm
    "Working Capital (LTM)"                            NUMERIC DEFAULT 0,            -- balance_sheet: Working Capital (LTM) | alias: working_capital_ltm
    "Cash And Equivalents (LTM)"                       NUMERIC DEFAULT 0,            -- balance_sheet: Cash And Equivalents (LTM) | alias: cash_and_equivalents_ltm
    "Cash And Equivalents (FQ)"                        NUMERIC DEFAULT 0,            -- balance_sheet: Cash And Equivalents (FQ) | alias: cash_and_equivalents_fq
    "Cash And Equivalents (FY)"                        NUMERIC DEFAULT 0,            -- balance_sheet: Cash And Equivalents (FY) | alias: cash_and_equivalents_fy
    "Cash And Equivalents (5YAVGFQ)"                   NUMERIC DEFAULT 0,            -- balance_sheet: Cash And Equivalents (5YAVGFQ) | alias: cash_and_equivalents_5yavgfq
    "Inventory (FQ)"                                   NUMERIC DEFAULT 0,            -- balance_sheet: Inventory (FQ) | alias: inventory_fq
    "Inventory (FY)"                                   NUMERIC DEFAULT 0,            -- balance_sheet: Inventory (FY) | alias: inventory_fy
    "Goodwill (5YAVGFQ)"                               NUMERIC DEFAULT 0,            -- balance_sheet: Goodwill (5YAVGFQ) | alias: goodwill_5yavgfq
    "Inventory (5YAVGFQ)"                              NUMERIC DEFAULT 0,            -- balance_sheet: Inventory (5YAVGFQ) | alias: inventory_5yavgfq
    "Retained Earnings (FQ)"                           NUMERIC DEFAULT 0,            -- balance_sheet: Retained Earnings (FQ) | alias: retained_earnings_fq
    "Retained Earnings (FY)"                           NUMERIC DEFAULT 0,            -- balance_sheet: Retained Earnings (FY) | alias: retained_earnings_fy
    "Retained Earnings (5YAVGFQ)"                      NUMERIC DEFAULT 0,            -- balance_sheet: Retained Earnings (5YAVGFQ) | alias: retained_earnings_5yavgfq
    "Working Capital (FQ)"                             NUMERIC DEFAULT 0,            -- balance_sheet: Working Capital (FQ) | alias: working_capital_fq
    "Working Capital (FY)"                             NUMERIC DEFAULT 0,            -- balance_sheet: Working Capital (FY) | alias: working_capital_fy
    "Working Capital (5YAVGFY)"                        NUMERIC DEFAULT 0,            -- balance_sheet: Working Capital (5YAVGFY) | alias: working_capital_5yavgfy
    "Gross Intangible Assets (LTM)"                    NUMERIC DEFAULT 0,            -- balance_sheet: Gross Intangible Assets (LTM) | alias: gross_intangible_assets_ltm
    "Gross Intangible Assets (FY)"                     NUMERIC DEFAULT 0,            -- balance_sheet: Gross Intangible Assets (FY) | alias: gross_intangible_assets_fy
    "Gross Intangible Assets (5YAVGFQ)"                NUMERIC DEFAULT 0,            -- balance_sheet: Gross Intangible Assets (5YAVGFQ) | alias: gross_intangible_assets_5yavgfq
    "Accounts Receivable/Total (FY)"                   NUMERIC DEFAULT 0,            -- balance_sheet: Accounts Receivable/Total (FY) | alias: accounts_receivable_total_fy
    "Accounts Receivable/Total (-1FY)"                 NUMERIC DEFAULT 0,            -- balance_sheet: Accounts Receivable/Total (-1FY) | alias: accounts_receivable_total_1fy
    "Accounts Receivable/Total (5YAVGFQ)"              NUMERIC DEFAULT 0,            -- balance_sheet: Accounts Receivable/Total (5YAVGFQ) | alias: accounts_receivable_total_5yavgfq
    "Working Capital (-1FQ)"                           NUMERIC DEFAULT 0,            -- balance_sheet: Working Capital (-1FQ) | alias: working_capital_1fq
    "Working Capital (-2FQ)"                           NUMERIC DEFAULT 0,            -- balance_sheet: Working Capital (-2FQ) | alias: working_capital_2fq
    "Working Capital (-3FQ)"                           NUMERIC DEFAULT 0,            -- balance_sheet: Working Capital (-3FQ) | alias: working_capital_3fq
    "Working Capital (-4FQ)"                           NUMERIC DEFAULT 0,            -- balance_sheet: Working Capital (-4FQ) | alias: working_capital_4fq
    "Working Capital (-1FY)"                           NUMERIC DEFAULT 0,            -- balance_sheet: Working Capital (-1FY) | alias: working_capital_1fy
    "Working Capital (-2FY)"                           NUMERIC DEFAULT 0,            -- balance_sheet: Working Capital (-2FY) | alias: working_capital_2fy
    "Working Capital (-3FY)"                           NUMERIC DEFAULT 0,            -- balance_sheet: Working Capital (-3FY) | alias: working_capital_3fy
    "Working Capital (-4FY)"                           NUMERIC DEFAULT 0,            -- balance_sheet: Working Capital (-4FY) | alias: working_capital_4fy
    "Total Debt (FQ)"                                  NUMERIC DEFAULT 0,            -- balance_sheet: Total Debt (FQ) | alias: total_debt_fq
    "Total Debt (-1FQ)"                                NUMERIC DEFAULT 0,            -- balance_sheet: Total Debt (-1FQ) | alias: total_debt_1fq
    "Total Debt (-2FQ)"                                NUMERIC DEFAULT 0,            -- balance_sheet: Total Debt (-2FQ) | alias: total_debt_2fq
    "Total Debt (-3FQ)"                                NUMERIC DEFAULT 0,            -- balance_sheet: Total Debt (-3FQ) | alias: total_debt_3fq
    "Total Debt (-4FQ)"                                NUMERIC DEFAULT 0,            -- balance_sheet: Total Debt (-4FQ) | alias: total_debt_4fq
    "Total Debt (-1FY)"                                NUMERIC DEFAULT 0,            -- balance_sheet: Total Debt (-1FY) | alias: total_debt_1fy
    "Total Debt (-2FY)"                                NUMERIC DEFAULT 0,            -- balance_sheet: Total Debt (-2FY) | alias: total_debt_2fy
    "Total Debt (-3FY)"                                NUMERIC DEFAULT 0,            -- balance_sheet: Total Debt (-3FY) | alias: total_debt_3fy
    "Total Debt (-4FY)"                                NUMERIC DEFAULT 0,            -- balance_sheet: Total Debt (-4FY) | alias: total_debt_4fy
    "Total Assets (FQ)"                                NUMERIC DEFAULT 0,            -- balance_sheet: Total Assets (FQ) | alias: total_assets_fq
    "Total Assets (-1FQ)"                              NUMERIC DEFAULT 0,            -- balance_sheet: Total Assets (-1FQ) | alias: total_assets_1fq
    "Total Assets (-2FQ)"                              NUMERIC DEFAULT 0,            -- balance_sheet: Total Assets (-2FQ) | alias: total_assets_2fq
    "Total Assets (-3FQ)"                              NUMERIC DEFAULT 0,            -- balance_sheet: Total Assets (-3FQ) | alias: total_assets_3fq
    "Total Assets (-4FQ)"                              NUMERIC DEFAULT 0,            -- balance_sheet: Total Assets (-4FQ) | alias: total_assets_4fq
    "Total Assets (-1FY)"                              NUMERIC DEFAULT 0,            -- balance_sheet: Total Assets (-1FY) | alias: total_assets_1fy
    "Total Assets (-2FY)"                              NUMERIC DEFAULT 0,            -- balance_sheet: Total Assets (-2FY) | alias: total_assets_2fy
    "Total Assets (-3FY)"                              NUMERIC DEFAULT 0,            -- balance_sheet: Total Assets (-3FY) | alias: total_assets_3fy
    "Total Assets (-4FY)"                              NUMERIC DEFAULT 0,            -- balance_sheet: Total Assets (-4FY) | alias: total_assets_4fy
    "Inventory (-1FQ)"                                 NUMERIC DEFAULT 0,            -- balance_sheet: Inventory (-1FQ) | alias: inventory_1fq
    "Inventory (-2FQ)"                                 NUMERIC DEFAULT 0,            -- balance_sheet: Inventory (-2FQ) | alias: inventory_2fq
    "Inventory (-3FQ)"                                 NUMERIC DEFAULT 0,            -- balance_sheet: Inventory (-3FQ) | alias: inventory_3fq
    "Inventory (-4FQ)"                                 NUMERIC DEFAULT 0,            -- balance_sheet: Inventory (-4FQ) | alias: inventory_4fq
    "Inventory (-1FY)"                                 NUMERIC DEFAULT 0,            -- balance_sheet: Inventory (-1FY) | alias: inventory_1fy
    "Inventory (-2FY)"                                 NUMERIC DEFAULT 0,            -- balance_sheet: Inventory (-2FY) | alias: inventory_2fy
    "Inventory (-3FY)"                                 NUMERIC DEFAULT 0,            -- balance_sheet: Inventory (-3FY) | alias: inventory_3fy
    "Inventory (-4FY)"                                 NUMERIC DEFAULT 0,            -- balance_sheet: Inventory (-4FY) | alias: inventory_4fy
    "Goodwill (-1FQ)"                                  NUMERIC DEFAULT 0,            -- balance_sheet: Goodwill (-1FQ) | alias: goodwill_1fq
    "Goodwill (-2FQ)"                                  NUMERIC DEFAULT 0,            -- balance_sheet: Goodwill (-2FQ) | alias: goodwill_2fq
    "Goodwill (-3FQ)"                                  NUMERIC DEFAULT 0,            -- balance_sheet: Goodwill (-3FQ) | alias: goodwill_3fq
    "Goodwill (-4FQ)"                                  NUMERIC DEFAULT 0,            -- balance_sheet: Goodwill (-4FQ) | alias: goodwill_4fq
    "Goodwill (-2FY)"                                  NUMERIC DEFAULT 0,            -- balance_sheet: Goodwill (-2FY) | alias: goodwill_2fy
    "Goodwill (-3FY)"                                  NUMERIC DEFAULT 0,            -- balance_sheet: Goodwill (-3FY) | alias: goodwill_3fy
    "Goodwill (-4FY)"                                  NUMERIC DEFAULT 0,            -- balance_sheet: Goodwill (-4FY) | alias: goodwill_4fy
    "Retained Earnings (-1FQ)"                         NUMERIC DEFAULT 0,            -- balance_sheet: Retained Earnings (-1FQ) | alias: retained_earnings_1fq
    "Retained Earnings (-2FQ)"                         NUMERIC DEFAULT 0,            -- balance_sheet: Retained Earnings (-2FQ) | alias: retained_earnings_2fq
    "Retained Earnings (-3FQ)"                         NUMERIC DEFAULT 0,            -- balance_sheet: Retained Earnings (-3FQ) | alias: retained_earnings_3fq
    "Retained Earnings (-4FQ)"                         NUMERIC DEFAULT 0,            -- balance_sheet: Retained Earnings (-4FQ) | alias: retained_earnings_4fq
    "Retained Earnings (-1FY)"                         NUMERIC DEFAULT 0,            -- balance_sheet: Retained Earnings (-1FY) | alias: retained_earnings_1fy
    "Retained Earnings (-2FY)"                         NUMERIC DEFAULT 0,            -- balance_sheet: Retained Earnings (-2FY) | alias: retained_earnings_2fy
    "Retained Earnings (-3FY)"                         NUMERIC DEFAULT 0,            -- balance_sheet: Retained Earnings (-3FY) | alias: retained_earnings_3fy
    "Retained Earnings (-4FY)"                         NUMERIC DEFAULT 0,            -- balance_sheet: Retained Earnings (-4FY) | alias: retained_earnings_4fy
    "Cash And Equivalents (-1FQ)"                      NUMERIC DEFAULT 0,            -- balance_sheet: Cash And Equivalents (-1FQ) | alias: cash_and_equivalents_1fq
    "Cash And Equivalents (-2FQ)"                      NUMERIC DEFAULT 0,            -- balance_sheet: Cash And Equivalents (-2FQ) | alias: cash_and_equivalents_2fq
    "Cash And Equivalents (-3FQ)"                      NUMERIC DEFAULT 0,            -- balance_sheet: Cash And Equivalents (-3FQ) | alias: cash_and_equivalents_3fq
    "Cash And Equivalents (-4FQ)"                      NUMERIC DEFAULT 0,            -- balance_sheet: Cash And Equivalents (-4FQ) | alias: cash_and_equivalents_4fq
    "Cash And Equivalents (-1FY)"                      NUMERIC DEFAULT 0,            -- balance_sheet: Cash And Equivalents (-1FY) | alias: cash_and_equivalents_1fy
    "Cash And Equivalents (-2FY)"                      NUMERIC DEFAULT 0,            -- balance_sheet: Cash And Equivalents (-2FY) | alias: cash_and_equivalents_2fy
    "Cash And Equivalents (-3FY)"                      NUMERIC DEFAULT 0,            -- balance_sheet: Cash And Equivalents (-3FY) | alias: cash_and_equivalents_3fy
    "Cash And Equivalents (-4FY)"                      NUMERIC DEFAULT 0,            -- balance_sheet: Cash And Equivalents (-4FY) | alias: cash_and_equivalents_4fy
    "Gross Intangible Assets (FQ)"                     NUMERIC DEFAULT 0,            -- balance_sheet: Gross Intangible Assets (FQ) | alias: gross_intangible_assets_fq
    "Gross Intangible Assets (-1FQ)"                   NUMERIC DEFAULT 0,            -- balance_sheet: Gross Intangible Assets (-1FQ) | alias: gross_intangible_assets_1fq
    "Gross Intangible Assets (-2FQ)"                   NUMERIC DEFAULT 0,            -- balance_sheet: Gross Intangible Assets (-2FQ) | alias: gross_intangible_assets_2fq
    "Gross Intangible Assets (-3FQ)"                   NUMERIC DEFAULT 0,            -- balance_sheet: Gross Intangible Assets (-3FQ) | alias: gross_intangible_assets_3fq
    "Gross Intangible Assets (-4FQ)"                   NUMERIC DEFAULT 0,            -- balance_sheet: Gross Intangible Assets (-4FQ) | alias: gross_intangible_assets_4fq
    "Gross Intangible Assets (-1FY)"                   NUMERIC DEFAULT 0,            -- balance_sheet: Gross Intangible Assets (-1FY) | alias: gross_intangible_assets_1fy
    "Gross Intangible Assets (-2FY)"                   NUMERIC DEFAULT 0,            -- balance_sheet: Gross Intangible Assets (-2FY) | alias: gross_intangible_assets_2fy
    "Gross Intangible Assets (-3FY)"                   NUMERIC DEFAULT 0,            -- balance_sheet: Gross Intangible Assets (-3FY) | alias: gross_intangible_assets_3fy
    "Gross Intangible Assets (-4FY)"                   NUMERIC DEFAULT 0,            -- balance_sheet: Gross Intangible Assets (-4FY) | alias: gross_intangible_assets_4fy

    -- ===========================================
    -- CASH FLOW role
    -- ===========================================
    "CFF (LTM)"                                        NUMERIC DEFAULT 0,            -- cash_flow: CFF (LTM) | alias: cff_ltm
    "CFI (LTM)"                                        NUMERIC DEFAULT 0,            -- cash_flow: CFI (LTM) | alias: cfi_ltm
    "FCF (LTM)"                                        NUMERIC DEFAULT 0,            -- cash_flow: Free cash flow (Last Twelve Months) | alias: fcf_ltm
    "CFO (LTM)"                                        NUMERIC DEFAULT 0,            -- cash_flow: Cash from operations (Last Twelve Months) | alias: cfo_ltm
    "Cash Acquisitions (LTM)"                          NUMERIC DEFAULT 0,            -- cash_flow: Cash Acquisitions (LTM) | alias: cash_acquisitions_ltm
    "Cash Acquisitions (FY)"                           NUMERIC DEFAULT 0,            -- cash_flow: Cash Acquisitions (FY) | alias: cash_acquisitions_fy
    "Cash Acquisitions (-1FY)"                         NUMERIC DEFAULT 0,            -- cash_flow: Cash Acquisitions (-1FY) | alias: cash_acquisitions_1fy
    "Capital Expenditure (LTM)"                        NUMERIC DEFAULT 0,            -- cash_flow: Capital expenditure (Last Twelve Months) | alias: capital_expenditure_ltm
    "Capital Expenditure (-1FY)"                       NUMERIC DEFAULT 0,            -- cash_flow: Capital Expenditure (-1FY) | alias: capital_expenditure_1fy
    "Capital Expenditure (FY)"                         NUMERIC DEFAULT 0,            -- cash_flow: Capital Expenditure (FY) | alias: capital_expenditure_fy
    "CFF (FY)"                                         NUMERIC DEFAULT 0,            -- cash_flow: CFF (FY) | alias: cff_fy
    "CFF (-1FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: CFF (-1FY) | alias: cff_1fy
    "CFI (FY)"                                         NUMERIC DEFAULT 0,            -- cash_flow: CFI (FY) | alias: cfi_fy
    "CFI (-1FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: CFI (-1FY) | alias: cfi_1fy
    "CFO (FY)"                                         NUMERIC DEFAULT 0,            -- cash_flow: CFO (FY) | alias: cfo_fy
    "CFO (-1FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: CFO (-1FY) | alias: cfo_1fy
    "FCF (FY)"                                         NUMERIC DEFAULT 0,            -- cash_flow: FCF (FY) | alias: fcf_fy
    "FCF (-1FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: FCF (-1FY) | alias: fcf_1fy
    "Capital Expenditure (FQ)"                         NUMERIC DEFAULT 0,            -- cash_flow: Capital Expenditure (FQ) | alias: capital_expenditure_fq
    "Capital Expenditure (5YAVGFQ)"                    NUMERIC DEFAULT 0,            -- cash_flow: Capital Expenditure (5YAVGFQ) | alias: capital_expenditure_5yavgfq
    "CFF (FQ)"                                         NUMERIC DEFAULT 0,            -- cash_flow: CFF (FQ) | alias: cff_fq
    "CFI (FQ)"                                         NUMERIC DEFAULT 0,            -- cash_flow: CFI (FQ) | alias: cfi_fq
    "CFO (FQ)"                                         NUMERIC DEFAULT 0,            -- cash_flow: CFO (FQ) | alias: cfo_fq
    "FCF (FQ)"                                         NUMERIC DEFAULT 0,            -- cash_flow: FCF (FQ) | alias: fcf_fq
    "FCF (5YAVGFQ)"                                    NUMERIC DEFAULT 0,            -- cash_flow: FCF (5YAVGFQ) | alias: fcf_5yavgfq
    "Cash Acquisitions (FQ)"                           NUMERIC DEFAULT 0,            -- cash_flow: Cash Acquisitions (FQ) | alias: cash_acquisitions_fq
    "Cash Acquisitions (5YAVGFQ)"                      NUMERIC DEFAULT 0,            -- cash_flow: Cash Acquisitions (5YAVGFQ) | alias: cash_acquisitions_5yavgfq
    "Common Dividends Paid (LTM)"                      NUMERIC DEFAULT 0,            -- cash_flow: Common Dividends Paid (LTM) | alias: common_dividends_paid_ltm
    "Common Dividends Paid (FY)"                       NUMERIC DEFAULT 0,            -- cash_flow: Common Dividends Paid (FY) | alias: common_dividends_paid_fy
    "CFO (-1FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: CFO (-1FQFQ) | alias: cfo_1fqfq
    "CFO (-2FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: CFO (-2FQFQ) | alias: cfo_2fqfq
    "CFO (-3FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: CFO (-3FQFQ) | alias: cfo_3fqfq
    "CFO (-4FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: CFO (-4FQFQ) | alias: cfo_4fqfq
    "CFI (-1FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: CFI (-1FQFQ) | alias: cfi_1fqfq
    "CFI (-2FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: CFI (-2FQFQ) | alias: cfi_2fqfq
    "CFI (-3FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: CFI (-3FQFQ) | alias: cfi_3fqfq
    "CFI (-4FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: CFI (-4FQFQ) | alias: cfi_4fqfq
    "CFI (-2FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: CFI (-2FY) | alias: cfi_2fy
    "CFI (-3FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: CFI (-3FY) | alias: cfi_3fy
    "CFI (-4FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: CFI (-4FY) | alias: cfi_4fy
    "FCF (-1FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: FCF (-1FQFQ) | alias: fcf_1fqfq
    "FCF (-2FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: FCF (-2FQFQ) | alias: fcf_2fqfq
    "FCF (-3FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: FCF (-3FQFQ) | alias: fcf_3fqfq
    "FCF (-4FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: FCF (-4FQFQ) | alias: fcf_4fqfq
    "CFF (-2FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: CFF (-2FY) | alias: cff_2fy
    "CFF (-3FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: CFF (-3FY) | alias: cff_3fy
    "CFF (-4FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: CFF (-4FY) | alias: cff_4fy
    "CFF (-1FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: CFF (-1FQFQ) | alias: cff_1fqfq
    "CFF (-2FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: CFF (-2FQFQ) | alias: cff_2fqfq
    "CFF (-3FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: CFF (-3FQFQ) | alias: cff_3fqfq
    "CFF (-4FQFQ)"                                     NUMERIC DEFAULT 0,            -- cash_flow: CFF (-4FQFQ) | alias: cff_4fqfq
    "CFO (-2FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: CFO (-2FY) | alias: cfo_2fy
    "CFO (-3FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: CFO (-3FY) | alias: cfo_3fy
    "CFO (-4FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: CFO (-4FY) | alias: cfo_4fy
    "Cash Acquisitions (-1FQFQ)"                       NUMERIC DEFAULT 0,            -- cash_flow: Cash Acquisitions (-1FQFQ) | alias: cash_acquisitions_1fqfq
    "Cash Acquisitions (-2FQFQ)"                       NUMERIC DEFAULT 0,            -- cash_flow: Cash Acquisitions (-2FQFQ) | alias: cash_acquisitions_2fqfq
    "Cash Acquisitions (-3FQFQ)"                       NUMERIC DEFAULT 0,            -- cash_flow: Cash Acquisitions (-3FQFQ) | alias: cash_acquisitions_3fqfq
    "Cash Acquisitions (-4FQFQ)"                       NUMERIC DEFAULT 0,            -- cash_flow: Cash Acquisitions (-4FQFQ) | alias: cash_acquisitions_4fqfq
    "FCF (-2FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: FCF (-2FY) | alias: fcf_2fy
    "FCF (-3FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: FCF (-3FY) | alias: fcf_3fy
    "FCF (-4FY)"                                       NUMERIC DEFAULT 0,            -- cash_flow: FCF (-4FY) | alias: fcf_4fy
    "Cash Acquisitions (-2FY)"                         NUMERIC DEFAULT 0,            -- cash_flow: Cash Acquisitions (-2FY) | alias: cash_acquisitions_2fy
    "Cash Acquisitions (-3FY)"                         NUMERIC DEFAULT 0,            -- cash_flow: Cash Acquisitions (-3FY) | alias: cash_acquisitions_3fy
    "Cash Acquisitions (-4FY)"                         NUMERIC DEFAULT 0,            -- cash_flow: Cash Acquisitions (-4FY) | alias: cash_acquisitions_4fy
    "Capital Expenditure (-1FQFQ)"                     NUMERIC DEFAULT 0,            -- cash_flow: Capital Expenditure (-1FQFQ) | alias: capital_expenditure_1fqfq
    "Capital Expenditure (-2FQFQ)"                     NUMERIC DEFAULT 0,            -- cash_flow: Capital Expenditure (-2FQFQ) | alias: capital_expenditure_2fqfq
    "Capital Expenditure (-3FQFQ)"                     NUMERIC DEFAULT 0,            -- cash_flow: Capital Expenditure (-3FQFQ) | alias: capital_expenditure_3fqfq
    "Capital Expenditure (-4FQFQ)"                     NUMERIC DEFAULT 0,            -- cash_flow: Capital Expenditure (-4FQFQ) | alias: capital_expenditure_4fqfq
    "Capital Expenditure (-2FY)"                       NUMERIC DEFAULT 0,            -- cash_flow: Capital Expenditure (-2FY) | alias: capital_expenditure_2fy
    "Capital Expenditure (-3FY)"                       NUMERIC DEFAULT 0,            -- cash_flow: Capital Expenditure (-3FY) | alias: capital_expenditure_3fy
    "Capital Expenditure (-4FY)"                       NUMERIC DEFAULT 0,            -- cash_flow: Capital Expenditure (-4FY) | alias: capital_expenditure_4fy

    -- ===========================================
    -- RATIO role
    -- ===========================================
    "P/E (NTM)"                                        NUMERIC,                      -- ratio: P/E (NTM) | alias: p_e_ntm
    "P/E (LTM)"                                        NUMERIC,                      -- ratio: P/E (LTM) | alias: p_e_ltm
    "Altman Z-Score (FY)"                              NUMERIC,                      -- ratio: Altman Z-Score (FY) | alias: altman_z_score_fy
    "Altman Z-Score (FQ)"                              NUMERIC,                      -- ratio: Altman Z-Score (FQ) | alias: altman_z_score_fq
    "Altman Z-Score (LTM)"                             NUMERIC,                      -- ratio: Altman Z-Score (LTM) | alias: altman_z_score_ltm
    "P/TBV (LTM)"                                      NUMERIC,                      -- ratio: P/TBV (LTM) | alias: p_tbv_ltm
    "Return On Equity % (LTM)"                         NUMERIC,                      -- ratio: Return on equity percentage (Last Twelve Months) | alias: return_on_equity_pct_ltm
    "Return On Equity % (FY)"                          NUMERIC,                      -- ratio: Return On Equity % (FY) | alias: return_on_equity_pct_fy
    "Current Ratio (FY)"                               NUMERIC,                      -- ratio: Current Ratio (FY) | alias: current_ratio_fy
    "Current Ratio (LTM)"                              NUMERIC,                      -- ratio: Current Ratio (LTM) | alias: current_ratio_ltm
    "Asset Turnover (FY)"                              NUMERIC,                      -- ratio: Asset Turnover (FY) | alias: asset_turnover_fy
    "Asset Turnover (LTM)"                             NUMERIC,                      -- ratio: Asset Turnover (LTM) | alias: asset_turnover_ltm
    "EPS Norm - Est Avg (NTM)"                         NUMERIC,                      -- ratio: EPS Norm - Est Avg (NTM) | alias: eps_norm_est_avg_ntm
    "EPS/Adj. (-1FY)"                                  NUMERIC,                      -- ratio: EPS/Adj. (-1FY) | alias: eps_adj_1fy
    "EPS/Adj. (FY)"                                    NUMERIC,                      -- ratio: EPS/Adj. (FY) | alias: eps_adj_fy
    "EPS/Adj. (LTM)"                                   NUMERIC,                      -- ratio: EPS/Adj. (LTM) | alias: eps_adj_ltm
    "EPS Norm - Est Avg (FY1E)"                        NUMERIC,                      -- ratio: EPS Norm - Est Avg (FY1E) | alias: eps_norm_est_avg_fy1e
    "Return on Assets (ROA) % (LTM)"                   NUMERIC,                      -- ratio: Return on assets percentage (Last Twelve Months) | alias: return_on_assets_roa_pct_ltm
    "Return on Assets (ROA) % (FY)"                    NUMERIC,                      -- ratio: Return on Assets (ROA) % (FY) | alias: return_on_assets_roa_pct_fy
    "P/B (LTM)"                                        NUMERIC,                      -- ratio: P/B (LTM) | alias: p_b_ltm
    "P/B (-1FY)"                                       NUMERIC,                      -- ratio: P/B (-1FY) | alias: p_b_1fy
    "P/B (5YAVG)"                                      NUMERIC,                      -- ratio: P/B (5YAVG) | alias: p_b_5yavg
    "EV/Sales (EST FY1)"                               NUMERIC,                      -- ratio: EV/Sales (EST FY1) | alias: ev_sales_est_fy1
    "EV/Sales (LTM)"                                   NUMERIC,                      -- ratio: EV/Sales (LTM) | alias: ev_sales_ltm
    "EV/Sales (NTM)"                                   NUMERIC,                      -- ratio: EV/Sales (NTM) | alias: ev_sales_ntm
    "EV/Sales (-1FYLTM)"                               NUMERIC,                      -- ratio: EV/Sales (-1FYLTM) | alias: ev_sales_1fyltm
    "EV/Sales (-2FYLTM)"                               NUMERIC,                      -- ratio: EV/Sales (-2FYLTM) | alias: ev_sales_2fyltm
    "EV/Sales (-3FYLTM)"                               NUMERIC,                      -- ratio: EV/Sales (-3FYLTM) | alias: ev_sales_3fyltm
    "EV/Sales (3YAVGLTM)"                              NUMERIC,                      -- ratio: EV/Sales (3YAVGLTM) | alias: ev_sales_3yavgltm
    "EV/Sales (-1FQLTM)"                               NUMERIC,                      -- ratio: EV/Sales (-1FQLTM) | alias: ev_sales_1fqltm
    "EV/Sales (-2FQLTM)"                               NUMERIC,                      -- ratio: EV/Sales (-2FQLTM) | alias: ev_sales_2fqltm
    "EV/Sales (-3FQLTM)"                               NUMERIC,                      -- ratio: EV/Sales (-3FQLTM) | alias: ev_sales_3fqltm
    "EV/Sales (-4FQLTM)"                               NUMERIC,                      -- ratio: EV/Sales (-4FQLTM) | alias: ev_sales_4fqltm
    "EV/EBITDA (LTM)"                                  NUMERIC,                      -- ratio: EV/EBITDA (LTM) | alias: ev_ebitda_ltm
    "EV/EBITDA (NTM)"                                  NUMERIC,                      -- ratio: EV/EBITDA (NTM) | alias: ev_ebitda_ntm
    "EV/EBITDA (-1FYLTM)"                              NUMERIC,                      -- ratio: EV/EBITDA (-1FYLTM) | alias: ev_ebitda_1fyltm
    "EV/EBITDA (-1FQLTM)"                              NUMERIC,                      -- ratio: EV/EBITDA (-1FQLTM) | alias: ev_ebitda_1fqltm
    "EV/EBITDA (3YAVGLTM)"                             NUMERIC,                      -- ratio: EV/EBITDA (3YAVGLTM) | alias: ev_ebitda_3yavgltm
    "EV/EBITDA (EST FY1)"                              NUMERIC,                      -- ratio: EV/EBITDA (EST FY1) | alias: ev_ebitda_est_fy1
    "P/E (EST FY1)"                                    NUMERIC,                      -- ratio: P/E (EST FY1) | alias: p_e_est_fy1
    "P/E (-1FYLTM)"                                    NUMERIC,                      -- ratio: P/E (-1FYLTM) | alias: p_e_1fyltm
    "P/E (-2FYLTM)"                                    NUMERIC,                      -- ratio: P/E (-2FYLTM) | alias: p_e_2fyltm
    "P/E (-3FYLTM)"                                    NUMERIC,                      -- ratio: P/E (-3FYLTM) | alias: p_e_3fyltm
    "P/E (3YAVGLTM)"                                   NUMERIC,                      -- ratio: P/E (3YAVGLTM) | alias: p_e_3yavgltm
    "P/E (-1FQLTM)"                                    NUMERIC,                      -- ratio: P/E (-1FQLTM) | alias: p_e_1fqltm
    "P/E (-2FQLTM)"                                    NUMERIC,                      -- ratio: P/E (-2FQLTM) | alias: p_e_2fqltm
    "P/E (-3FQLTM)"                                    NUMERIC,                      -- ratio: P/E (-3FQLTM) | alias: p_e_3fqltm
    "P/E (5YAVGLTM)"                                   NUMERIC,                      -- ratio: P/E (5YAVGLTM) | alias: p_e_5yavgltm
    "P/E (-0FQQoQLTM)"                                 NUMERIC,                      -- ratio: P/E (-0FQQoQLTM) | alias: p_e_0fqqoqltm
    "P/E (-0FYYoYLTM)"                                 NUMERIC,                      -- ratio: P/E (-0FYYoYLTM) | alias: p_e_0fyyoyltm
    "P/E (-1FYYoYLTM)"                                 NUMERIC,                      -- ratio: P/E (-1FYYoYLTM) | alias: p_e_1fyyoyltm
    "P/E (-0FQYoYLTM)"                                 NUMERIC,                      -- ratio: P/E (-0FQYoYLTM) | alias: p_e_0fqyoyltm
    "Net EPS - Basic (LTM)"                            NUMERIC,                      -- ratio: Net EPS - Basic (LTM) | alias: net_eps_basic_ltm
    "Net EPS - Basic (FQ)"                             NUMERIC,                      -- ratio: Net EPS - Basic (FQ) | alias: net_eps_basic_fq
    "Net EPS - Basic (FY)"                             NUMERIC,                      -- ratio: Net EPS - Basic (FY) | alias: net_eps_basic_fy
    "Net EPS - Basic (-1FQFQ)"                         NUMERIC,                      -- ratio: Net EPS - Basic (-1FQFQ) | alias: net_eps_basic_1fqfq
    "Net EPS - Basic (-2FQFQ)"                         NUMERIC,                      -- ratio: Net EPS - Basic (-2FQFQ) | alias: net_eps_basic_2fqfq
    "Net EPS - Basic (-3FQFQ)"                         NUMERIC,                      -- ratio: Net EPS - Basic (-3FQFQ) | alias: net_eps_basic_3fqfq
    "Net EPS - Basic (-4FQFQ)"                         NUMERIC,                      -- ratio: Net EPS - Basic (-4FQFQ) | alias: net_eps_basic_4fqfq
    "Net EPS - Basic (-1FY)"                           NUMERIC,                      -- ratio: Net EPS - Basic (-1FY) | alias: net_eps_basic_1fy
    "Net EPS - Basic (-2FY)"                           NUMERIC,                      -- ratio: Net EPS - Basic (-2FY) | alias: net_eps_basic_2fy
    "Net EPS - Basic (-3FY)"                           NUMERIC,                      -- ratio: Net EPS - Basic (-3FY) | alias: net_eps_basic_3fy
    "Net EPS - Basic (-4FY)"                           NUMERIC,                      -- ratio: Net EPS - Basic (-4FY) | alias: net_eps_basic_4fy
    "Net EPS - Basic (-5FY)"                           NUMERIC,                      -- ratio: Net EPS - Basic (-5FY) | alias: net_eps_basic_5fy
    "EPS GAAP - Est Avg (NTM)"                         NUMERIC,                      -- ratio: EPS GAAP - Est Avg (NTM) | alias: eps_gaap_est_avg_ntm
    "EPS GAAP - Est Avg (FY1E)"                        NUMERIC,                      -- ratio: EPS GAAP - Est Avg (FY1E) | alias: eps_gaap_est_avg_fy1e
    "Basic EPS - Cont (LTM)"                           NUMERIC,                      -- ratio: Basic EPS - Cont (LTM) | alias: basic_eps_cont_ltm
    "Basic EPS - Cont (FQ)"                            NUMERIC,                      -- ratio: Basic EPS - Cont (FQ) | alias: basic_eps_cont_fq
    "Basic EPS - Cont (FY)"                            NUMERIC,                      -- ratio: Basic EPS - Cont (FY) | alias: basic_eps_cont_fy
    "Basic EPS - Cont (-1FQFQ)"                        NUMERIC,                      -- ratio: Basic EPS - Cont (-1FQFQ) | alias: basic_eps_cont_1fqfq
    "Basic EPS - Cont (-2FQFQ)"                        NUMERIC,                      -- ratio: Basic EPS - Cont (-2FQFQ) | alias: basic_eps_cont_2fqfq
    "Basic EPS - Cont (-3FQFQ)"                        NUMERIC,                      -- ratio: Basic EPS - Cont (-3FQFQ) | alias: basic_eps_cont_3fqfq
    "Basic EPS - Cont (-4FQFQ)"                        NUMERIC,                      -- ratio: Basic EPS - Cont (-4FQFQ) | alias: basic_eps_cont_4fqfq
    "Basic EPS - Cont (-1FY)"                          NUMERIC,                      -- ratio: Basic EPS - Cont (-1FY) | alias: basic_eps_cont_1fy
    "Basic EPS - Cont (-2FY)"                          NUMERIC,                      -- ratio: Basic EPS - Cont (-2FY) | alias: basic_eps_cont_2fy
    "Basic EPS - Cont (-3FY)"                          NUMERIC,                      -- ratio: Basic EPS - Cont (-3FY) | alias: basic_eps_cont_3fy
    "Basic EPS - Cont (-4FY)"                          NUMERIC,                      -- ratio: Basic EPS - Cont (-4FY) | alias: basic_eps_cont_4fy
    "EPS/Adj. (FQ)"                                    NUMERIC,                      -- ratio: EPS/Adj. (FQ) | alias: eps_adj_fq
    "EPS/Adj. (-1FQFQ)"                                NUMERIC,                      -- ratio: EPS/Adj. (-1FQFQ) | alias: eps_adj_1fqfq
    "EPS/Adj. (-2FQFQ)"                                NUMERIC,                      -- ratio: EPS/Adj. (-2FQFQ) | alias: eps_adj_2fqfq
    "EPS/Adj. (-3FQFQ)"                                NUMERIC,                      -- ratio: EPS/Adj. (-3FQFQ) | alias: eps_adj_3fqfq
    "EPS/Adj. (-4FQFQ)"                                NUMERIC,                      -- ratio: EPS/Adj. (-4FQFQ) | alias: eps_adj_4fqfq
    "EPS/Adj. (-2FY)"                                  NUMERIC,                      -- ratio: EPS/Adj. (-2FY) | alias: eps_adj_2fy
    "EPS/Adj. (-3FY)"                                  NUMERIC,                      -- ratio: EPS/Adj. (-3FY) | alias: eps_adj_3fy
    "EPS/Adj. (-4FY)"                                  NUMERIC,                      -- ratio: EPS/Adj. (-4FY) | alias: eps_adj_4fy

    -- ===========================================
    -- PERCENTAGE role
    -- ===========================================
    "Total Return (YTD)"                               NUMERIC,                      -- percentage: Total Return (YTD) | alias: total_return_ytd
    "Beta (1Y)"                                        NUMERIC,                      -- percentage: Beta (1Y) | alias: beta_1y
    "Beta (2Y)"                                        NUMERIC,                      -- percentage: Beta (2Y) | alias: beta_2y
    "Beta (5Y)"                                        NUMERIC,                      -- percentage: Beta (5Y) | alias: beta_5y
    "Total Revenues/CAGR (5Y FY)"                      NUMERIC,                      -- percentage: Total Revenues/CAGR (5Y FY) | alias: total_revenues_cagr_5y_fy
    "Tot. Return %/CAGR (3Y)"                          NUMERIC,                      -- percentage: Tot. Return %/CAGR (3Y) | alias: tot_return_pct_cagr_3y
    "Tot. Return %/CAGR (10Y)"                         NUMERIC,                      -- percentage: Tot. Return %/CAGR (10Y) | alias: tot_return_pct_cagr_10y
    "Total Return (5Y)"                                NUMERIC,                      -- percentage: Total Return (5Y) | alias: total_return_5y
    "Total Return (10Y)"                               NUMERIC,                      -- percentage: Total Return (10Y) | alias: total_return_10y
    "Net Income Margin % (FY)"                         NUMERIC,                      -- percentage: Net Income Margin % (FY) | alias: net_income_margin_pct_fy
    "Net Income Margin % (LTM)"                        NUMERIC,                      -- percentage: Net Income Margin % (LTM) | alias: net_income_margin_pct_ltm
    "Volatility (1M)"                                  NUMERIC,                      -- percentage: Volatility (1M) | alias: volatility_1m
    "Volatility (3M)"                                  NUMERIC,                      -- percentage: Volatility (3M) | alias: volatility_3m
    "Volatility (6M)"                                  NUMERIC,                      -- percentage: Volatility (6M) | alias: volatility_6m
    "Volatility (1Y)"                                  NUMERIC,                      -- percentage: Volatility (1Y) | alias: volatility_1y
    "Div Yield (Ind)"                                  NUMERIC DEFAULT 0,            -- percentage: Div Yield (Ind) | alias: div_yield_ind
    "Div Yield (LTM)"                                  NUMERIC DEFAULT 0,            -- percentage: Div Yield (LTM) | alias: div_yield_ltm
    "Gross Profit Margin % (FY)"                       NUMERIC,                      -- percentage: Gross Profit Margin % (FY) | alias: gross_profit_margin_pct_fy
    "Gross Profit Margin % (LTM)"                      NUMERIC,                      -- percentage: Gross Profit Margin % (LTM) | alias: gross_profit_margin_pct_ltm
    "Buyback Yield (LTM)"                              NUMERIC,                      -- percentage: Buyback Yield (LTM) | alias: buyback_yield_ltm
    "Div Yield (-1FYInd)"                              NUMERIC DEFAULT 0,            -- percentage: Div Yield (-1FYInd) | alias: div_yield_1fyind
    "Div Yield (TTM)"                                  NUMERIC DEFAULT 0,            -- percentage: Div Yield (TTM) | alias: div_yield_ttm
    "Div Yield (NTM)"                                  NUMERIC DEFAULT 0,            -- percentage: Div Yield (NTM) | alias: div_yield_ntm
    "Div Yield (5YAVGLTM)"                             NUMERIC DEFAULT 0,            -- percentage: Div Yield (5YAVGLTM) | alias: div_yield_5yavgltm
    "Revenues - Est YoY % (FY1E)"                      NUMERIC DEFAULT 0,            -- percentage: Revenues - Est YoY % (FY1E) | alias: revenues_est_yoy_pct_fy1e
    "Price Chg. % (1M)"                                NUMERIC DEFAULT 0,            -- percentage: Price Chg. % (1M) | alias: price_chg_pct_1m
    "Price Chg. % (3M)"                                NUMERIC DEFAULT 0,            -- percentage: Price Chg. % (3M) | alias: price_chg_pct_3m
    "1-Day %"                                          NUMERIC DEFAULT 0,            -- percentage: 1-Day % | alias: one_day_pct
    "EPS Est Avg Rev % (FY1E - 1W)"                    NUMERIC DEFAULT 0,            -- percentage: EPS Est Avg Rev % (FY1E - 1W) | alias: eps_est_avg_rev_pct_fy1e_1w
    "EPS Est Avg Rev % (FY1E - 1M)"                    NUMERIC DEFAULT 0,            -- percentage: EPS Est Avg Rev % (FY1E - 1M) | alias: eps_est_avg_rev_pct_fy1e_1m
    "EPS Est Avg Rev % (FY1E - 3M)"                    NUMERIC DEFAULT 0,            -- percentage: EPS Est Avg Rev % (FY1E - 3M) | alias: eps_est_avg_rev_pct_fy1e_3m
    "EPS Est Avg Rev % (FY1E - 6M)"                    NUMERIC DEFAULT 0,            -- percentage: EPS Est Avg Rev % (FY1E - 6M) | alias: eps_est_avg_rev_pct_fy1e_6m
    "EPS Est Avg Rev % (FY1E - 1Y)"                    NUMERIC DEFAULT 0,            -- percentage: EPS Est Avg Rev % (FY1E - 1Y) | alias: eps_est_avg_rev_pct_fy1e_1y
    "Div Yield (-2FYInd)"                              NUMERIC DEFAULT 0,            -- percentage: Div Yield (-2FYInd) | alias: div_yield_2fyind
    "Div Yield (-3FYInd)"                              NUMERIC DEFAULT 0,            -- percentage: Div Yield (-3FYInd) | alias: div_yield_3fyind
    "Div Yield (-4FYInd)"                              NUMERIC DEFAULT 0,            -- percentage: Div Yield (-4FYInd) | alias: div_yield_4fyind
    "Div Yield (-5FYInd)"                              NUMERIC DEFAULT 0,            -- percentage: Div Yield (-5FYInd) | alias: div_yield_5fyind
    "EPS GAAP Est Avg Rev % (FY1E - 1M)"               NUMERIC DEFAULT 0,            -- percentage: EPS GAAP Est Avg Rev % (FY1E - 1M) | alias: eps_gaap_est_avg_rev_pct_fy1e_1m
    "EPS GAAP Est Avg Rev % (FY1E - 3M)"               NUMERIC DEFAULT 0,            -- percentage: EPS GAAP Est Avg Rev % (FY1E - 3M) | alias: eps_gaap_est_avg_rev_pct_fy1e_3m
    "EPS GAAP Est Avg Rev % (FY1E - 6M)"               NUMERIC DEFAULT 0,            -- percentage: EPS GAAP Est Avg Rev % (FY1E - 6M) | alias: eps_gaap_est_avg_rev_pct_fy1e_6m
    "EPS GAAP Est Avg Rev % (FY1E - 1Y)"               NUMERIC DEFAULT 0,            -- percentage: EPS GAAP Est Avg Rev % (FY1E - 1Y) | alias: eps_gaap_est_avg_rev_pct_fy1e_1y

    -- ===========================================
    -- COUNT role
    -- ===========================================
    "Dividend Streak"                                  NUMERIC DEFAULT 0,            -- count: Consecutive years of dividend payments | alias: dividend_streak
    "Price Target - #"                                 NUMERIC DEFAULT 0,            -- count: Number of analyst price targets (alias for price_target_num) | alias: price_target_count
    "Analyst Rating"                                   NUMERIC DEFAULT 0,            -- count: Analyst Rating | alias: analyst_rating
    "# Strong Sell Ratings"                            NUMERIC DEFAULT 0,            -- count: # Strong Sell Ratings | alias: num_strong_sell_ratings
    "# Strong Buys Ratings"                            NUMERIC DEFAULT 0,            -- count: # Strong Buys Ratings | alias: num_strong_buys_ratings
    "# Hold Ratings"                                   NUMERIC DEFAULT 0,            -- count: # Hold Ratings | alias: num_hold_ratings
    "# Buys Ratings"                                   NUMERIC DEFAULT 0,            -- count: # Buys Ratings | alias: num_buys_ratings
    "# Sell Ratings"                                   NUMERIC DEFAULT 0,            -- count: # Sell Ratings | alias: num_sell_ratings
    "# No Opinion Ratings"                             NUMERIC DEFAULT 0,            -- count: # No Opinion Ratings | alias: num_no_opinion_ratings
    "Shrs Out"                                         NUMERIC DEFAULT 0,            -- count: Shares outstanding | alias: shares_outstanding
    "Shrs Out (-1FY)"                                  NUMERIC DEFAULT 0,            -- count: Shares outstanding (previous FY) | alias: shrs_out_1fy
    "Full Time Employees (FQ)"                         NUMERIC DEFAULT 0,            -- count: Full time employees (Fiscal Quarter) | alias: full_time_employees_fq
    "Full Time Employees (FY)"                         NUMERIC DEFAULT 0,            -- count: Full time employees (Fiscal Year) | alias: full_time_employees_fy
    "Full Time Employees (-1FY)"                       NUMERIC DEFAULT 0,            -- count: Full Time Employees (-1FY) | alias: full_time_employees_1fy
    "Full Time Employees (-2FY)"                       NUMERIC DEFAULT 0,            -- count: Full Time Employees (-2FY) | alias: full_time_employees_2fy
    "Full Time Employees (-3FY)"                       NUMERIC DEFAULT 0,            -- count: Full Time Employees (-3FY) | alias: full_time_employees_3fy
    "Avg Employees (5YAVGFY)"                          NUMERIC DEFAULT 0,            -- count: Avg Employees (5YAVGFY) | alias: avg_employees_5yavgfy
    "EPS Norm - Est # (FY1E)"                          NUMERIC DEFAULT 0,            -- count: EPS Norm - Est # (FY1E) | alias: eps_norm_est_num_fy1e
    "Price Target - # (3M Ago)"                        NUMERIC DEFAULT 0,            -- count: Price Target - # (3M Ago) | alias: price_target_num_3m_ago
    "Price Target - # (6M Ago)"                        NUMERIC DEFAULT 0,            -- count: Price Target - # (6M Ago) | alias: price_target_num_6m_ago
    "Price Target - # (YTD Ago)"                       NUMERIC DEFAULT 0,            -- count: Price Target - # (YTD Ago) | alias: price_target_num_ytd_ago
    "Price Target - # (1Y Ago)"                        NUMERIC DEFAULT 0,            -- count: Price Target - # (1Y Ago) | alias: price_target_num_1y_ago
    "Price Target - # (1W Ago)"                        NUMERIC DEFAULT 0,            -- count: Price Target - # (1W Ago) | alias: price_target_num_1w_ago
    "Price Target - # (1M Ago)"                        NUMERIC DEFAULT 0,            -- count: Price Target - # (1M Ago) | alias: price_target_num_1m_ago
    "Price Target - # (MTD Ago)"                       NUMERIC DEFAULT 0,            -- count: Price Target - # (MTD Ago) | alias: price_target_num_mtd_ago
    "Price Target - # (QTD Ago)"                       NUMERIC DEFAULT 0,            -- count: Price Target - # (QTD Ago) | alias: price_target_num_qtd_ago

    -- ===========================================
    -- NON RECURRING role
    -- ===========================================
    "Gain (Loss) On Sale Of Assets (LTM)"              NUMERIC DEFAULT 0,            -- income_statement: Gain (Loss) On Sale Of Assets (LTM) | alias: gain_loss_on_sale_of_assets_ltm
    "Impairment of Goodwill (FQ)"                      NUMERIC DEFAULT 0,            -- income_statement: Impairment of Goodwill (FQ) | alias: impairment_of_goodwill_fq
    "Impairment of Goodwill (LTM)"                     NUMERIC DEFAULT 0,            -- income_statement: Impairment of Goodwill (LTM) | alias: impairment_of_goodwill_ltm
    "Impairment of Goodwill (-1FY)"                    NUMERIC DEFAULT 0,            -- income_statement: Impairment of Goodwill (-1FY) | alias: impairment_of_goodwill_1fy
    "Impairment of Goodwill (FY)"                      NUMERIC DEFAULT 0,            -- income_statement: Impairment of Goodwill (FY) | alias: impairment_of_goodwill_fy
    "Asset Writedown (LTM)"                            NUMERIC DEFAULT 0,            -- income_statement: Asset Writedown (LTM) | alias: asset_writedown_ltm
    "Asset Writedown (FY)"                             NUMERIC DEFAULT 0,            -- income_statement: Asset Writedown (FY) | alias: asset_writedown_fy
    "Asset Writedown (-1FY)"                           NUMERIC DEFAULT 0,            -- income_statement: Asset Writedown (-1FY) | alias: asset_writedown_1fy
    "Restructuring Charges (LTM)"                      NUMERIC DEFAULT 0,            -- income_statement: Restructuring Charges (LTM) | alias: restructuring_charges_ltm
    "Restructuring Charges (FQ)"                       NUMERIC DEFAULT 0,            -- income_statement: Restructuring Charges (FQ) | alias: restructuring_charges_fq
    "Restructuring Charges (-1FY)"                     NUMERIC DEFAULT 0,            -- income_statement: Restructuring Charges (-1FY) | alias: restructuring_charges_1fy
    "Restructuring Charges (FY)"                       NUMERIC DEFAULT 0,            -- income_statement: Restructuring Charges (FY) | alias: restructuring_charges_fy
    "Merger & Restructuring Charges (LTM)"             NUMERIC DEFAULT 0,            -- income_statement: Merger & Restructuring Charges (LTM) | alias: merger_and_restructuring_charges_ltm
    "Other Unusual Items/Total (LTM)"                  NUMERIC DEFAULT 0,            -- income_statement: Other Unusual Items/Total (LTM) | alias: other_unusual_items_total_ltm
    "Asset Writedown (FQ)"                             NUMERIC DEFAULT 0,            -- income_statement: Asset Writedown (FQ) | alias: asset_writedown_fq
    "Asset Writedown (5YAVGFQ)"                        NUMERIC DEFAULT 0,            -- income_statement: Asset Writedown (5YAVGFQ) | alias: asset_writedown_5yavgfq
    "Impairment of Goodwill (5YAVGFQ)"                 NUMERIC DEFAULT 0,            -- income_statement: Impairment of Goodwill (5YAVGFQ) | alias: impairment_of_goodwill_5yavgfq
    "Restructuring Charges (5YAVGFQ)"                  NUMERIC DEFAULT 0,            -- income_statement: Restructuring Charges (5YAVGFQ) | alias: restructuring_charges_5yavgfq
    "Merger & Restructuring Charges (FQ)"              NUMERIC DEFAULT 0,            -- income_statement: Merger & Restructuring Charges (FQ) | alias: merger_and_restructuring_charges_fq
    "Merger & Restructuring Charges (FY)"              NUMERIC DEFAULT 0,            -- income_statement: Merger & Restructuring Charges (FY) | alias: merger_and_restructuring_charges_fy
    "Merger & Restructuring Charges (5YAVGFQ)"         NUMERIC DEFAULT 0,            -- income_statement: Merger & Restructuring Charges (5YAVGFQ) | alias: merger_and_restructuring_charges_5yavgfq
    "Merger & Restructuring Charges (-1FQFQ)"          NUMERIC DEFAULT 0,            -- income_statement: Merger & Restructuring Charges (-1FQFQ) | alias: merger_and_restructuring_charges_1fqfq
    "Merger & Restructuring Charges (-2FQFQ)"          NUMERIC DEFAULT 0,            -- income_statement: Merger & Restructuring Charges (-2FQFQ) | alias: merger_and_restructuring_charges_2fqfq
    "Merger & Restructuring Charges (-3FQFQ)"          NUMERIC DEFAULT 0,            -- income_statement: Merger & Restructuring Charges (-3FQFQ) | alias: merger_and_restructuring_charges_3fqfq
    "Merger & Restructuring Charges (-4FQFQ)"          NUMERIC DEFAULT 0,            -- income_statement: Merger & Restructuring Charges (-4FQFQ) | alias: merger_and_restructuring_charges_4fqfq
    "Merger & Restructuring Charges (-1FY)"            NUMERIC DEFAULT 0,            -- income_statement: Merger & Restructuring Charges (-1FY) | alias: merger_and_restructuring_charges_1fy
    "Merger & Restructuring Charges (-2FY)"            NUMERIC DEFAULT 0,            -- income_statement: Merger & Restructuring Charges (-2FY) | alias: merger_and_restructuring_charges_2fy
    "Merger & Restructuring Charges (-3FY)"            NUMERIC DEFAULT 0,            -- income_statement: Merger & Restructuring Charges (-3FY) | alias: merger_and_restructuring_charges_3fy
    "Merger & Restructuring Charges (-4FY)"            NUMERIC DEFAULT 0,            -- income_statement: Merger & Restructuring Charges (-4FY) | alias: merger_and_restructuring_charges_4fy
    "Impairment of Goodwill (-1FQFQ)"                  NUMERIC DEFAULT 0,            -- income_statement: Impairment of Goodwill (-1FQFQ) | alias: impairment_of_goodwill_1fqfq
    "Impairment of Goodwill (-2FQFQ)"                  NUMERIC DEFAULT 0,            -- income_statement: Impairment of Goodwill (-2FQFQ) | alias: impairment_of_goodwill_2fqfq
    "Impairment of Goodwill (-3FQFQ)"                  NUMERIC DEFAULT 0,            -- income_statement: Impairment of Goodwill (-3FQFQ) | alias: impairment_of_goodwill_3fqfq
    "Impairment of Goodwill (-4FQFQ)"                  NUMERIC DEFAULT 0,            -- income_statement: Impairment of Goodwill (-4FQFQ) | alias: impairment_of_goodwill_4fqfq
    "Impairment of Goodwill (-2FY)"                    NUMERIC DEFAULT 0,            -- income_statement: Impairment of Goodwill (-2FY) | alias: impairment_of_goodwill_2fy
    "Impairment of Goodwill (-3FY)"                    NUMERIC DEFAULT 0,            -- income_statement: Impairment of Goodwill (-3FY) | alias: impairment_of_goodwill_3fy
    "Impairment of Goodwill (-4FY)"                    NUMERIC DEFAULT 0,            -- income_statement: Impairment of Goodwill (-4FY) | alias: impairment_of_goodwill_4fy
    "Asset Writedown (-1FQFQ)"                         NUMERIC DEFAULT 0,            -- income_statement: Asset Writedown (-1FQFQ) | alias: asset_writedown_1fqfq
    "Asset Writedown (-2FQFQ)"                         NUMERIC DEFAULT 0,            -- income_statement: Asset Writedown (-2FQFQ) | alias: asset_writedown_2fqfq
    "Asset Writedown (-3FQFQ)"                         NUMERIC DEFAULT 0,            -- income_statement: Asset Writedown (-3FQFQ) | alias: asset_writedown_3fqfq
    "Asset Writedown (-4FQFQ)"                         NUMERIC DEFAULT 0,            -- income_statement: Asset Writedown (-4FQFQ) | alias: asset_writedown_4fqfq
    "Asset Writedown (-2FY)"                           NUMERIC DEFAULT 0,            -- income_statement: Asset Writedown (-2FY) | alias: asset_writedown_2fy
    "Asset Writedown (-3FY)"                           NUMERIC DEFAULT 0,            -- income_statement: Asset Writedown (-3FY) | alias: asset_writedown_3fy
    "Asset Writedown (-4FY)"                           NUMERIC DEFAULT 0,            -- income_statement: Asset Writedown (-4FY) | alias: asset_writedown_4fy
    "Asset Writedown (-5FY)"                           NUMERIC DEFAULT 0,            -- income_statement: Asset Writedown (-5FY) | alias: asset_writedown_5fy
    "Gain (Loss) On Sale Of Assets (FQ)"               NUMERIC DEFAULT 0,            -- income_statement: Gain (Loss) On Sale Of Assets (FQ) | alias: gain_loss_on_sale_of_assets_fq
    "Gain (Loss) On Sale Of Assets (FY)"               NUMERIC DEFAULT 0,            -- income_statement: Gain (Loss) On Sale Of Assets (FY) | alias: gain_loss_on_sale_of_assets_fy
    "Gain (Loss) On Sale Of Assets (-1FQFQ)"           NUMERIC DEFAULT 0,            -- income_statement: Gain (Loss) On Sale Of Assets (-1FQFQ) | alias: gain_loss_on_sale_of_assets_1fqfq
    "Gain (Loss) On Sale Of Assets (-2FQFQ)"           NUMERIC DEFAULT 0,            -- income_statement: Gain (Loss) On Sale Of Assets (-2FQFQ) | alias: gain_loss_on_sale_of_assets_2fqfq
    "Gain (Loss) On Sale Of Assets (-3FQFQ)"           NUMERIC DEFAULT 0,            -- income_statement: Gain (Loss) On Sale Of Assets (-3FQFQ) | alias: gain_loss_on_sale_of_assets_3fqfq
    "Gain (Loss) On Sale Of Assets (-4FQFQ)"           NUMERIC DEFAULT 0,            -- income_statement: Gain (Loss) On Sale Of Assets (-4FQFQ) | alias: gain_loss_on_sale_of_assets_4fqfq
    "Gain (Loss) On Sale Of Assets (-1FY)"             NUMERIC DEFAULT 0,            -- income_statement: Gain (Loss) On Sale Of Assets (-1FY) | alias: gain_loss_on_sale_of_assets_1fy
    "Gain (Loss) On Sale Of Assets (-2FY)"             NUMERIC DEFAULT 0,            -- income_statement: Gain (Loss) On Sale Of Assets (-2FY) | alias: gain_loss_on_sale_of_assets_2fy
    "Gain (Loss) On Sale Of Assets (-3FY)"             NUMERIC DEFAULT 0,            -- income_statement: Gain (Loss) On Sale Of Assets (-3FY) | alias: gain_loss_on_sale_of_assets_3fy
    "Gain (Loss) On Sale Of Assets (-4FY)"             NUMERIC DEFAULT 0,            -- income_statement: Gain (Loss) On Sale Of Assets (-4FY) | alias: gain_loss_on_sale_of_assets_4fy
    "Restructuring Charges (-1FQFQ)"                   NUMERIC DEFAULT 0,            -- income_statement: Restructuring Charges (-1FQFQ) | alias: restructuring_charges_1fqfq
    "Restructuring Charges (-2FQFQ)"                   NUMERIC DEFAULT 0,            -- income_statement: Restructuring Charges (-2FQFQ) | alias: restructuring_charges_2fqfq
    "Restructuring Charges (-3FQFQ)"                   NUMERIC DEFAULT 0,            -- income_statement: Restructuring Charges (-3FQFQ) | alias: restructuring_charges_3fqfq
    "Restructuring Charges (-4FQFQ)"                   NUMERIC DEFAULT 0,            -- income_statement: Restructuring Charges (-4FQFQ) | alias: restructuring_charges_4fqfq
    "Restructuring Charges (-2FY)"                     NUMERIC DEFAULT 0,            -- income_statement: Restructuring Charges (-2FY) | alias: restructuring_charges_2fy
    "Restructuring Charges (-3FY)"                     NUMERIC DEFAULT 0,            -- income_statement: Restructuring Charges (-3FY) | alias: restructuring_charges_3fy
    "Restructuring Charges (-4FY)"                     NUMERIC DEFAULT 0,            -- income_statement: Restructuring Charges (-4FY) | alias: restructuring_charges_4fy
    "Interest And Investment Income (LTM)"             NUMERIC DEFAULT 0,            -- income_statement: Interest And Investment Income (LTM) | alias: interest_and_investment_income_ltm
    "Interest And Investment Income (FQ)"              NUMERIC DEFAULT 0,            -- income_statement: Interest And Investment Income (FQ) | alias: interest_and_investment_income_fq
    "Interest And Investment Income (FY)"              NUMERIC DEFAULT 0,            -- income_statement: Interest And Investment Income (FY) | alias: interest_and_investment_income_fy
    "Interest And Investment Income (-1FQFQ)"          NUMERIC DEFAULT 0,            -- income_statement: Interest And Investment Income (-1FQFQ) | alias: interest_and_investment_income_1fqfq
    "Interest And Investment Income (-2FQFQ)"          NUMERIC DEFAULT 0,            -- income_statement: Interest And Investment Income (-2FQFQ) | alias: interest_and_investment_income_2fqfq
    "Interest And Investment Income (-3FQFQ)"          NUMERIC DEFAULT 0,            -- income_statement: Interest And Investment Income (-3FQFQ) | alias: interest_and_investment_income_3fqfq
    "Interest And Investment Income (-4FQFQ)"          NUMERIC DEFAULT 0,            -- income_statement: Interest And Investment Income (-4FQFQ) | alias: interest_and_investment_income_4fqfq
    "Interest And Investment Income (-1FY)"            NUMERIC DEFAULT 0,            -- income_statement: Interest And Investment Income (-1FY) | alias: interest_and_investment_income_1fy
    "Interest And Investment Income (-2FY)"            NUMERIC DEFAULT 0,            -- income_statement: Interest And Investment Income (-2FY) | alias: interest_and_investment_income_2fy
    "Interest And Investment Income (-3FY)"            NUMERIC DEFAULT 0,            -- income_statement: Interest And Investment Income (-3FY) | alias: interest_and_investment_income_3fy
    "Interest And Investment Income (-4FY)"            NUMERIC DEFAULT 0,            -- income_statement: Interest And Investment Income (-4FY) | alias: interest_and_investment_income_4fy
    "Effective Tax Rate - (Ratio) (LTM)"               NUMERIC DEFAULT 0,            -- percentage: "Effective Tax Rate - (Ratio) (LTM)" | alias: effective_tax_rate_ltm
    "Effective Tax Rate - (Ratio) (FQ)"                NUMERIC DEFAULT 0,            -- percentage: "Effective Tax Rate - (Ratio) (LTM)" | alias: effective_tax_rate_fq
    "Effective Tax Rate - (Ratio) (-1FQFQ)"            NUMERIC DEFAULT 0,            -- percentage: "Effective Tax Rate - (Ratio) (LTM)" | alias: effective_tax_rate_1fqfq
    "Effective Tax Rate - (Ratio) (-2FQFQ)"            NUMERIC DEFAULT 0,            -- percentage: "Effective Tax Rate - (Ratio) (LTM)" | alias: effective_tax_rate_2fqfq
    "Effective Tax Rate - (Ratio) (-4FQFQ)"            NUMERIC DEFAULT 0,            -- percentage: "Effective Tax Rate - (Ratio) (LTM)" | alias: effective_tax_rate_4fqfq
    "Effective Tax Rate - (Ratio) (-3FQFQ)"            NUMERIC DEFAULT 0,            -- percentage: "Effective Tax Rate - (Ratio) (LTM)" | alias: effective_tax_rate_3fqfq
    "Effective Tax Rate - (Ratio) (FY)"                NUMERIC DEFAULT 0,            -- percentage: "Effective Tax Rate - (Ratio) (LTM)" | alias: effective_tax_rate_fy
    "Effective Tax Rate - (Ratio) (-1FY)"              NUMERIC DEFAULT 0,            -- percentage: "Effective Tax Rate - (Ratio) (LTM)" | alias: effective_tax_rate_1fy
    "Effective Tax Rate - (Ratio) (-2FY)"              NUMERIC DEFAULT 0,            -- percentage: "Effective Tax Rate - (Ratio) (LTM)" | alias: effective_tax_rate_2fy
    "Effective Tax Rate - (Ratio) (-3FY)"              NUMERIC DEFAULT 0,            -- percentage: "Effective Tax Rate - (Ratio) (LTM)" | alias: effective_tax_rate_3fy
    "Effective Tax Rate - (Ratio) (-4FY)"              NUMERIC DEFAULT 0,            -- percentage: "Effective Tax Rate - (Ratio) (LTM)" | alias: effective_tax_rate_4fy
    "FCF - Est Avg (FY1E)"                             NUMERIC DEFAULT 0,            -- cash_flow: FCF - Est Avg (FY1E) | alias: fcf_est_avg_fy1e
    "FCF - Est Avg (FY2E)"                             NUMERIC DEFAULT 0,            -- cash_flow: FCF - Est Avg (FY2E) | alias: fcf_est_avg_fy2e
    "FCF - Est Avg (FY3E)"                             NUMERIC DEFAULT 0,            -- cash_flow: FCF - Est Avg (FY3E) | alias: fcf_est_avg_fy3e
    "FCF - Est Avg (FY4E)"                             NUMERIC DEFAULT 0,            -- cash_flow: FCF - Est Avg (FY4E) | alias: fcf_est_avg_fy4e
    "FCF - Est Avg (FY5E)"                             NUMERIC DEFAULT 0,            -- cash_flow: FCF - Est Avg (FY5E) | alias: fcf_est_avg_fy5e
    "Total Operating Expenses (LTM)"                   NUMERIC DEFAULT 0,            -- income_statement: Total Operating Expenses (LTM) | alias: total_operating_expenses_ltm
    "Total Operating Expenses (FQ)"                    NUMERIC DEFAULT 0,            -- income_statement: Total Operating Expenses (FQ) | alias: total_operating_expenses_fq
    "Total Operating Expenses (FY)"                    NUMERIC DEFAULT 0,            -- income_statement: Total Operating Expenses (FY) | alias: total_operating_expenses_fy
    "Total Operating Expenses (-1FQFQ)"                NUMERIC DEFAULT 0,            -- income_statement: Total Operating Expenses (-1FQFQ) | alias: total_operating_expenses_1fqfq
    "Total Operating Expenses (-2FQFQ)"                NUMERIC DEFAULT 0,            -- income_statement: Total Operating Expenses (-2FQFQ) | alias: total_operating_expenses_2fqfq
    "Total Operating Expenses (-3FQFQ)"                NUMERIC DEFAULT 0,            -- income_statement: Total Operating Expenses (-3FQFQ) | alias: total_operating_expenses_3fqfq
    "Total Operating Expenses (-4FQFQ)"                NUMERIC DEFAULT 0,            -- income_statement: Total Operating Expenses (-4FQFQ) | alias: total_operating_expenses_4fqfq
    "Total Operating Expenses (-1FY)"                  NUMERIC DEFAULT 0,            -- income_statement: Total Operating Expenses (-1FY) | alias: total_operating_expenses_1fy
    "Total Operating Expenses (-2FY)"                  NUMERIC DEFAULT 0,            -- income_statement: Total Operating Expenses (-2FY) | alias: total_operating_expenses_2fy
    "Total Operating Expenses (-3FY)"                  NUMERIC DEFAULT 0,            -- income_statement: Total Operating Expenses (-3FY) | alias: total_operating_expenses_3fy
    "Total Operating Expenses (-4FY)"                  NUMERIC DEFAULT 0,            -- income_statement: Total Operating Expenses (-4FY) | alias: total_operating_expenses_4fy

    -- ===========================================
    -- FEATURE role
    -- ===========================================
    "Fiscal Month"                                     INTEGER,                      -- feature: Months between Income Statement Report Date and FY End Date | alias: fiscal_month
    "Fiscal Quarter"                                   INTEGER,                      -- feature: Fiscal quarter (1-4) from report date | alias: fiscal_quarter
    "Fiscal Year"                                      INTEGER,                      -- feature: Fiscal year from report date | alias: fiscal_year
    "Reporting Lag"                                    NUMERIC                       -- feature: Reporting Lag | alias: reporting_lag
) TABLESPACE pg_default;
COMMENT ON TABLE equities IS 'Equities screening data with financial metrics and company information';

-- ============================================================
-- Index Optimization Migration for postgres.public schema
-- ============================================================

BEGIN;

-- 2. EQUITIES: Consolidate geographic indexes
DROP INDEX IF EXISTS idx_equities_region;
DROP INDEX IF EXISTS idx_equities_country;
DROP INDEX IF EXISTS idx_equities_trading_country;
DROP INDEX IF EXISTS idx_equities_exchange;
CREATE INDEX idx_equities_geography ON equities ("Region", "Country", "Exchange");

-- 3. EQUITIES: Consolidate classification indexes
DROP INDEX IF EXISTS idx_equities_sector;
DROP INDEX IF EXISTS idx_equities_industry;
DROP INDEX IF EXISTS idx_equities_style_class;
DROP INDEX IF EXISTS idx_equities_size_class;
CREATE INDEX idx_equities_classification
    ON equities ("Sector", "Industry", "Size Class", "Style Class");

-- 4. EQUITIES: Optimize name index
DROP INDEX IF EXISTS idx_equities_name;
CREATE INDEX idx_equities_name ON equities ("Name" text_pattern_ops);

-- 5. EQUITIES: Add analytical indexes
CREATE INDEX idx_equities_fiscal
    ON equities ("Fiscal Year", "Fiscal Quarter", "Income Statement Report Date");
CREATE INDEX idx_equities_market_cap ON equities ("Market Cap" DESC NULLS LAST);

-- 6. MV_ALL_STOCK_FEATURES: Remove redundant ISIN index
DROP INDEX IF EXISTS idx_mv_all_stock_features_isin;

-- 7. EQUITIES_SCHEMA_METADATA: Consider dropping DDL index (verify usage first)
-- DROP INDEX IF EXISTS idx_equities_schema_metadata_ddl;

COMMIT;

