create view vw_identifier_columns
            (isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
             dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
             next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
             dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
             dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
             next_fy_end_date, next_income_statement_report_date, reference_date)
as
SELECT "ISIN"                              AS isin,
       "Ticker"                            AS ticker,
       "Name"                              AS name,
       "Description"                       AS description,
       "Region"                            AS region,
       "Country"                           AS country,
       "Trading Country"                   AS trading_country,
       "Exchange"                          AS exchange,
       "Sector"                            AS sector,
       "Industry"                          AS industry,
       "Dividend Record (Frequency)"       AS dividend_record_frequency,
       "Earnings Report (Frequency)"       AS earnings_report_frequency,
       "FY End"                            AS fy_end,
       "Next Earnings (Report)"            AS next_earnings_report,
       "Next Earnings (Status)"            AS next_earnings_status,
       "Next Earnings (When)"              AS next_earnings_when,
       "Next Fiscal Quarter"               AS next_fiscal_quarter,
       "Reporting Interval"                AS reporting_interval,
       "Size Class"                        AS size_class,
       "Style Class"                       AS style_class,
       "Unit"                              AS unit,
       "Dividend Record (Announce Date)"   AS dividend_record_announce_date,
       "Dividend Record (Ex Date)"         AS dividend_record_ex_date,
       "Dividend Record (Payable Date)"    AS dividend_record_payable_date,
       "Dividend Record (Record Date)"     AS dividend_record_record_date,
       "FY End Date"                       AS fy_end_date,
       "Income Statement Report Date"      AS income_statement_report_date,
       "Last Updated"                      AS last_updated,
       "Next Earnings"                     AS next_earnings,
       "Next FY End Date"                  AS next_fy_end_date,
       "Next Income Statement Report Date" AS next_income_statement_report_date,
       "Reference Date"                    AS reference_date
FROM equities e;

alter table vw_identifier_columns
    owner to postgres;

