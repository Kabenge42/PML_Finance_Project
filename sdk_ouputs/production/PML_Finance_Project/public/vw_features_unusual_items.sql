create view vw_features_unusual_items
            (isin, ticker, name, description, region, country, trading_country, exchange, sector, industry,
             dividend_record_frequency, earnings_report_frequency, fy_end, next_earnings_report, next_earnings_status,
             next_earnings_when, next_fiscal_quarter, reporting_interval, size_class, style_class, unit,
             dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date,
             dividend_record_record_date, fy_end_date, income_statement_report_date, last_updated, next_earnings,
             next_fy_end_date, next_income_statement_report_date, reference_date, other_unusual_items_ltm,
             impairment_goodwill_ltm, asset_writedown_ltm, restructuring_charges_ltm, total_unusual_items,
             unusual_items_to_revenue, unusual_items_to_ebitda, has_unusual_items_flag, earnings_quality_impact)
as
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
       uif.other_unusual_items_ltm,
       uif.impairment_goodwill_ltm,
       uif.asset_writedown_ltm,
       uif.restructuring_charges_ltm,
       uif.total_unusual_items,
       uif.unusual_items_to_revenue,
       uif.unusual_items_to_ebitda,
       uif.has_unusual_items_flag,
       uif.earnings_quality_impact
FROM vw_identifier_columns id
         LEFT JOIN calc_unusual_items_features() uif(isin, other_unusual_items_ltm, impairment_goodwill_ltm,
                                                     asset_writedown_ltm, restructuring_charges_ltm,
                                                     total_unusual_items, unusual_items_to_revenue,
                                                     unusual_items_to_ebitda, has_unusual_items_flag,
                                                     earnings_quality_impact) USING (isin);

comment on view vw_features_unusual_items is 'Non-recurring and unusual items analysis for earnings quality assessment.
    Source function: calc_unusual_items_features';

alter table vw_features_unusual_items
    owner to postgres;

