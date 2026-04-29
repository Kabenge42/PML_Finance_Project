create function calc_dividend_timing(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin                     text,
                days_since_ex_date       integer,
                days_to_payment          integer,
                dividend_announced_flag  integer,
                ex_date_approaching_flag integer,
                dividend_frequency_score integer,
                dividend_consistency     numeric,
                recent_dividend_change   numeric,
                dividend_yield_vs_5y_avg numeric
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                                     AS isin,
       (CURRENT_DATE - "Dividend Record (Ex Date)")::INTEGER      AS days_since_ex_date,
       ("Dividend Record (Payable Date)" - CURRENT_DATE)::INTEGER AS days_to_payment,
       CASE
           WHEN (CURRENT_DATE - "Dividend Record (Announce Date)") <= 30
               THEN 1
           ELSE 0
           END                                                    AS dividend_announced_flag,
       CASE
           WHEN ("Dividend Record (Ex Date)" - CURRENT_DATE) BETWEEN 0 AND 14
               THEN 1
           ELSE 0
           END                                                    AS ex_date_approaching_flag,
       CASE "Dividend Record (Frequency)"
           WHEN 'Quarterly' THEN 4
           WHEN 'Semi-Annual' THEN 2
           WHEN 'Annual' THEN 1
           WHEN 'Monthly' THEN 12
           ELSE 0
           END                                                    AS dividend_frequency_score,
       LEAST(1.0, "Dividend Streak"::NUMERIC / 10.0)              AS dividend_consistency,
       CASE
           WHEN "Div Yield (-1FYInd)" > 0
               THEN ("Div Yield (Ind)" - "Div Yield (-1FYInd)") /
                    NULLIF("Div Yield (-1FYInd)", 0) * 100
           END                                                    AS recent_dividend_change,
       "Div Yield (LTM)" / NULLIF("Div Yield (5YAVGLTM)", 0)      AS dividend_yield_vs_5y_avg
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_dividend_timing(text) owner to postgres;

