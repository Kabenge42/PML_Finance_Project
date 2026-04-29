create function calc_employment_dynamics(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, fte_growth_2y_pct numeric, fte_acceleration numeric, workforce_volatility numeric, hiring_intensity numeric, productivity_trend numeric, headcount_vs_revenue numeric, workforce_efficiency_gain numeric, layoff_risk_flag integer, rapid_hiring_flag integer, sustainable_growth_flag integer)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"                                             AS isin,
       CASE
           WHEN "Full Time Employees (-2FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-2FY)") /
                    NULLIF("Full Time Employees (-2FY)", 0) * 100
           END                                            AS fte_growth_2y_pct,
       CASE
           WHEN "Full Time Employees (-1FY)" > 0 AND "Full Time Employees (-3FY)" > 0
               THEN (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) -
                    (POWER("Full Time Employees (FY)" / NULLIF("Full Time Employees (-3FY)", 0), 1.0 / 3.0) - 1)
           END * 100                                      AS fte_acceleration,
       ABS(("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
           NULLIF("Full Time Employees (-1FY)", 0) -
           ("Full Time Employees (-1FY)" - "Full Time Employees (-2FY)") /
           NULLIF("Full Time Employees (-2FY)", 0)) * 100 AS workforce_volatility,
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) > 0
               THEN (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) /
                    NULLIF((("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                            NULLIF(ABS("Total Revenues (-1FY)"), 0)), 0)
           END                                            AS hiring_intensity,
       CASE
           WHEN "Full Time Employees (FY)" > 0 AND "Full Time Employees (-1FY)" > 0
               THEN (("Total Revenues (FY)" / "Full Time Employees (FY)") -
                     ("Total Revenues (-1FY)" / "Full Time Employees (-1FY)")) /
                    NULLIF(ABS("Total Revenues (-1FY)" / "Full Time Employees (-1FY)"), 0) * 100
           END                                            AS productivity_trend,
       (("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
        NULLIF("Full Time Employees (-1FY)", 0) * 100) -
       (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
        NULLIF(ABS("Total Revenues (-1FY)"), 0) * 100)    AS headcount_vs_revenue,
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) >
                ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0)
               THEN (("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                     NULLIF(ABS("Total Revenues (-1FY)"), 0) -
                     ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                     NULLIF("Full Time Employees (-1FY)", 0)) * 100
           ELSE 0
           END                                            AS workforce_efficiency_gain,
       CASE
           WHEN "Full Time Employees (FY)" < "Full Time Employees (-1FY)"
               AND "Total Revenues (FY)" < "Total Revenues (-1FY)"
               THEN 1
           ELSE 0
           END                                            AS layoff_risk_flag,
       CASE
           WHEN ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0) > 0.20
               THEN 1
           ELSE 0
           END                                            AS rapid_hiring_flag,
       CASE
           WHEN ("Total Revenues (FY)" - "Total Revenues (-1FY)") /
                NULLIF(ABS("Total Revenues (-1FY)"), 0) >
                ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                NULLIF("Full Time Employees (-1FY)", 0)
               AND ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") > 0
               THEN 1
           ELSE 0
           END                                            AS sustainable_growth_flag
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_employment_dynamics(text) owner to postgres;

