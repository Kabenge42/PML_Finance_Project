create function calc_employment_features(p_isin text DEFAULT NULL::text)
    returns TABLE(isin text, revenue_per_employee numeric, profit_per_employee numeric, ebitda_per_employee numeric, assets_per_employee numeric, fte_growth_1y_pct numeric, fte_growth_3y_pct numeric, workforce_stability numeric)
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"  AS isin,
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Total Revenues (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END AS revenue_per_employee,
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Normalized Net Income (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END AS profit_per_employee,
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "EBITDA (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END AS ebitda_per_employee,
       CASE
           WHEN "Full Time Employees (FY)" > 0
               THEN "Total Assets (FY)" / NULLIF("Full Time Employees (FY)", 0)
           END AS assets_per_employee,
       CASE
           WHEN "Full Time Employees (-1FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-1FY)") /
                    NULLIF("Full Time Employees (-1FY)", 0) * 100
           END AS fte_growth_1y_pct,
       CASE
           WHEN "Full Time Employees (-3FY)" > 0
               THEN ("Full Time Employees (FY)" - "Full Time Employees (-3FY)") /
                    NULLIF("Full Time Employees (-3FY)", 0) * 100
           END AS fte_growth_3y_pct,
       CASE
           WHEN "Avg Employees (5YAVGFY)" > 0
               THEN "Full Time Employees (FY)" / NULLIF("Avg Employees (5YAVGFY)", 0)
           END AS workforce_stability
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_employment_features(text) owner to postgres;

