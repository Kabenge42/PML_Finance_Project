create function pml.kalman_pt_v2_asof(p_asof date DEFAULT CURRENT_DATE)
	returns TABLE(isin text, days_to_next_earnings integer, days_since_last_report integer, days_to_next_fy_end integer, days_to_next_report integer, days_to_expected_report integer, days_since_fy_end integer, asof_date date)
	stable
	parallel safe
	language sql
as
$$
SELECT v.isin,
       (v.next_earnings - p_asof)::INT,
       (p_asof - v.income_statement_report_date)::INT,
       (v.next_fy_end_date - p_asof)::INT,
       (v.next_income_statement_report_date - p_asof)::INT,
       (v.expected_report_date - p_asof)::INT,
       (v.fy_end_date - p_asof)::INT,
       p_asof
FROM pml.mv_pymc_kalman_pt_v2 v;
$$
;

comment on function pml.kalman_pt_v2_asof(date) is 'Recompute the date-relative horizon columns against an arbitrary as-of date. Horizons only -- the price/target trails are not versioned, so this is not a full historical replay. Sign conventions match mv_pymc_kalman_pt.'
;

alter function pml.kalman_pt_v2_asof(date) owner to postgres
;