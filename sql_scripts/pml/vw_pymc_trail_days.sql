create view pml.vw_pymc_trail_days(lookback_key, response_column, trail_days, trail_rank)
as
SELECT lookback_key,
       response_column,
       trail_days,
       trail_rank
FROM (VALUES ('now'::text, 'feat_log_uplift_now'::text, 0, 0),
             ('1w'::text, 'feat_log_uplift_1w'::text, 7, 1),
             ('1m'::text, 'feat_log_uplift_1m'::text, 30, 2),
             ('3m'::text, 'feat_log_uplift_3m'::text, 91, 3),
             ('6m'::text, 'feat_log_uplift_6m'::text, 182, 4),
             ('1y'::text, 'feat_log_uplift_1y'::text, 365, 5)) v(lookback_key, response_column, trail_days, trail_rank)
;

comment on view pml.vw_pymc_trail_days is 'SSOT for the Kalman v2 OU kernel x-axis: lookback key -> response column -> nominal calendar offset in days. Read by KalmanFilterModel_v2.load_trail_days_map(); replaces the per-row trail_days_* columns dropped from mv_pymc_kalman_pt_v2 in favour of one metadata row per lookback.'
;

alter table pml.vw_pymc_trail_days
	owner to postgres
;