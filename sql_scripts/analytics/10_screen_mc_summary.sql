create table analytics."10_screen_mc_summary"
(
	isin        text,
	er_mean     double precision,
	er_sd       double precision,
	er_p05      double precision,
	er_p50      double precision,
	er_p95      double precision,
	prob_pos    double precision,
	run_id      text,
	exported_at timestamp with time zone
)
;

alter table analytics."10_screen_mc_summary"
	owner to postgres
;