create table analytics."10_screen_mc_summary_v2"
(
	isin         text,
	er_mean      double precision,
	er_sd        double precision,
	er_p05       double precision,
	er_p50       double precision,
	er_p95       double precision,
	mc_prob_pos  double precision,
	run_id       text,
	exported_at  timestamp with time zone,
	source_sha   text,
	source_dirty boolean
)
;

alter table analytics."10_screen_mc_summary_v2"
	owner to postgres
;