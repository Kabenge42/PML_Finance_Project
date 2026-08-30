create table analytics."09_diagnostics_v2"
(
	index        text,
	mean         double precision,
	sd           double precision,
	eti89_lb     double precision,
	eti89_ub     double precision,
	ess_bulk     double precision,
	ess_tail     double precision,
	r_hat        double precision,
	mcse_mean    double precision,
	mcse_sd      double precision,
	run_id       text,
	exported_at  timestamp with time zone,
	source_sha   text,
	source_dirty boolean
)
;

alter table analytics."09_diagnostics_v2"
	owner to postgres
;