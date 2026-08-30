create table analytics."09b_comparison_v2"
(
	arm             text,
	rank            bigint,
	elpd_diff       double precision,
	dse             double precision,
	p_worse         double precision,
	diag_diff       text,
	diag_elpd       text,
	p               double precision,
	elpd            double precision,
	se              double precision,
	weight          double precision,
	divergences     bigint,
	min_ess_bulk    double precision,
	min_ess_param   text,
	max_r_hat       double precision,
	max_r_hat_param text,
	n_group_levels  bigint,
	backend         text,
	run_id          text,
	exported_at     timestamp with time zone,
	source_sha      text,
	source_dirty    boolean
)
;

alter table analytics."09b_comparison_v2"
	owner to postgres
;