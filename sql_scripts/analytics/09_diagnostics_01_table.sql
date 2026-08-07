CREATE TABLE analytics."09_diagnostics_01_table"
(
	index     text,
	mean      double precision,
	sd        double precision,
	eti89_lb  double precision,
	eti89_ub  double precision,
	ess_bulk  double precision,
	ess_tail  double precision,
	r_hat     double precision,
	mcse_mean double precision,
	mcse_sd   double precision
);

ALTER TABLE analytics."09_diagnostics_01_table"
	OWNER TO postgres;