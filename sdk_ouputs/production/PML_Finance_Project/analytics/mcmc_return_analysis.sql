create table analytics.mcmc_return_analysis
(
    level          text,
    "group"        text,
    posterior_mean double precision,
    posterior_std  double precision,
    r_hat          double precision,
    converged      boolean,
    ess_bulk       double precision,
    ess_tail       double precision,
    ci_95_lower    double precision,
    ci_95_upper    double precision,
    student_t_mu   double precision,
    student_t_df   double precision,
    n_obs          double precision,
    raw_mean       double precision,
    shrinkage      double precision,
    prob_positive  double precision
);

alter table analytics.mcmc_return_analysis
    owner to postgres;

