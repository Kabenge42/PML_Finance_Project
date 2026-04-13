create table analytics.monte_carlo_simulation
(
    isin                 text,
    ticker               text,
    name                 text,
    region               text,
    country              text,
    exchange             text,
    sector               text,
    industry             text,
    last_price           double precision,
    expected_upside_mc   double precision,
    implied_return_mc    double precision,
    price_target_mc      double precision,
    pt_median            double precision,
    pt_spread            double precision,
    upside_std           double precision,
    var_5_pct            double precision,
    prob_positive_upside double precision,
    risk_reward_ratio    double precision
);

alter table analytics.monte_carlo_simulation
    owner to postgres;

