create table analytics.kalman_filtered_price_targets
(
    isin                   text,
    ticker                 text,
    name                   text,
    country                text,
    exchange               text,
    sector                 text,
    industry               text,
    implied_return_kalman  double precision,
    expected_upside_kalman double precision,
    price_target_kalman    double precision,
    kalman_estimate        double precision,
    kalman_variance        double precision,
    kalman_gain            double precision,
    signal_strength        double precision,
    original_price         double precision,
    original_target        double precision
);

alter table analytics.kalman_filtered_price_targets
    owner to postgres;

