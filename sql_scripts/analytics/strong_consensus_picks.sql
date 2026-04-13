create table analytics.strong_consensus_picks
(
    isin                       text,
    ticker                     text,
    name                       text,
    region                     text,
    country                    text,
    exchange                   text,
    sector                     text,
    industry                   text,
    prob_positive_upside       double precision,
    price_target_mc            double precision,
    expected_upside_mc         double precision,
    risk_reward_ratio          double precision,
    implied_return_mc          double precision,
    var_5_pct                  double precision,
    implied_return_kalman      double precision,
    expected_upside_kalman     double precision,
    price_target_kalman        double precision,
    kalman_estimate            double precision,
    kalman_variance            double precision,
    implied_return_pt          double precision,
    achievement_probability    double precision,
    price_target_prob_weighted double precision,
    confidence_level           text,
    analyst_conviction         double precision,
    eps_revision_momentum      double precision,
    analyst_rating_normalized  double precision,
    mc_bullish                 boolean,
    kal_bullish                boolean,
    pt_bullish                 boolean,
    agreement_score            bigint,
    signal                     text
);

alter table analytics.strong_consensus_picks
    owner to postgres;

