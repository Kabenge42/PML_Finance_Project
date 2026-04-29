create table analytics.prob_valuation_ratios
(
    category        text,
    feature         text,
    p_distress_high double precision,
    p_distress_low  double precision,
    lift_high       double precision,
    lift_low        double precision,
    separation      double precision
);

alter table analytics.prob_valuation_ratios
    owner to postgres;

