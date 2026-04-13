create table analytics.probability_analytics_summary
(
    metric text,
    value  double precision
);

alter table analytics.probability_analytics_summary
    owner to postgres;

