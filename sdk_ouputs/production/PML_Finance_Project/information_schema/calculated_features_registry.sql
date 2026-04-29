create table information_schema.calculated_features_registry
(
    feature_key        varchar(128),
    feature_alias      varchar(128),
    category           varchar(64),
    source_function    varchar(128),
    description        text,
    source_columns     text[],
    primary_source_col text,
    calculation_type   varchar(32),
    data_type          varchar(32),
    updated_at         timestamp
);

alter table information_schema.calculated_features_registry
    owner to postgres;

create index idx_calc_features_category
    on information_schema.calculated_features_registry (category);

create index idx_calc_features_source_fn
    on information_schema.calculated_features_registry (source_function);

create index idx_calc_features_primary_col
    on information_schema.calculated_features_registry (primary_source_col);

create unique index idx_uq_calc_features_feature_key
    on information_schema.calculated_features_registry (feature_key);

