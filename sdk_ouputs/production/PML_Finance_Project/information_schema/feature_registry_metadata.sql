create table information_schema.feature_registry_metadata
(
    function_name     varchar(128) not null
        primary key,
    category          varchar(64)  not null,
    feature_count     smallint,
    description       text,
    python_equivalent varchar(128),
    updated_at        timestamp default CURRENT_TIMESTAMP
);

alter table information_schema.feature_registry_metadata
    owner to postgres;

create index idx_feature_registry_category
    on information_schema.feature_registry_metadata (category);

create index idx_feature_registry_python_equiv
    on information_schema.feature_registry_metadata (python_equivalent);

create unique index idx_uq_feature_registry_function_name
    on information_schema.feature_registry_metadata (function_name);

