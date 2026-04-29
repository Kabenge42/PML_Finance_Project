create table information_schema.equities_schema_metadata
(
    column_name  text,
    column_alias text,
    role         text,
    column_count integer,
    description  text,
    updated_at   timestamp,
    column_type  text
);

alter table information_schema.equities_schema_metadata
    owner to postgres;

