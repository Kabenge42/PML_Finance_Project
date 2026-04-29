create table equities_schema_metadata
(
    column_name  text                          not null
        primary key,
    column_alias text      default 'n/a'::text not null,
    role         text                          not null,
    column_count integer,
    description  text,
    updated_at   timestamp default CURRENT_TIMESTAMP,
    column_type  text
);

comment on table equities_schema_metadata is 'Metadata table documenting all columns in the equities table with their roles, aliases, and DDL definitions';

alter table equities_schema_metadata
    owner to postgres;

create index idx_equities_schema_metadata_role
    on equities_schema_metadata (role);

create index idx_equities_schema_metadata_ddl
    on equities_schema_metadata (column_type);

