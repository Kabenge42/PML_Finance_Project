create table public.equities_schema_metadata
(
	column_name  text                          not null
		primary key,
	column_alias text      default 'n/a'::text not null,
	role         text                          not null,
	column_count integer,
	description  text,
	column_type  text,
	updated_at   timestamp default CURRENT_TIMESTAMP
)
;

comment on table public.equities_schema_metadata is 'Metadata table documenting all columns in the equities table with their roles, aliases, and DDL definitions'
;

alter table public.equities_schema_metadata
	owner to postgres
;

create index idx_equities_schema_metadata_role
	on public.equities_schema_metadata (role)
;

create index idx_equities_schema_metadata_ddl
	on public.equities_schema_metadata (column_type)
;