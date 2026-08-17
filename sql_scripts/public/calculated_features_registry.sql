create table public.calculated_features_registry
(
	feature_key        varchar(128) not null
		primary key,
	feature_alias      varchar(128) not null,
	category           varchar(64)  not null,
	source_function    varchar(128),
	description        text,
	source_columns     text[],
	primary_source_col text
		references public.equities_schema_metadata,
	calculation_type   varchar(32),
	data_type          varchar(32),
	updated_at         timestamp default CURRENT_TIMESTAMP
)
;

alter table public.calculated_features_registry
	owner to postgres
;

create index idx_calc_features_category
	on public.calculated_features_registry (category)
;

create index idx_calc_features_source_fn
	on public.calculated_features_registry (source_function)
;

create index idx_calc_features_primary_col
	on public.calculated_features_registry (primary_source_col)
;

create unique index idx_uq_calc_features_feature_key
	on public.calculated_features_registry (feature_key)
;