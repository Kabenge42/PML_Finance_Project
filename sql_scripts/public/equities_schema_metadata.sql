CREATE TABLE public.equities_schema_metadata
(
	column_name  text                          NOT NULL PRIMARY KEY,
	column_alias text      DEFAULT 'n/a'::text NOT NULL,
	role         text                          NOT NULL,
	column_count integer,
	description  text,
	column_type  text,
	updated_at   timestamp DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE public.equities_schema_metadata IS 'Metadata table documenting all columns in the equities table with their roles, aliases, and DDL definitions';

ALTER TABLE public.equities_schema_metadata
	OWNER TO postgres;

CREATE INDEX idx_equities_schema_metadata_role ON public.equities_schema_metadata (role);

CREATE INDEX idx_equities_schema_metadata_ddl ON public.equities_schema_metadata (column_type);