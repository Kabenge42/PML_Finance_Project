CREATE TABLE public.calculated_features_registry
(
	feature_key        varchar(128) NOT NULL PRIMARY KEY,
	feature_alias      varchar(128) NOT NULL,
	category           varchar(64)  NOT NULL,
	source_function    varchar(128),
	description        text,
	source_columns     text[],
	primary_source_col text REFERENCES public.equities_schema_metadata,
	calculation_type   varchar(32),
	data_type          varchar(32),
	updated_at         timestamp DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE public.calculated_features_registry
	OWNER TO postgres;

CREATE INDEX idx_calc_features_category ON public.calculated_features_registry (category);

CREATE INDEX idx_calc_features_source_fn ON public.calculated_features_registry (source_function);

CREATE INDEX idx_calc_features_primary_col ON public.calculated_features_registry (primary_source_col);

CREATE UNIQUE INDEX idx_uq_calc_features_feature_key ON public.calculated_features_registry (feature_key);