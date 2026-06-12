CREATE TABLE public.feature_registry_metadata
(
	function_name     varchar(128) NOT NULL PRIMARY KEY,
	category          varchar(64)  NOT NULL,
	feature_count     smallint,
	description       text,
	python_equivalent varchar(128),
	updated_at        timestamp DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE public.feature_registry_metadata
	OWNER TO postgres;

CREATE INDEX idx_feature_registry_category ON public.feature_registry_metadata (category);

CREATE INDEX idx_feature_registry_python_equiv ON public.feature_registry_metadata (python_equivalent);

CREATE UNIQUE INDEX idx_uq_feature_registry_function_name ON public.feature_registry_metadata (function_name);