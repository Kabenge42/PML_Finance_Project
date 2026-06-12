CREATE TABLE public.feature_catalogue
(
	category         varchar(64),
	feature_alias    varchar(128),
	source_function  varchar(128),
	calculation_type varchar(32),
	data_type        varchar(32)
);

ALTER TABLE public.feature_catalogue
	OWNER TO postgres;