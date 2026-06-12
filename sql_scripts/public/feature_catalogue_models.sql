CREATE TABLE public.feature_catalogue_models
(
	feature_alias text NOT NULL,
	model_name    text NOT NULL
);

ALTER TABLE public.feature_catalogue_models
	OWNER TO postgres;

CREATE INDEX ix_fcm_model_name ON public.feature_catalogue_models (model_name);