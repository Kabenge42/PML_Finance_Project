CREATE FUNCTION public.calc_gaap_revision_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                         text,
		        "gaap_revision_momentum"       numeric,
		        "gaap_revision_1m"             numeric,
		        "gaap_revision_3m"             numeric,
		        "gaap_revision_6m"             numeric,
		        "gaap_revision_1y"             numeric,
		        "gaap_vs_norm_revision_spread" numeric,
		        "gaap_revision_acceleration"   numeric,
		        "gaap_positive_revision_flag"  integer,
		        "revision_quality_divergence"  numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_gaap_revision_features(text) OWNER TO postgres;