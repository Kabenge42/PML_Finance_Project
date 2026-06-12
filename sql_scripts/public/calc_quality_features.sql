CREATE FUNCTION public.calc_quality_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                        text,
		        "has_goodwill_impairment"     integer,
		        "has_asset_writedown"         integer,
		        "has_restructuring"           integer,
		        "goodwill_to_assets_pct"      numeric,
		        "intangible_intensity"        numeric,
		        "exceptional_items_to_ebitda" numeric,
		        "altman_z_score"              numeric,
		        "altman_z_trend"              numeric,
		        "current_ratio"               numeric,
		        "quick_ratio"                 numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_quality_features(text) OWNER TO postgres;