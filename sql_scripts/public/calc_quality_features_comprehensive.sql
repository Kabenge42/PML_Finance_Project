CREATE FUNCTION public.calc_quality_features_comprehensive(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                          text,
		        "goodwill_impairment_ltm"       numeric,
		        "asset_writedown_ltm"           numeric,
		        "restructuring_ltm"             numeric,
		        "has_goodwill_impairment_ltm"   integer,
		        "goodwill_impairment_frequency" integer,
		        "asset_writedown_frequency"     integer,
		        "restructuring_frequency"       integer,
		        "exceptional_items_total_ltm"   numeric,
		        "exceptional_items_to_ebitda"   numeric,
		        "quality_issues_count_5y"       integer,
		        "accounting_quality_score"      numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_quality_features_comprehensive(text) OWNER TO postgres;