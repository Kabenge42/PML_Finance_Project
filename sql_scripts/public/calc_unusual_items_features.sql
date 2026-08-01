CREATE FUNCTION public.calc_unusual_items_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "other_unusual_items_ltm" numeric, "impairment_goodwill_ltm" numeric, "asset_writedown_ltm" numeric, "restructuring_charges_ltm" numeric, "total_unusual_items" numeric, "unusual_items_to_revenue" numeric, "unusual_items_to_ebitda" numeric, "has_unusual_items_flag" integer, "earnings_quality_impact" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_unusual_items_features(text) OWNER TO postgres;