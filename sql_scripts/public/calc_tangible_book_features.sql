CREATE FUNCTION public.calc_tangible_book_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                    text,
		        "tangible_book_value_fy"  numeric,
		        "tangible_book_value_ltm" numeric,
		        "tangible_book_per_share" numeric,
		        "price_to_tangible_book"  numeric,
		        "tangible_equity_ratio"   numeric,
		        "intangibles_to_equity"   numeric,
		        "goodwill_to_equity"      numeric,
		        "tangible_asset_quality"  numeric,
		        "tbv_yoy_growth"          numeric,
		        "tbv_vs_calculated"       numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_tangible_book_features(text) OWNER TO postgres;