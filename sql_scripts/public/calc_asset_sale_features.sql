CREATE FUNCTION public.calc_asset_sale_features(p_isin text default NULL::text)
	RETURNS table("isin" text, "gain_loss_on_sale_of_assets_ltm" numeric, "asset_sale_frequency" integer, "asset_sale_trend" numeric)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_asset_sale_features(text) OWNER TO postgres;