CREATE FUNCTION public.calc_total_assets_temporal(p_isin text default NULL::text)
	RETURNS table("isin" text, "assets_fq" numeric, "assets_fy" numeric, "assets_ltm" numeric, "assets_1fq" numeric, "assets_2fq" numeric, "assets_3fq" numeric, "assets_4fq" numeric, "assets_1fy" numeric, "assets_2fy" numeric, "assets_3fy" numeric, "assets_4fy" numeric, "assets_qoq_growth" numeric, "assets_yoy_growth" numeric, "assets_3y_cagr" numeric, "asset_growth_accel" numeric, "asset_base_stable" integer)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_total_assets_temporal(text) OWNER TO postgres;