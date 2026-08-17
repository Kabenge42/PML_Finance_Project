create function public.calc_quality_features(p_isin text default NULL::text)
	returns table("isin" text, "has_goodwill_impairment" integer, "has_asset_writedown" integer, "has_restructuring" integer, "goodwill_to_assets_pct" numeric, "intangible_intensity" numeric, "exceptional_items_to_ebitda" numeric, "altman_z_score" numeric, "altman_z_trend" numeric, "current_ratio" numeric, "quick_ratio" numeric)
	stable
	parallel safe
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function public.calc_quality_features(text) owner to postgres
;