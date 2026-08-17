create function public.calc_asset_sale_features(p_isin text default NULL::text)
	returns table("isin" text, "gain_loss_on_sale_of_assets_ltm" numeric, "asset_sale_frequency" integer, "asset_sale_trend" numeric)
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

alter function public.calc_asset_sale_features(text) owner to postgres
;