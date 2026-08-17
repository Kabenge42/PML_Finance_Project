create function public.calc_unusual_items_features(p_isin text default NULL::text)
	returns table("isin" text, "other_unusual_items_ltm" numeric, "impairment_goodwill_ltm" numeric, "asset_writedown_ltm" numeric, "restructuring_charges_ltm" numeric, "total_unusual_items" numeric, "unusual_items_to_revenue" numeric, "unusual_items_to_ebitda" numeric, "has_unusual_items_flag" integer, "earnings_quality_impact" numeric)
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

alter function public.calc_unusual_items_features(text) owner to postgres
;