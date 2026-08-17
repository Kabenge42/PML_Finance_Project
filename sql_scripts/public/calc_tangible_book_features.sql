create function public.calc_tangible_book_features(p_isin text default NULL::text)
	returns table("isin" text, "tangible_book_value_fy" numeric, "tangible_book_value_ltm" numeric, "tangible_book_per_share" numeric, "price_to_tangible_book" numeric, "tangible_equity_ratio" numeric, "intangibles_to_equity" numeric, "goodwill_to_equity" numeric, "tangible_asset_quality" numeric, "tbv_yoy_growth" numeric, "tbv_vs_calculated" numeric)
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

alter function public.calc_tangible_book_features(text) owner to postgres
;