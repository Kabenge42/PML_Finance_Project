create function public.calc_all_enhanced_features(p_isin text default NULL::text)
	returns table("isin" text, "feature_count" integer, "reference_date" timestamp)
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

alter function public.calc_all_enhanced_features(text) owner to postgres
;