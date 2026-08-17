create function public.get_feature_registry_summary()
	returns table("category" text, "function_count" integer, "total_features" integer)
	stable
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function public.get_feature_registry_summary() owner to postgres
;