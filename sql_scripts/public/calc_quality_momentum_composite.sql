create function public.calc_quality_momentum_composite(p_isin text default NULL::text)
	returns table("isin" text, "quality_momentum_score" numeric)
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

alter function public.calc_quality_momentum_composite(text) owner to postgres
;