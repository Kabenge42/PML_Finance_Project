create function public.calc_composite_scores(p_isin text default NULL::text)
	returns table("isin" text, "piotroski_f_score" integer, "dilution_score" numeric, "quality_momentum_score" numeric)
	stable
	parallel safe
	language plpgsql
as
$$
begin
	-- missing source code
end;
$$
;

alter function public.calc_composite_scores(text) owner to postgres
;