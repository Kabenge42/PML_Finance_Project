create function public.calc_shareholder_dilution_features(p_isin text default NULL::text)
	returns table("isin" text, "dilution_score" numeric)
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

alter function public.calc_shareholder_dilution_features(text) owner to postgres
;