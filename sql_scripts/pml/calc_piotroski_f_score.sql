create function calc_piotroski_f_score(p_isin text default NULL::text)
	returns table("isin" text, "piotroski_f_score" integer)
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

alter function calc_piotroski_f_score(text) owner to postgres
;