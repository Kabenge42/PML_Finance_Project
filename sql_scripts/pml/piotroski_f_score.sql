create function piotroski_f_score(roa numeric, roa_prev numeric, cfo numeric, ni numeric, ltde numeric, ltde_prev numeric, cr numeric, cr_prev numeric, shrs numeric, shrs_prev numeric, gpm numeric, gpm_prev numeric, at numeric, at_prev numeric) returns integer
	immutable
	parallel safe
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function piotroski_f_score(numeric, numeric, numeric, numeric, numeric, numeric, numeric, numeric, numeric, numeric, numeric, numeric, numeric, numeric) owner to postgres
;

create function piotroski_f_score(roa double precision, roa_prev double precision, cfo double precision, ni double precision, ltde double precision, ltde_prev double precision, cr double precision, cr_prev double precision, shrs double precision, shrs_prev double precision, gpm double precision, gpm_prev double precision, at double precision, at_prev double precision) returns integer
	immutable
	parallel safe
	language sql
as
$$
	begin
-- missing source code
end;
$$
;

alter function piotroski_f_score(double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision, double precision) owner to postgres
;