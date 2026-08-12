-- Cyclic dependencies found

create function piotroski_f_score(roa double precision, roa_prev double precision, cfo double precision, ni double precision, ltde double precision, ltde_prev double precision, cr double precision, cr_prev double precision, shrs double precision, shrs_prev double precision, gpm double precision, gpm_prev double precision, at double precision, at_prev double precision) returns integer
	immutable
	parallel safe
	language sql
as
$$
SELECT (CASE WHEN roa > 0 THEN 1 ELSE 0 END +
        CASE WHEN cfo > 0 THEN 1 ELSE 0 END +
        CASE WHEN roa > roa_prev THEN 1 ELSE 0 END +
        CASE WHEN cfo > ni THEN 1 ELSE 0 END +
        CASE WHEN ltde < ltde_prev THEN 1 ELSE 0 END +
        CASE WHEN cr > cr_prev THEN 1 ELSE 0 END +
        CASE WHEN shrs <= shrs_prev THEN 1 ELSE 0 END +
        CASE WHEN gpm > gpm_prev THEN 1 ELSE 0 END +
        CASE WHEN at > at_prev THEN 1 ELSE 0 END)::INTEGER;
$$
;

alter function piotroski_f_score(unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown) owner to postgres
;

create function piotroski_f_score(roa numeric, roa_prev numeric, cfo numeric, ni numeric, ltde numeric, ltde_prev numeric, cr numeric, cr_prev numeric, shrs numeric, shrs_prev numeric, gpm numeric, gpm_prev numeric, at numeric, at_prev numeric) returns integer
	immutable
	parallel safe
	language sql
as
$$
SELECT (CASE WHEN roa > 0 THEN 1 ELSE 0 END +
        CASE WHEN cfo > 0 THEN 1 ELSE 0 END +
        CASE WHEN roa > roa_prev THEN 1 ELSE 0 END +
        CASE WHEN cfo > ni THEN 1 ELSE 0 END +
        CASE WHEN ltde < ltde_prev THEN 1 ELSE 0 END +
        CASE WHEN cr > cr_prev THEN 1 ELSE 0 END +
        CASE WHEN shrs <= shrs_prev THEN 1 ELSE 0 END +
        CASE WHEN gpm > gpm_prev THEN 1 ELSE 0 END +
        CASE WHEN at > at_prev THEN 1 ELSE 0 END)::INTEGER;
$$
;

alter function piotroski_f_score(unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown, unknown) owner to postgres
;