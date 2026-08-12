-- Cyclic dependencies found

create function altman_zone(z double precision) returns integer
	immutable
	parallel safe
	language sql
as
$$
SELECT CASE WHEN z IS NULL THEN NULL WHEN z < 1.81 THEN 1 WHEN z < 2.99 THEN 2 ELSE 3 END;
$$
;

alter function altman_zone(unknown) owner to postgres
;

create function altman_zone(z numeric) returns integer
	immutable
	parallel safe
	language sql
as
$$
SELECT CASE WHEN z IS NULL THEN NULL WHEN z < 1.81 THEN 1 WHEN z < 2.99 THEN 2 ELSE 3 END;
$$
;

alter function altman_zone(unknown) owner to postgres
;