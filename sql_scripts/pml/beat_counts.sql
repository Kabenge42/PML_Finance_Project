-- Cyclic dependencies found

create function beat_counts(surprises double precision[])
	returns TABLE(n_total integer, n_beats integer)
	immutable
	parallel safe
	language sql
as
$$
SELECT SUM(CASE WHEN s IS NOT NULL THEN 1 ELSE 0 END)::INT           AS n_total,
       SUM(CASE WHEN s IS NOT NULL AND s > 0 THEN 1 ELSE 0 END)::INT AS n_beats
FROM UNNEST(surprises) AS s;
$$
;

alter function beat_counts(unknown) owner to postgres
;

create function beat_counts(surprises numeric[])
	returns TABLE(n_total integer, n_beats integer)
	immutable
	parallel safe
	language sql
as
$$
SELECT SUM(CASE WHEN s IS NOT NULL THEN 1 ELSE 0 END)::INT           AS n_total,
       SUM(CASE WHEN s IS NOT NULL AND s > 0 THEN 1 ELSE 0 END)::INT AS n_beats
FROM UNNEST(surprises) AS s;
$$
;

alter function beat_counts(unknown) owner to postgres
;