-- Cyclic dependencies found

CREATE FUNCTION beat_counts(surprises double precision[])
	RETURNS TABLE(n_total integer, n_beats integer)
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT SUM(CASE WHEN s IS NOT NULL THEN 1 ELSE 0 END)::INT           AS n_total,
       SUM(CASE WHEN s IS NOT NULL AND s > 0 THEN 1 ELSE 0 END)::INT AS n_beats
FROM UNNEST(surprises) AS s;
$$;

ALTER FUNCTION beat_counts(unknown) OWNER TO postgres;

CREATE FUNCTION beat_counts(surprises numeric[])
	RETURNS TABLE(n_total integer, n_beats integer)
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT SUM(CASE WHEN s IS NOT NULL THEN 1 ELSE 0 END)::INT           AS n_total,
       SUM(CASE WHEN s IS NOT NULL AND s > 0 THEN 1 ELSE 0 END)::INT AS n_beats
FROM UNNEST(surprises) AS s;
$$;

ALTER FUNCTION beat_counts(unknown) OWNER TO postgres;