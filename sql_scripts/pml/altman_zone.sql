-- Cyclic dependencies found

CREATE FUNCTION pml.altman_zone(z double precision) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN z IS NULL THEN NULL WHEN z < 1.81 THEN 1 WHEN z < 2.99 THEN 2 ELSE 3 END;
$$;

ALTER FUNCTION pml.altman_zone(unknown) OWNER TO postgres;

CREATE FUNCTION pml.altman_zone(z numeric) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN z IS NULL THEN NULL WHEN z < 1.81 THEN 1 WHEN z < 2.99 THEN 2 ELSE 3 END;
$$;

ALTER FUNCTION pml.altman_zone(unknown) OWNER TO postgres;