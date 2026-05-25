-- Cyclic dependencies found

CREATE FUNCTION pml.winsorise(val double precision, lo double precision, hi double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT GREATEST(lo, LEAST(hi, val));
$$;

ALTER FUNCTION pml.winsorise(unknown, unknown, unknown) OWNER TO postgres;

CREATE FUNCTION pml.winsorise(val numeric, lo numeric, hi numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT GREATEST(lo, LEAST(hi, val));
$$;

ALTER FUNCTION pml.winsorise(unknown, unknown, unknown) OWNER TO postgres;