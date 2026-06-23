-- Cyclic dependencies found

CREATE FUNCTION winsorise(val double precision, lo double precision, hi double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT GREATEST(lo, LEAST(hi, val));
$$;

ALTER FUNCTION winsorise(unknown, unknown, unknown) OWNER TO postgres;

CREATE FUNCTION winsorise(val numeric, lo numeric, hi numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT GREATEST(lo, LEAST(hi, val));
$$;

ALTER FUNCTION winsorise(unknown, unknown, unknown) OWNER TO postgres;