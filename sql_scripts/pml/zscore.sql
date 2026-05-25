-- Cyclic dependencies found

CREATE FUNCTION pml.zscore(val double precision, mu double precision, sigma double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (val - mu) / NULLIF(sigma, 0);
$$;

ALTER FUNCTION pml.zscore(unknown, unknown, unknown) OWNER TO postgres;

CREATE FUNCTION pml.zscore(val numeric, mu numeric, sigma numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (val - mu) / NULLIF(sigma, 0);
$$;

ALTER FUNCTION pml.zscore(unknown, unknown, unknown) OWNER TO postgres;