-- Cyclic dependencies found

CREATE FUNCTION pml.accruals_ratio(ni double precision, cfo double precision, scale double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT pml.safe_divide(ni - cfo, NULLIF(scale, 0));
$$;

ALTER FUNCTION pml.accruals_ratio(unknown, unknown, unknown) OWNER TO postgres;

CREATE FUNCTION pml.accruals_ratio(ni numeric, cfo numeric, scale numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT pml.safe_divide(ni - cfo, NULLIF(scale, 0));
$$;

ALTER FUNCTION pml.accruals_ratio(unknown, unknown, unknown) OWNER TO postgres;