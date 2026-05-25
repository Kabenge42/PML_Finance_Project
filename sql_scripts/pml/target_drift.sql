-- Cyclic dependencies found

CREATE FUNCTION pml.target_drift(arr double precision[]) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT AVG(pml.calc_change_ratio(arr[i], arr[i + 1]))
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1);
$$;

ALTER FUNCTION pml.target_drift(unknown) OWNER TO postgres;

CREATE FUNCTION pml.target_drift(arr numeric[]) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT AVG(pml.calc_change_ratio(arr[i], arr[i + 1]))
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1);
$$;

ALTER FUNCTION pml.target_drift(unknown) OWNER TO postgres;