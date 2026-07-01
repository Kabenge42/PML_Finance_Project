CREATE FUNCTION target_drift(arr numeric[]) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT AVG(pml.calc_change_ratio(arr[i], arr[i + 1]))
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1);
$$;

ALTER FUNCTION target_drift(numeric[]) OWNER TO postgres;

CREATE FUNCTION target_drift(arr double precision[]) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT AVG(pml.calc_change_ratio(arr[i], arr[i + 1]))
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1);
$$;

ALTER FUNCTION target_drift(double precision[]) OWNER TO postgres;

CREATE FUNCTION target_drift(arr numeric[], min_points integer) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN pml.target_drift_n(arr) >= min_points THEN pml.target_drift(arr) END;
$$;

ALTER FUNCTION target_drift(numeric[], integer) OWNER TO postgres;

CREATE FUNCTION target_drift(arr double precision[], min_points integer) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT CASE WHEN pml.target_drift_n(arr) >= min_points THEN pml.target_drift(arr) END;
$$;

ALTER FUNCTION target_drift(double precision[], integer) OWNER TO postgres;