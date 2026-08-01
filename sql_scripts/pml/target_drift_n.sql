CREATE FUNCTION target_drift_n(arr numeric[]) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT COUNT(*)::INT
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1)
  AND arr[i] IS NOT NULL
  AND arr[i + 1] IS NOT NULL
  AND arr[i + 1] <> 0;
$$;

ALTER FUNCTION target_drift_n(numeric[]) OWNER TO postgres;

CREATE FUNCTION target_drift_n(arr double precision[]) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT COUNT(*)::INT
FROM generate_subscripts(arr, 1) AS i
WHERE i < array_length(arr, 1)
  AND arr[i] IS NOT NULL
  AND arr[i + 1] IS NOT NULL
  AND arr[i + 1] <> 0;
$$;

ALTER FUNCTION target_drift_n(double precision[]) OWNER TO postgres;