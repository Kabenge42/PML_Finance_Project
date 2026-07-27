CREATE FUNCTION target_drift_n(arr numeric[]) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION target_drift_n(numeric[]) OWNER TO postgres;

CREATE FUNCTION target_drift_n(arr double precision[]) RETURNS integer
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION target_drift_n(double precision[]) OWNER TO postgres;