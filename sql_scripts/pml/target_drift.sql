CREATE FUNCTION target_drift(arr numeric[]) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION target_drift(numeric[]) OWNER TO postgres;

CREATE FUNCTION target_drift(arr double precision[]) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION target_drift(double precision[]) OWNER TO postgres;

CREATE FUNCTION target_drift(arr numeric[], min_points integer) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION target_drift(numeric[], integer) OWNER TO postgres;

CREATE FUNCTION target_drift(arr double precision[], min_points integer) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION target_drift(double precision[], integer) OWNER TO postgres;