CREATE FUNCTION pct_change(current_val numeric, previous_val numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) * 100 AS result;
$$;

ALTER FUNCTION pct_change(numeric, numeric) OWNER TO postgres;

CREATE FUNCTION pct_change(current_val double precision, previous_val double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0) * 100;
$$;

ALTER FUNCTION pct_change(double precision, double precision) OWNER TO postgres;