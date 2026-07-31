-- Cyclic dependencies found

CREATE FUNCTION calc_change_ratio(current_val double precision, previous_val double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0);
$$;

ALTER FUNCTION calc_change_ratio(unknown, unknown) OWNER TO postgres;

CREATE FUNCTION calc_change_ratio(current_val numeric, previous_val numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (current_val - previous_val) / NULLIF(previous_val, 0);
$$;

ALTER FUNCTION calc_change_ratio(unknown, unknown) OWNER TO postgres;