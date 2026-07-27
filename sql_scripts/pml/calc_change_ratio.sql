CREATE FUNCTION calc_change_ratio(current_val numeric, previous_val numeric) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION calc_change_ratio(numeric, numeric) OWNER TO postgres;

CREATE FUNCTION calc_change_ratio(current_val double precision, previous_val double precision) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION calc_change_ratio(double precision, double precision) OWNER TO postgres;