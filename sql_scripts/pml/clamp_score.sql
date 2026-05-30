CREATE FUNCTION clamp_score(val numeric, min_val numeric DEFAULT 0, max_val numeric DEFAULT 100) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT GREATEST(min_val, LEAST(max_val, val)) AS result;
$$;

ALTER FUNCTION clamp_score(numeric, numeric, numeric) OWNER TO postgres;

CREATE FUNCTION clamp_score(val     double precision, min_val double precision DEFAULT 0,
                            max_val double precision DEFAULT 100) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT GREATEST(min_val, LEAST(max_val, val));
$$;

ALTER FUNCTION clamp_score(double precision, double precision, double precision) OWNER TO postgres;