-- Cyclic dependencies found

CREATE FUNCTION safe_logit(p double precision, eps double precision DEFAULT 0.000001) RETURNS double precision
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT LN(GREATEST(eps, LEAST(1 - eps, p)) / (1 - GREATEST(eps, LEAST(1 - eps, p))));
$$;

ALTER FUNCTION safe_logit(unknown, unknown) OWNER TO postgres;

CREATE FUNCTION safe_logit(p numeric, eps numeric DEFAULT 0.000001) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT LN(GREATEST(eps, LEAST(1 - eps, p)) / (1 - GREATEST(eps, LEAST(1 - eps, p))));
$$;

ALTER FUNCTION safe_logit(unknown, unknown) OWNER TO postgres;