CREATE FUNCTION beat_counts(surprises numeric[])
	RETURNS table("n_total" integer, "n_beats" integer)
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION beat_counts(numeric[]) OWNER TO postgres;

CREATE FUNCTION beat_counts(surprises double precision[])
	RETURNS table("n_total" integer, "n_beats" integer)
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION beat_counts(double precision[]) OWNER TO postgres;