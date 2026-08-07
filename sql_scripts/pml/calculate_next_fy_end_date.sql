CREATE FUNCTION calculate_next_fy_end_date(fy_end_date unknown) RETURNS date
	IMMUTABLE STRICT PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION calculate_next_fy_end_date(unknown) OWNER TO postgres;