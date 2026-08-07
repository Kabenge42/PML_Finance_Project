CREATE FUNCTION text_to_numeric_safe(input_text unknown) RETURNS numeric
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION text_to_numeric_safe(unknown) OWNER TO postgres;