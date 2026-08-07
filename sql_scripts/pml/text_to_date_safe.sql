CREATE FUNCTION text_to_date_safe(input_text unknown, date_format unknown default 'AUTO'::text) RETURNS date
	IMMUTABLE STRICT
	LANGUAGE plpgsql AS
$$
BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION text_to_date_safe(unknown, unknown) OWNER TO postgres;