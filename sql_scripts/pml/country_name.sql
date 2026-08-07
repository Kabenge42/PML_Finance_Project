CREATE FUNCTION country_name(code_text unknown) RETURNS text
	STABLE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION country_name(unknown) OWNER TO postgres;