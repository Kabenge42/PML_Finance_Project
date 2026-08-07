CREATE FUNCTION currency_name(code_text unknown) RETURNS text
	STABLE
	LANGUAGE sql AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION currency_name(unknown) OWNER TO postgres;