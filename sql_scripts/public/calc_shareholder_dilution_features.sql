CREATE FUNCTION public.calc_shareholder_dilution_features(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"           text,
		        "dilution_score" numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_shareholder_dilution_features(text) OWNER TO postgres;