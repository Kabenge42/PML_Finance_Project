CREATE FUNCTION public.calc_quality_momentum_composite(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"                   text,
		        "quality_momentum_score" numeric
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_quality_momentum_composite(text) OWNER TO postgres;