CREATE FUNCTION public.calc_piotroski_f_score(p_isin text DEFAULT NULL::text)
	RETURNS table
	        (
		        "isin"              text,
		        "piotroski_f_score" integer
	        )
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$ BEGIN
	-- missing source code
END;
$$;

ALTER FUNCTION public.calc_piotroski_f_score(text) OWNER TO postgres;