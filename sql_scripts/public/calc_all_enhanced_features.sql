CREATE FUNCTION public.calc_all_enhanced_features(p_isin text DEFAULT NULL::text)
	RETURNS TABLE(isin text, feature_count integer, reference_date timestamp without time zone)
	STABLE PARALLEL SAFE
	LANGUAGE sql
AS
$$
SELECT "ISIN"            AS isin,
       (SELECT COUNT(*)::INTEGER AS count
        FROM information_schema.routines
        WHERE routine_name LIKE 'calc_%'
	      AND routine_schema = 'public'
       )                 AS feature_count,
       CURRENT_TIMESTAMP AS reference_date
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

ALTER FUNCTION public.calc_all_enhanced_features(unknown) OWNER TO postgres;