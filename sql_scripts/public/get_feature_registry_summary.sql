CREATE FUNCTION public.get_feature_registry_summary()
	RETURNS TABLE(category text, function_count integer, total_features integer)
	STABLE
	LANGUAGE sql
AS
$$
SELECT category, COUNT(*)::INTEGER AS function_count, SUM(COALESCE(feature_count, 0))::INTEGER AS total_features
FROM feature_registry_metadata
GROUP BY category
ORDER BY total_features DESC;
$$;

ALTER FUNCTION public.get_feature_registry_summary() OWNER TO postgres;