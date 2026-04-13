create function get_feature_registry_summary()
    returns TABLE
            (
                category       text,
                function_count integer,
                total_features integer
            )
    stable
    language sql
as
$$
SELECT category,
       COUNT(*)::INTEGER                        AS function_count,
       SUM(COALESCE(feature_count, 0))::INTEGER AS total_features
FROM feature_registry_metadata
GROUP BY category
ORDER BY total_features DESC;
$$;

alter function get_feature_registry_summary() owner to postgres;

