create function calc_all_enhanced_features(p_isin text DEFAULT NULL::text)
    returns TABLE
            (
                isin           text,
                feature_count  integer,
                reference_date timestamp without time zone
            )
    stable
    parallel safe
    language sql
as
$$
SELECT "ISIN"            AS isin,
       (SELECT COUNT(*)::INTEGER as count
        FROM information_schema.routines
        WHERE routine_name LIKE 'calc_%'
          AND routine_schema = 'public')
                         AS feature_count,
       CURRENT_TIMESTAMP AS reference_date
FROM postgres.public.equities
WHERE p_isin IS NULL
   OR "ISIN" = p_isin;
$$;

alter function calc_all_enhanced_features(text) owner to postgres;

