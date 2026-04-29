create function refresh_all_stock_features() returns void
    language plpgsql
as
$$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_all_stock_features;
    RAISE NOTICE 'mv_all_stock_features refreshed at %', NOW();
END;
$$;

comment on function refresh_all_stock_features() is 'Refreshes the mv_all_stock_features materialized view concurrently (non-blocking).
    Call periodically after equities table updates.';

alter function refresh_all_stock_features() owner to postgres;

