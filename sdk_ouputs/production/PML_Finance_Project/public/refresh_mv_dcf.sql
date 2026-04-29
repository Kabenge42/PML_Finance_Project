create function refresh_mv_dcf() returns void
    language plpgsql
as
$$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dcf;
    RAISE NOTICE 'mv_dcf refreshed at %', NOW();
END;
$$;

comment on function refresh_mv_dcf() is 'Refreshes the mv_dcf materialized view concurrently (non-blocking).
    Call periodically after equities table updates.';

alter function refresh_mv_dcf() owner to postgres;

