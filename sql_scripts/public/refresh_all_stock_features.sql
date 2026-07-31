CREATE FUNCTION public.refresh_all_stock_features() RETURNS void
	LANGUAGE plpgsql AS
$$
BEGIN
	REFRESH MATERIALIZED VIEW CONCURRENTLY mv_all_stock_features;
	RAISE NOTICE 'mv_all_stock_features refreshed at %', NOW();
END;
$$;

COMMENT ON FUNCTION public.refresh_all_stock_features() IS 'Refreshes the mv_all_stock_features materialized view concurrently (non-blocking).
    Call periodically after equities table updates.';

ALTER FUNCTION public.refresh_all_stock_features() OWNER TO postgres;