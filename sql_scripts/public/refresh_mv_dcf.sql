CREATE FUNCTION public.refresh_mv_dcf() RETURNS void
	LANGUAGE plpgsql AS
$$
BEGIN
	-- missing source code
END;
$$;

COMMENT ON FUNCTION public.refresh_mv_dcf() IS 'Refreshes the mv_dcf materialized view concurrently (non-blocking).
    Call periodically after equities table updates.';

ALTER FUNCTION public.refresh_mv_dcf() OWNER TO postgres;