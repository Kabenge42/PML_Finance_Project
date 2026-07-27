CREATE PROCEDURE refresh_pymc_materialized_views(use_concurrently boolean default true, assert_coverage boolean default false)
	LANGUAGE plpgsql AS
$$
BEGIN
	-- missing source code
END;
$$;

ALTER PROCEDURE refresh_pymc_materialized_views(boolean, boolean) OWNER TO postgres;