CREATE PROCEDURE refresh_pymc_materialized_views(IN use_concurrently boolean DEFAULT TRUE)
	LANGUAGE plpgsql AS
$$
DECLARE
	mv          TEXT;
	schema_part TEXT;
	table_part  TEXT;
	mvs         TEXT[] := ARRAY [ 'pml.mv_pymc_earnings_beat', 'pml.mv_pymc_price_target', 'pml.mv_pymc_kalman_pt', 'pml.mv_pymc_dcf_pt', 'pml.mv_pymc_dividend_safety', 'pml.mv_pymc_credit_risk', 'pml.mv_pymc_accounting_anomaly' ];
BEGIN
	FOREACH mv IN ARRAY mvs
		LOOP
			-- Split "schema.table" into its two identifier parts so %I quotes correctly
			schema_part := split_part(mv, '.', 1);
			table_part := split_part(mv, '.', 2);

			IF use_concurrently THEN
				EXECUTE format('REFRESH MATERIALIZED VIEW CONCURRENTLY %I.%I', schema_part, table_part);
				ELSE
					EXECUTE format('REFRESH MATERIALIZED VIEW %I.%I', schema_part, table_part);
			END IF;
			END LOOP;
END;
$$;

ALTER PROCEDURE refresh_pymc_materialized_views(boolean) OWNER TO postgres;