create procedure pml.refresh_pymc_materialized_views(IN use_concurrently boolean DEFAULT true, IN assert_coverage boolean DEFAULT false)
	language plpgsql
as
$$
DECLARE
	mv          TEXT;
	schema_part TEXT;
	table_part  TEXT;
	-- ORDER IS LOAD-BEARING, not alphabetical or historical: FOREACH walks the
	-- array in sequence, and mv_pymc_kalman_pt_v2 SELECTs from
	-- mv_pymc_kalman_pt. Refreshing the child first rebuilds it against a stale
	-- parent -- the mixed-vintage failure the analytics export already learned
	-- the hard way. Keep v2 immediately after v1.
	mvs         TEXT[] := ARRAY [ 'pml.mv_pymc_earnings_beat', 'pml.mv_pymc_price_target', 'pml.mv_pymc_kalman_pt', 'pml.mv_pymc_kalman_pt_v2', 'pml.mv_pymc_dcf_pt', 'pml.mv_pymc_dividend_safety', 'pml.mv_pymc_credit_risk', 'pml.mv_pymc_accounting_anomaly' ];
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

	-- Fail loudly if the MV feature surface and the catalogue have diverged.
	IF assert_coverage THEN PERFORM pml.assert_pymc_catalogue_coverage(); END IF;
END;
$$
;

alter procedure pml.refresh_pymc_materialized_views(boolean, boolean) owner to postgres
;