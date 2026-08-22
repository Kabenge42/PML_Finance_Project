create procedure pml.refresh_kalman_pt_v2(IN use_concurrently boolean DEFAULT true, IN refresh_parent boolean DEFAULT true)
	language plpgsql
as
$$
BEGIN
	IF refresh_parent THEN
		IF use_concurrently THEN
			REFRESH MATERIALIZED VIEW CONCURRENTLY pml.mv_pymc_kalman_pt;
			ELSE
				REFRESH MATERIALIZED VIEW pml.mv_pymc_kalman_pt;
		END IF;
		RAISE NOTICE 'refreshed pml.mv_pymc_kalman_pt';
	END IF;

	IF use_concurrently THEN
		REFRESH MATERIALIZED VIEW CONCURRENTLY pml.mv_pymc_kalman_pt_v2;
		ELSE
			REFRESH MATERIALIZED VIEW pml.mv_pymc_kalman_pt_v2;
	END IF;
	RAISE NOTICE 'refreshed pml.mv_pymc_kalman_pt_v2';
END;
$$
;

comment on procedure pml.refresh_kalman_pt_v2(boolean, boolean) is 'Refresh the Kalman MV pair in dependency order (parent mv_pymc_kalman_pt, then child mv_pymc_kalman_pt_v2). Narrow alternative to refresh_pymc_materialized_views when iterating on the v2 model.'
;

alter procedure pml.refresh_kalman_pt_v2(boolean, boolean) owner to postgres
;