create function pml.assert_pymc_catalogue_coverage() returns void
	language plpgsql
as
$$
DECLARE
	v_count      INT;
	v_violations TEXT;
BEGIN
	SELECT COUNT(*),
	       string_agg(format('%s.%s [%s]', model_target, feat_name, status), ', ' ORDER BY model_target, feat_name)
	INTO v_count, v_violations
	FROM pml.vw_pymc_catalogue_coverage_check
	WHERE status <> 'OK';

	IF v_count > 0 THEN
		RAISE EXCEPTION 'PyMC catalogue coverage check failed for % column(s): %', v_count, v_violations USING HINT =
				'Every feat_/observed_/n_ column emitted by each mv_pymc_* must have exactly one pml.vw_pymc_feature_catalogue row with a matching feature_alias for its model_target (see pml.vw_pymc_catalogue_coverage_check).';
	END IF;

	-- The trail-days map is the same class of contract -- a Python-visible
	-- surface that must agree with what the MV emits -- so it is asserted from
	-- the same entry point rather than needing its own call site.
	PERFORM pml.assert_pymc_trail_days_map();
END;
$$
;

alter function pml.assert_pymc_catalogue_coverage() owner to postgres
;