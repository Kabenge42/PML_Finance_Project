CREATE FUNCTION assert_pymc_catalogue_coverage() RETURNS void
	LANGUAGE plpgsql AS
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
END;
$$;

ALTER FUNCTION assert_pymc_catalogue_coverage() OWNER TO postgres;