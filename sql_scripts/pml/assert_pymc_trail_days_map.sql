create function pml.assert_pymc_trail_days_map() returns void
	language plpgsql
as
$$
DECLARE
	v_missing TEXT;
	v_extra   TEXT;
BEGIN
	WITH mv_cols AS (SELECT a.attname::TEXT AS col
	                 FROM pg_attribute a
		                      JOIN pg_class c ON c.oid = a.attrelid
		                      JOIN pg_namespace n ON n.oid = c.relnamespace
	                 WHERE n.nspname = 'pml'
		               AND c.relname = 'mv_pymc_kalman_pt_v2'
		               AND a.attnum > 0
		               AND NOT a.attisdropped)
	SELECT string_agg(m.response_column, ', ' ORDER BY m.trail_rank)
	INTO v_missing
	FROM pml.vw_pymc_trail_days m
	WHERE NOT EXISTS (SELECT 1 FROM mv_cols WHERE col = m.response_column);

	WITH mv_cols AS (SELECT a.attname::TEXT AS col
	                 FROM pg_attribute a
		                      JOIN pg_class c ON c.oid = a.attrelid
		                      JOIN pg_namespace n ON n.oid = c.relnamespace
	                 WHERE n.nspname = 'pml'
		               AND c.relname = 'mv_pymc_kalman_pt_v2'
		               AND a.attnum > 0
		               AND NOT a.attisdropped
		               AND a.attname::TEXT LIKE 'feat\_log\_uplift\_%')
	SELECT string_agg(col, ', ' ORDER BY col)
	INTO v_extra
	FROM mv_cols
	WHERE NOT EXISTS (SELECT 1 FROM pml.vw_pymc_trail_days m WHERE m.response_column = col);

	IF v_missing IS NOT NULL THEN
		RAISE EXCEPTION 'vw_pymc_trail_days maps response column(s) mv_pymc_kalman_pt_v2 does not emit: %', v_missing USING HINT =
				'Add the feat_log_uplift_* column to the MV, or drop the row from the trail-days map. The model builds its OU kernel x-axis from this view.';
	END IF;

	IF v_extra IS NOT NULL THEN
		RAISE EXCEPTION 'mv_pymc_kalman_pt_v2 emits response column(s) absent from vw_pymc_trail_days: %', v_extra USING HINT =
				'Every feat_log_uplift_* trail needs a calendar offset in pml.vw_pymc_trail_days or the model cannot place it on the OU grid.';
	END IF;
END;
$$
;

alter function pml.assert_pymc_trail_days_map() owner to postgres
;