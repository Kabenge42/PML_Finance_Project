-- Piotroski F-score (0-9): scalar 9-signal fundamental-quality composite.
-- One point each for: positive ROA, positive CFO, rising ROA, accruals quality
-- (CFO > net income), falling long-term-debt/equity, rising current ratio,
-- no share dilution (shares out <=), rising gross margin, rising asset turnover.
-- NULL-tolerant: a NULL comparison scores 0 for that signal (never NULL overall).
-- Called 4x per row by pml.mv_pymc_kalman_pt (fy vs neg1fy .. neg3fy vs neg4fy
-- lag pairs) and by pml.calc_piotroski_f_score (LTM screener variant).

CREATE OR REPLACE FUNCTION pml.piotroski_f_score(roa       NUMERIC,
                                                 roa_prev  NUMERIC,
                                                 cfo       NUMERIC,
                                                 ni        NUMERIC,
                                                 ltde      NUMERIC,
                                                 ltde_prev NUMERIC,
                                                 cr        NUMERIC,
                                                 cr_prev   NUMERIC,
                                                 shrs      NUMERIC,
                                                 shrs_prev NUMERIC,
                                                 gpm       NUMERIC,
                                                 gpm_prev  NUMERIC,
                                                 at        NUMERIC,
                                                 at_prev   NUMERIC) RETURNS INTEGER
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (CASE WHEN roa > 0 THEN 1 ELSE 0 END +
        CASE WHEN cfo > 0 THEN 1 ELSE 0 END +
        CASE WHEN roa > roa_prev THEN 1 ELSE 0 END +
        CASE WHEN cfo > ni THEN 1 ELSE 0 END +
        CASE WHEN ltde < ltde_prev THEN 1 ELSE 0 END +
        CASE WHEN cr > cr_prev THEN 1 ELSE 0 END +
        CASE WHEN shrs <= shrs_prev THEN 1 ELSE 0 END +
        CASE WHEN gpm > gpm_prev THEN 1 ELSE 0 END +
        CASE WHEN at > at_prev THEN 1 ELSE 0 END)::INTEGER;
$$;

ALTER FUNCTION pml.piotroski_f_score(NUMERIC, NUMERIC, NUMERIC, NUMERIC, NUMERIC, NUMERIC, NUMERIC, NUMERIC, NUMERIC, NUMERIC, NUMERIC, NUMERIC, NUMERIC, NUMERIC) OWNER TO postgres;

CREATE OR REPLACE FUNCTION pml.piotroski_f_score(roa       DOUBLE PRECISION,
                                                 roa_prev  DOUBLE PRECISION,
                                                 cfo       DOUBLE PRECISION,
                                                 ni        DOUBLE PRECISION,
                                                 ltde      DOUBLE PRECISION,
                                                 ltde_prev DOUBLE PRECISION,
                                                 cr        DOUBLE PRECISION,
                                                 cr_prev   DOUBLE PRECISION,
                                                 shrs      DOUBLE PRECISION,
                                                 shrs_prev DOUBLE PRECISION,
                                                 gpm       DOUBLE PRECISION,
                                                 gpm_prev  DOUBLE PRECISION,
                                                 at        DOUBLE PRECISION,
                                                 at_prev   DOUBLE PRECISION) RETURNS INTEGER
	IMMUTABLE PARALLEL SAFE
	LANGUAGE sql AS
$$
SELECT (CASE WHEN roa > 0 THEN 1 ELSE 0 END +
        CASE WHEN cfo > 0 THEN 1 ELSE 0 END +
        CASE WHEN roa > roa_prev THEN 1 ELSE 0 END +
        CASE WHEN cfo > ni THEN 1 ELSE 0 END +
        CASE WHEN ltde < ltde_prev THEN 1 ELSE 0 END +
        CASE WHEN cr > cr_prev THEN 1 ELSE 0 END +
        CASE WHEN shrs <= shrs_prev THEN 1 ELSE 0 END +
        CASE WHEN gpm > gpm_prev THEN 1 ELSE 0 END +
        CASE WHEN at > at_prev THEN 1 ELSE 0 END)::INTEGER;
$$;

ALTER FUNCTION pml.piotroski_f_score(DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION) OWNER TO postgres;
