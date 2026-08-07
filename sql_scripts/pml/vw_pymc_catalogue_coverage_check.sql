CREATE VIEW vw_pymc_catalogue_coverage_check(model_target, feat_name, catalogue_rows, status) AS
WITH mv_map(mv_name, model_target) AS (VALUES ('mv_pymc_earnings_beat'::text, 'earnings_beat'::text),
                                              ('mv_pymc_price_target'::text, 'price_target'::text),
                                              ('mv_pymc_kalman_pt'::text, 'kalman_pt'::text),
                                              ('mv_pymc_dcf_pt'::text, 'dcf_pt'::text),
                                              ('mv_pymc_dividend_safety'::text, 'dividend_safety'::text),
                                              ('mv_pymc_credit_risk'::text, 'credit_risk'::text),
                                              ('mv_pymc_accounting_anomaly'::text, 'accounting_anomaly'::text)
                                      ),
     mv_cols                       AS (SELECT mm.model_target, a.attname::text AS feat_name
                                       FROM mv_map                mm
	                                            JOIN pg_class     cl ON cl.relname = mm.mv_name AND cl.relkind = 'm'::"char"
	                                            JOIN pg_namespace ns ON ns.oid = cl.relnamespace AND ns.nspname = 'pml'::name
	                                            JOIN pg_attribute a
	                                                 ON a.attrelid = cl.oid AND a.attnum > 0 AND NOT a.attisdropped
                                       WHERE a.attname ~~ 'feat\_%'::text
	                                      OR a.attname ~~ 'observed\_%'::text
	                                      OR a.attname ~~ 'n\_%'::text
                                      ),
     cat                           AS (SELECT vw_pymc_feature_catalogue.model_target,
                                              vw_pymc_feature_catalogue.feature_alias,
                                              count(*) AS n_rows
                                       FROM vw_pymc_feature_catalogue
                                       WHERE vw_pymc_feature_catalogue.feature_alias ~~ 'feat\_%'::text
	                                      OR vw_pymc_feature_catalogue.feature_alias ~~ 'observed\_%'::text
	                                      OR vw_pymc_feature_catalogue.feature_alias ~~ 'n\_%'::text
                                       GROUP BY vw_pymc_feature_catalogue.model_target,
                                                vw_pymc_feature_catalogue.feature_alias
                                      )
SELECT mc.model_target,
       mc.feat_name,
       COALESCE(cat.n_rows, 0::bigint) AS catalogue_rows,
       CASE
	       WHEN cat.n_rows IS NULL THEN 'MISSING_FROM_CATALOGUE'::text
	       WHEN cat.n_rows > 1 THEN 'DUPLICATE_CATALOGUE_ALIAS'::text
	       ELSE 'OK'::text END         AS status
FROM mv_cols mc
	     LEFT JOIN cat ON cat.model_target = mc.model_target AND cat.feature_alias = mc.feat_name
UNION ALL
SELECT c.model_target, c.feature_alias AS feat_name, 0 AS catalogue_rows, 'PHANTOM_CATALOGUE_ALIAS'::text AS status
FROM cat                   c
	     LEFT JOIN mv_cols mc ON mc.model_target = c.model_target AND mc.feat_name = c.feature_alias
WHERE mc.feat_name IS NULL;

ALTER TABLE vw_pymc_catalogue_coverage_check
	OWNER TO postgres;