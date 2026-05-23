CREATE VIEW vw_pymc_feature_coverage(model_target, pymc_role, n_columns) AS
SELECT model_target, pymc_role, count(*) AS n_columns
FROM vw_pymc_feature_catalogue
GROUP BY model_target, pymc_role
ORDER BY model_target, pymc_role;

ALTER TABLE vw_pymc_feature_coverage
	OWNER TO postgres;