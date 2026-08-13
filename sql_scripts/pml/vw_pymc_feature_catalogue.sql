create view pml.vw_pymc_feature_catalogue
			(model_target, pymc_role, column_name, category, feature_role, feature_alias, data_type, description)
as
SELECT m.model_name                                                 AS model_target,
       COALESCE(fa.pymc_role, md.pymc_role)                         AS pymc_role,
       md.column_name,
       md.category,
       md.feature_role,
       COALESCE(fa.feature_alias, md.feature_alias, md.column_name) AS feature_alias,
       md.data_type,
       md.description
FROM pml.pml_df_metadata                             md
	     CROSS JOIN LATERAL unnest(md.model_targets) m(model_name)
	     LEFT JOIN  pml.pml_df_feature_alias         fa
	                ON fa.column_name = md.column_name AND fa.model_target = m.model_name
WHERE COALESCE(fa.pymc_role, md.pymc_role) <> 'excluded'::text
;

alter table pml.vw_pymc_feature_catalogue
	owner to postgres
;