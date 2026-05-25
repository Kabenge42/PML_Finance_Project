CREATE VIEW pml.vw_pymc_feature_aliases
			(model_target, feature_aliases, observed_aliases, constant_data_aliases, coord_aliases) AS
SELECT model_target,
       array_agg(column_name ORDER BY column_name)
       FILTER (WHERE pymc_role = 'mutable_predictor'::text)                                         AS feature_aliases,
       array_agg(column_name ORDER BY column_name) FILTER (WHERE pymc_role = 'observed'::text)      AS observed_aliases,
       array_agg(column_name ORDER BY column_name)
       FILTER (WHERE pymc_role = 'constant_data'::text)                                             AS constant_data_aliases,
       array_agg(column_name ORDER BY column_name) FILTER (WHERE pymc_role = 'coord'::text)         AS coord_aliases
FROM vw_pymc_feature_catalogue
GROUP BY model_target;

ALTER TABLE pml.vw_pymc_feature_aliases
	OWNER TO postgres;