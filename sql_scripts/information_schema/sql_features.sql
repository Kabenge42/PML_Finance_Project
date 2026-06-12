CREATE TABLE information_schema.sql_features
(
	feature_id       information_schema.character_data,
	feature_name     information_schema.character_data,
	sub_feature_id   information_schema.character_data,
	sub_feature_name information_schema.character_data,
	is_supported     information_schema.yes_or_no,
	is_verified_by   information_schema.character_data,
	comments         information_schema.character_data
);

ALTER TABLE information_schema.sql_features
	OWNER TO postgres;

GRANT SELECT ON information_schema.sql_features TO PUBLIC;