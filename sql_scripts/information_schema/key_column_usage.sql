CREATE VIEW information_schema.key_column_usage
			(constraint_catalog, constraint_schema, constraint_name, table_catalog, table_schema, table_name,
			 column_name, ordinal_position, position_in_unique_constraint)
AS
-- missing source code
;

ALTER TABLE information_schema.key_column_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.key_column_usage TO PUBLIC;