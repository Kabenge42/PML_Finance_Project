CREATE VIEW information_schema.routine_sequence_usage
			(specific_catalog, specific_schema, specific_name, routine_catalog, routine_schema, routine_name,
			 sequence_catalog, sequence_schema, sequence_name)
AS
-- missing source code
;

ALTER TABLE information_schema.routine_sequence_usage
	OWNER TO postgres;

GRANT SELECT ON information_schema.routine_sequence_usage TO PUBLIC;