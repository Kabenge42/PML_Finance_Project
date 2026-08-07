CREATE VIEW information_schema.foreign_table_options
			(foreign_table_catalog, foreign_table_schema, foreign_table_name, option_name, option_value) AS
SELECT foreign_table_catalog,
       foreign_table_schema,
       foreign_table_name,
       (pg_options_to_table(ftoptions)).option_name::information_schema.sql_identifier  AS option_name,
       (pg_options_to_table(ftoptions)).option_value::information_schema.character_data AS option_value
FROM information_schema._pg_foreign_tables t;

ALTER TABLE information_schema.foreign_table_options
	OWNER TO postgres;

GRANT SELECT ON information_schema.foreign_table_options TO PUBLIC;