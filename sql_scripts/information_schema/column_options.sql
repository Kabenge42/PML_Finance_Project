CREATE VIEW information_schema.column_options
			(table_catalog, table_schema, table_name, column_name, option_name, option_value) AS
SELECT current_database()::information_schema.sql_identifier                                AS table_catalog,
       nspname::information_schema.sql_identifier                                           AS table_schema,
       relname::information_schema.sql_identifier                                           AS table_name,
       attname::information_schema.sql_identifier                                           AS column_name,
       (pg_options_to_table(attfdwoptions)).option_name::information_schema.sql_identifier  AS option_name,
       (pg_options_to_table(attfdwoptions)).option_value::information_schema.character_data AS option_value
FROM information_schema._pg_foreign_table_columns c;

ALTER TABLE information_schema.column_options
	OWNER TO postgres;

GRANT SELECT ON information_schema.column_options TO PUBLIC;