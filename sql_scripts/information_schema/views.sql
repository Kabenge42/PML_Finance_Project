CREATE VIEW information_schema.views
			(table_catalog, table_schema, table_name, view_definition, check_option, is_updatable, is_insertable_into,
			 is_trigger_updatable, is_trigger_deletable, is_trigger_insertable_into)
AS
-- missing source code
;

ALTER TABLE information_schema.views
	OWNER TO postgres;

GRANT SELECT ON information_schema.views TO PUBLIC;