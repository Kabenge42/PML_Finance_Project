CREATE VIEW information_schema.triggered_update_columns
			(trigger_catalog, trigger_schema, trigger_name, event_object_catalog, event_object_schema,
			 event_object_table, event_object_column)
AS
-- missing source code
;

ALTER TABLE information_schema.triggered_update_columns
	OWNER TO postgres;

GRANT SELECT ON information_schema.triggered_update_columns TO PUBLIC;