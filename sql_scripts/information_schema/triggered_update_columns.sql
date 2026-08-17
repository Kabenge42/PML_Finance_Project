create view information_schema.triggered_update_columns
			(trigger_catalog, trigger_schema, trigger_name, event_object_catalog, event_object_schema,
			 event_object_table, event_object_column)
as
-- missing source code
;

alter table information_schema.triggered_update_columns
	owner to postgres
;

grant select on information_schema.triggered_update_columns to public
;