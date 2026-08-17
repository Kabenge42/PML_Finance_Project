create view information_schema.triggers
			(trigger_catalog, trigger_schema, trigger_name, event_manipulation, event_object_catalog,
			 event_object_schema, event_object_table, action_order, action_condition, action_statement,
			 action_orientation, action_timing, action_reference_old_table, action_reference_new_table,
			 action_reference_old_row, action_reference_new_row, created)
as
-- missing source code
;

alter table information_schema.triggers
	owner to postgres
;

grant select on information_schema.triggers to public
;